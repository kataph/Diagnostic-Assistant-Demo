"""
Bypass-guard regression test for test_repair().

Scenario
--------
1. A cable is disconnected from the ctrl_switch (fault injection) so the
   switch is isolated from the positive rail and the lamp is OFF.
2. The control module is bypassed by inserting two low-resistance shorts
   that connect the PSU positive output node directly to the load positive
   input node, and similarly for the negative (0 V) rail.
   This short is NOT captured in the fault_snapshot, so restore_snapshot()
   does not remove it.
3. The lamp is now ON (driven by the bypass, not the switch).

Test assertion
--------------
test_repair({"battery"}) — an unrelated component with no fault — must return
False.

Without the bypass guard the switch-independence of the lamp would be
overlooked and test_repair would return True (false positive).
With the bypass guard, test_repair opens the ctrl_switch, finds the lamp is
STILL on (bypass provides an alternate current path), and correctly returns
False.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

import pytest
from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system


class TestBypassGuard:

    @staticmethod
    def _disconnect_cable_port(sys_, cable_id: str, port_name: str) -> None:
        """
        Disconnect a cable port and record the original node in _orig_connections
        so that apply_repairs() can reconnect it later.
        """
        cable = sys_.component(cable_id)
        original_node = cable.port(port_name).node_id
        cable._orig_connections = {port_name: original_node}
        sys_._graph.disconnect_port(cable_id, port_name)

    @pytest.fixture
    def faulted_system(self):
        """
        3-cubes system with ctrl_cable_in_pos.n disconnected (fault only, no bypass).
        Used to verify that test_repair still works correctly in the absence of a bypass.
        """
        sys_ = build_three_cubes_system(extra_tools={"multimeter"})
        nominal = sys_.simulate()
        sys_._nominal_emitting_light = nominal.emitting_light
        self._disconnect_cable_port(sys_, "ctrl_cable_in_pos", "n")
        sys_._fault_snapshot = sys_.snapshot()
        return sys_

    @pytest.fixture
    def bypassed_system(self):
        """
        3-cubes system with:
        - ctrl_cable_in_pos.n disconnected from the switch (fault)
        - A direct PSU+ → load+ short (and PSU− → load− short) bypassing
          the whole control module
        - fault_snapshot taken after the cable disconnect but before the short
        """
        sys_ = build_three_cubes_system(extra_tools={"multimeter"})
        # Nominal run — record which bulbs must be lit for success
        nominal = sys_.simulate()
        sys_._nominal_emitting_light = nominal.emitting_light

        # ── Fault: disconnect the positive input cable from the switch ─────
        self._disconnect_cable_port(sys_, "ctrl_cable_in_pos", "n")

        # Fault snapshot taken here: circuit is broken, lamp is OFF
        sys_._fault_snapshot = sys_.snapshot()

        # ── Bypass: short PSU+ output → load+ input (bypass control module)─
        # These nodes are NOT in the snapshot so restore_snapshot() leaves
        # the shorts in place, which is exactly what we want to test.
        psu_pos_node  = sys_.component("ctrl_cable_in_pos").port("p").node_id
        load_pos_node = sys_.component("load_cable_pos").port("p").node_id
        psu_neg_node  = sys_.component("ctrl_cable_in_neg").port("p").node_id
        load_neg_node = sys_.component("load_cable_neg").port("p").node_id

        sys_._graph.short_nodes(psu_pos_node,  load_pos_node,  "_bypass_pos")
        sys_._graph.short_nodes(psu_neg_node,  load_neg_node,  "_bypass_neg")

        return sys_

    # ---------------------------------------------------------------------- #

    def test_lamp_is_on_with_bypass(self, bypassed_system):
        """Sanity: the bypass makes the lamp light up despite the disconnected cable."""
        result = bypassed_system.simulate()
        assert result.emitting_light >= bypassed_system._nominal_emitting_light, \
            "Bypass must make the lamp light up even with the switch cable disconnected"

    def test_repair_of_unrelated_component_fails(self, bypassed_system):
        """
        test_repair on an unrelated component (battery) must return False.

        The lamp is on only because of the bypass short, not because the
        switch is in the circuit.  The bypass guard opens the ctrl_switch and
        checks whether the lamp goes off; since the bypass keeps the lamp on,
        the guard correctly rejects the apparent 'lamp_on=True' and returns False.
        """
        lamp_on = bypassed_system.test_repair({"battery"})
        assert not lamp_on, (
            "test_repair must return False when the lamp is lit only by a "
            "diagnostic bypass, not by a genuine switch-controlled path"
        )

    def test_repair_of_switch_cable_succeeds(self, faulted_system):
        """
        Without a bypass, test_repair on the actually-faulted cable returns True.

        Reconnecting ctrl_cable_in_pos.n brings the switch back into the
        positive rail.  The bypass guard opens the switch and confirms the lamp
        goes off, so the repair is accepted as genuine.
        """
        lamp_on = faulted_system.test_repair({"ctrl_cable_in_pos"})
        assert lamp_on, (
            "test_repair must return True when the actually-faulted cable is "
            "reconnected and no bypass is present"
        )

    def test_lamp_off_in_fault_state(self, faulted_system):
        """Sanity: lamp is off in the plain fault state (no bypass)."""
        result = faulted_system.simulate()
        assert not result.emitting_light >= faulted_system._nominal_emitting_light, \
            "Lamp must be OFF in fault state before any bypass or repair"
