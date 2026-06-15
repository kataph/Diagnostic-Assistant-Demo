"""
Non-regression tests for PSU-short + discharged-battery double fault.

Faults injected directly (not via SCENARIOS — this scenario was not in the
SCENARIOS_MASTER.csv catalogue):
  1. PSU output cables shorted together (ShortCircuit fault)
  2. Battery discharged (voltage=0.0)

Key invariants tested:
  - Repairing ONLY the battery does NOT restore the lamp (short still present).
  - Repairing ONLY the cables does NOT restore the lamp (battery still dead).
  - Repairing BOTH restores the lamp.
  - After test_repair, the circuit is restored to the fault state (no side-effects).
  - Short graph element is removed during repair and re-inserted on snapshot restore.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

import pytest
from diagnosable_systems_simulation.systems.ten_cubes.factory import build_ten_cubes_system
from diagnosable_systems_simulation.actions.fault_actions import ShortCircuit, DegradeComponent


def _apply(sys_, action, targets):
    result = sys_.apply_action(action, targets)
    assert result.success, f"Fault injection failed: {result.message}"


@pytest.fixture
def faulted_system():
    """10-cubes system with PSU-short + discharged-battery faults applied directly."""
    sys_ = build_ten_cubes_system(extra_tools={"multimeter"})
    sys_.simulate()
    sys_._nominal_emitting_light = sys_.last_result.emitting_light

    cable_pos = sys_.component("psu_cable_pos")
    cable_neg = sys_.component("psu_cable_neg")
    psu_pos_node = cable_pos.port("p").node_id
    gnd_node = cable_neg.port("p").node_id
    _apply(sys_, ShortCircuit(psu_pos_node, gnd_node, "psu_output_short"),
           {"start": cable_pos, "end": cable_neg})
    _apply(sys_, DegradeComponent({"voltage": 0.0}), {"subject": sys_.component("battery")})

    sys_._fault_snapshot = sys_.snapshot()
    return sys_


def _lamp_on(system) -> bool:
    result = system.simulate()
    bulb_ids = {
        cid for cid, c in system.all_components().items()
        if type(c).__name__ == "Bulb"
    }
    return bool(result.emitting_light & bulb_ids)


def _node_snapshot(system):
    return {cid: {p.name: p.node_id for p in c.ports}
            for cid, c in system.all_components().items()}


class TestScenario14Verification:
    def test_lamp_off_in_fault_state(self, faulted_system):
        """Sanity: the lamp is off when both faults are active."""
        assert not _lamp_on(faulted_system)

    def test_battery_repair_alone_insufficient(self, faulted_system):
        """Repairing only the battery does NOT restore the lamp (short still active)."""
        lamp = faulted_system.test_repair({"battery"})
        assert not lamp, "Battery repair alone should not fix the lamp while the short is still present"

    def test_cable_repair_alone_insufficient(self, faulted_system):
        """Repairing only the shorted cables does NOT restore the lamp (battery still dead)."""
        lamp = faulted_system.test_repair({"psu_cable_pos", "psu_cable_neg"})
        assert not lamp, "Cable repair alone should not fix the lamp while the battery is still discharged"

    def test_both_repairs_restore_lamp(self, faulted_system):
        """Repairing both faults restores the lamp."""
        lamp = faulted_system.test_repair({"battery", "psu_cable_pos", "psu_cable_neg"})
        assert lamp, "Repairing both battery and shorted cables should restore the lamp"

    def test_no_side_effects_after_battery_test(self, faulted_system):
        """test_repair leaves the circuit in the fault state (no persistent changes)."""
        before = _node_snapshot(faulted_system)
        faulted_system.test_repair({"battery"})
        after = _node_snapshot(faulted_system)
        assert before == after

    def test_no_side_effects_after_cable_test(self, faulted_system):
        """test_repair on cables leaves circuit unchanged (short restored)."""
        before = _node_snapshot(faulted_system)
        faulted_system.test_repair({"psu_cable_pos", "psu_cable_neg"})
        after = _node_snapshot(faulted_system)
        assert before == after

    def test_lamp_still_off_after_test_repair(self, faulted_system):
        """After any test_repair call, the circuit is still in fault state."""
        faulted_system.test_repair({"battery", "psu_cable_pos", "psu_cable_neg"})
        assert not _lamp_on(faulted_system), "After test_repair, fault state must be restored"

    def test_short_graph_element_restored_after_cable_test(self, faulted_system):
        """
        After test_repair on the shorted cables, the short wire must be back in the
        circuit so subsequent simulations reflect the real fault.
        """
        result_before = faulted_system.simulate()
        faulted_system.test_repair({"psu_cable_pos", "psu_cable_neg"})
        result_after = faulted_system.simulate()
        assert result_before.emitting_light == result_after.emitting_light, \
            "Short circuit must be re-inserted after test_repair restores the snapshot"
