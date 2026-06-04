"""
Tests for scenario 15 — ambient light sensor feedback loop.

Scenario 15 fault: the modules are stacked so the lamp illuminates the
light sensor.  The feedback loop (lamp ON → sensor lit → relay open →
lamp OFF → sensor dark → relay closed → lamp ON → …) never stabilises,
causing the simulation to return ``converged=False``.

The fix is to rotate/move either enclosure so the optical path is broken.
``RotateEnclosure`` on ``cube_ctrl`` or ``cube_load`` sets ``is_inverted = True``
on the chosen enclosure.  ``AmbientFeedbackCoupling._feedback_blocked()`` then
returns True, the sensor stays dark, the relay stays closed, and the lamp
converges stably ON.

Tests
-----
test_fault_causes_non_convergence
    The feedback loop makes the simulation diverge (converged=False).

test_rotate_ctrl_cube_repairs_system
    Rotating cube_ctrl breaks the optical path → lamp is stably ON.

test_rotate_load_cube_repairs_system
    Rotating cube_load breaks the optical path → lamp is stably ON.

test_repair_is_idempotent_ctrl
    Inverting cube_ctrl twice leaves the lamp still on.

test_restore_ctrl_cube_re_enables_fault
    After restoring cube_ctrl to normal orientation the fault comes back
    (simulation diverges again), confirming the fix is reversible.

test_restore_load_cube_re_enables_fault
    Same as above for cube_load.

test_no_side_effects_on_circuit_wiring
    Rotating and restoring an enclosure leaves circuit node assignments
    unchanged (no cable reconnections or graph edits).

test_repair_ctrl_enclosure_via_test_repair
    test_repair({"cube_ctrl"}) returns True — the PhysicalEnclosure branch
    in test_repair repositions the enclosure and checks the lamp.

test_repair_load_enclosure_via_test_repair
    test_repair({"cube_load"}) also returns True.

test_test_repair_enclosure_has_no_side_effects
    After test_repair on an enclosure the system is back in fault state:
    is_inverted is reset to False and the simulation diverges again.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

import pytest
from diagnosable_systems_simulation.systems.ambient_light_sensor.factory import build_ambient_light_system
from diagnosable_systems_simulation.actions.diagnostic_actions import RotateEnclosure, RestoreEnclosure
from Implementations.fault_injections import SCENARIOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_scenario15():
    return next(s for s in SCENARIOS if s.id == 15)


def _node_snapshot(system) -> dict:
    return {
        cid: {p.name: p.node_id for p in c.ports}
        for cid, c in system.all_components().items()
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def nominal_system():
    """Fresh ALS system with no fault applied."""
    sys_ = build_ambient_light_system(extra_tools={"multimeter"})
    sys_.simulate()
    sys_._nominal_emitting_light = sys_.last_result.emitting_light
    return sys_


@pytest.fixture
def faulted_system():
    """ALS system with scenario-15 fault applied (als_feedback=True)."""
    sys_ = build_ambient_light_system(extra_tools={"multimeter"})
    sys_.simulate()
    sys_._nominal_emitting_light = sys_.last_result.emitting_light
    scenario = _get_scenario15()
    for fn in scenario.fault_fns:
        fn(sys_)
    sys_._fault_snapshot = sys_.snapshot()
    return sys_


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestScenario15ALSRepair:

    def test_lamp_on_in_nominal_state(self, nominal_system):
        """Sanity: lamp is stably ON before any fault is injected."""
        result = nominal_system.simulate()
        assert result.converged, "Nominal system must converge"
        assert result.emitting_light >= nominal_system._nominal_emitting_light, \
            "Lamp must be on in nominal state"

    def test_fault_causes_non_convergence(self, faulted_system):
        """The feedback loop makes the simulation diverge (converged=False)."""
        result = faulted_system.simulate()
        assert not result.converged, \
            "Faulted ALS system must not converge (flickering lamp)"

    def test_rotate_ctrl_cube_repairs_system(self, faulted_system):
        """Rotating cube_ctrl breaks the optical path → lamp converges ON."""
        cube_ctrl = faulted_system.component("cube_ctrl")
        inv_result = faulted_system.apply_action(RotateEnclosure(), {"subject": cube_ctrl})
        assert inv_result.success, f"InvertEnclosure on cube_ctrl failed: {inv_result.message}"

        result = faulted_system.simulate()
        assert result.converged, "System must converge after rotating cube_ctrl"
        assert result.emitting_light >= faulted_system._nominal_emitting_light, \
            "Lamp must be on after rotating cube_ctrl"

    def test_rotate_load_cube_repairs_system(self, faulted_system):
        """Rotating cube_load breaks the optical path → lamp converges ON."""
        cube_load = faulted_system.component("cube_load")
        inv_result = faulted_system.apply_action(RotateEnclosure(), {"subject": cube_load})
        assert inv_result.success, f"InvertEnclosure on cube_load failed: {inv_result.message}"

        result = faulted_system.simulate()
        assert result.converged, "System must converge after rotating cube_load"
        assert result.emitting_light >= faulted_system._nominal_emitting_light, \
            "Lamp must be on after rotating cube_load"

    def test_repair_is_idempotent_ctrl(self, faulted_system):
        """Inverting cube_ctrl a second time (restore then invert again) keeps lamp on."""
        cube_ctrl = faulted_system.component("cube_ctrl")
        faulted_system.apply_action(RotateEnclosure(), {"subject": cube_ctrl})
        faulted_system.apply_action(RestoreEnclosure(), {"subject": cube_ctrl})
        faulted_system.apply_action(RotateEnclosure(), {"subject": cube_ctrl})

        result = faulted_system.simulate()
        assert result.converged
        assert result.emitting_light >= faulted_system._nominal_emitting_light

    def test_restore_ctrl_cube_re_enables_fault(self, faulted_system):
        """Restoring cube_ctrl re-enables the feedback loop (fault comes back)."""
        cube_ctrl = faulted_system.component("cube_ctrl")
        faulted_system.apply_action(RotateEnclosure(), {"subject": cube_ctrl})
        # Verify it's fixed first.
        assert faulted_system.simulate().converged

        faulted_system.apply_action(RestoreEnclosure(), {"subject": cube_ctrl})
        result = faulted_system.simulate()
        assert not result.converged, \
            "Restoring cube_ctrl must re-enable the feedback loop (diverges again)"

    def test_restore_load_cube_re_enables_fault(self, faulted_system):
        """Restoring cube_load re-enables the feedback loop (fault comes back)."""
        cube_load = faulted_system.component("cube_load")
        faulted_system.apply_action(RotateEnclosure(), {"subject": cube_load})
        assert faulted_system.simulate().converged

        faulted_system.apply_action(RestoreEnclosure(), {"subject": cube_load})
        result = faulted_system.simulate()
        assert not result.converged, \
            "Restoring cube_load must re-enable the feedback loop (diverges again)"

    def test_no_side_effects_on_circuit_wiring(self, faulted_system):
        """Rotating and restoring an enclosure leaves circuit node assignments unchanged."""
        before = _node_snapshot(faulted_system)

        cube_ctrl = faulted_system.component("cube_ctrl")
        faulted_system.apply_action(RotateEnclosure(), {"subject": cube_ctrl})
        faulted_system.simulate()
        faulted_system.apply_action(RestoreEnclosure(), {"subject": cube_ctrl})

        after = _node_snapshot(faulted_system)
        assert before == after, \
            "Rotating and restoring an enclosure must not alter circuit node assignments"

    def test_repair_ctrl_enclosure_via_test_repair(self, faulted_system):
        """test_repair({"cube_ctrl"}) returns True via the PhysicalEnclosure branch."""
        lamp_on = faulted_system.test_repair({"cube_ctrl"})
        assert lamp_on, "test_repair on cube_ctrl must report the lamp as restored"

    def test_repair_load_enclosure_via_test_repair(self, faulted_system):
        """test_repair({"cube_load"}) returns True via the PhysicalEnclosure branch."""
        lamp_on = faulted_system.test_repair({"cube_load"})
        assert lamp_on, "test_repair on cube_load must report the lamp as restored"

    def test_test_repair_enclosure_has_no_side_effects(self, faulted_system):
        """After test_repair on an enclosure the system is back in fault state."""
        cube_ctrl = faulted_system.component("cube_ctrl")

        faulted_system.test_repair({"cube_ctrl"})

        assert not cube_ctrl.is_inverted, \
            "test_repair must reset is_inverted to False after the check"
        result = faulted_system.simulate()
        assert not result.converged, \
            "After test_repair the fault must still be active (simulation diverges)"

    # ---------------------------------------------------------------------- #
    # verify_hypothesis persistence mechanic                                   #
    # ---------------------------------------------------------------------- #

    def test_persistence_mechanic_ctrl_enclosure(self, faulted_system):
        """
        Simulate what verify_hypothesis does to persist an enclosure repair:
        1. test_repair({"cube_ctrl"}) returns True.
        2. apply_repairs is a no-op for PhysicalEnclosure (no fault overlay),
           but we manually set is_inverted=True to mirror the persistence code.
        3. restore_snapshot(exclude_ids={"cube_ctrl"}) must leave cube_ctrl
           with is_inverted=True (excluded from restore).
        4. The simulation must now converge with the lamp on.
        """
        snap = faulted_system._fault_snapshot
        cube_ctrl = faulted_system.component("cube_ctrl")

        lamp_on = faulted_system.test_repair({"cube_ctrl"})
        assert lamp_on, "test_repair must confirm the enclosure repair"

        # Mirror the persistence block in verify_hypothesis:
        cube_ctrl.is_inverted = True
        faulted_system.restore_snapshot(snap, exclude_ids={"cube_ctrl"})

        assert cube_ctrl.is_inverted, \
            "restore_snapshot(exclude_ids={'cube_ctrl'}) must leave is_inverted=True"
        result = faulted_system.simulate()
        assert result.converged, \
            "System must converge after persistent enclosure repair"
        assert result.emitting_light >= faulted_system._nominal_emitting_light, \
            "Lamp must be on after persistent enclosure repair"

    def test_persistence_mechanic_load_enclosure(self, faulted_system):
        """Same persistence mechanic applied to cube_load instead of cube_ctrl."""
        snap = faulted_system._fault_snapshot
        cube_load = faulted_system.component("cube_load")

        lamp_on = faulted_system.test_repair({"cube_load"})
        assert lamp_on, "test_repair must confirm the enclosure repair"

        cube_load.is_inverted = True
        faulted_system.restore_snapshot(snap, exclude_ids={"cube_load"})

        assert cube_load.is_inverted, \
            "restore_snapshot(exclude_ids={'cube_load'}) must leave is_inverted=True"
        result = faulted_system.simulate()
        assert result.converged, \
            "System must converge after persistent enclosure repair"
        assert result.emitting_light >= faulted_system._nominal_emitting_light, \
            "Lamp must be on after persistent enclosure repair"

    def test_repair_potentiometers_does_not_solve_scenario(self, faulted_system):
        """
        Replacing the potentiometers must NOT fix the scenario.

        The potentiometers are bait components — they have no role in the
        feedback loop.  test_repair on them must return False because the
        simulation still does not converge (lamp state is ambiguous) even
        after the pots are "repaired".
        """
        lamp_on = faulted_system.test_repair(
            {"ctrl_sensitivity_pot", "ctrl_timing_pot"}
        )
        assert not lamp_on, (
            "test_repair on the potentiometers must return False: "
            "the feedback oscillation is caused by the optical path, "
            "not by the potentiometers"
        )

    def test_without_persistence_enclosure_resets(self, faulted_system):
        """
        Without the persistence step (no exclude_ids), restore_snapshot resets
        is_inverted to False and the fault returns.
        """
        snap = faulted_system._fault_snapshot
        cube_ctrl = faulted_system.component("cube_ctrl")

        faulted_system.test_repair({"cube_ctrl"})

        # Intentionally set is_inverted but do NOT exclude cube_ctrl from restore.
        cube_ctrl.is_inverted = True
        faulted_system.restore_snapshot(snap)   # no exclude_ids

        assert not cube_ctrl.is_inverted, \
            "Without exclude_ids restore_snapshot must reset is_inverted to False"
        result = faulted_system.simulate()
        assert not result.converged, \
            "Without exclude_ids the fault must be active again after restore_snapshot"
