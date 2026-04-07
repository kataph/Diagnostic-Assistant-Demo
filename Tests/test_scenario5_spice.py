"""
Smoke test: scenario 5 on the three-cubes system (SPICE backend).

Scenario 5: the PSU→ctrl cables are crossed — ctrl_cable_in_pos.p is
wired to the ground net and ctrl_cable_in_neg.p is wired to the +12 V net.
This supplies reverse voltage to the control/load chain, turning the lamp off
and the red LED on.

Steps
-----
1. Build the three-cubes system and attach a stdout logger.
2. Simulate nominal state; capture _nominal_emitting_light.
3. Inject scenario-5 fault (cross the cables).
4. Simulate faulted state; verify lamp is OFF.
5. Snapshot the fault state (_fault_snapshot).
6. Verify crossed wiring: ctrl_cable_in_pos.p is on the GND node.
7. Call test_repair({ctrl_cable_in_pos, ctrl_cable_in_neg}) — both cables
   must be repaired together because the fault lives in both.
   Assert lamp_on is True.
8. Verify that after test_repair the circuit is back in the fault state
   (no persistent side-effects).

Run with:
    python -m pytest Tests/test_scenario5_spice.py -s
or:
    python Tests/test_scenario5_spice.py
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from Implementations.scenarios import SCENARIOS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stdout_logger(name: str = "SpiceRunner") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_scenario5_repair_crossed_cables():
    """
    After injecting scenario-5 (crossed PSU→ctrl cables), repairing both
    ctrl_cable_in_pos and ctrl_cable_in_neg must restore the lamp.
    """
    # 1. Build system and attach logger.
    sim = build_three_cubes_system(extra_tools={"multimeter"})
    sim.add_logger(_make_stdout_logger())

    # 2. Nominal simulation.
    nominal_result = sim.simulate()
    sim._nominal_emitting_light = nominal_result.emitting_light
    print(f"\n[nominal] emitting_light: {sorted(nominal_result.emitting_light)}")
    assert "main_bulb" in nominal_result.emitting_light, (
        "Main bulb should be lit in nominal state"
    )

    # Record the nominal node IDs for each cable's p port before crossing.
    in_pos = sim.component("ctrl_cable_in_pos")
    in_neg = sim.component("ctrl_cable_in_neg")
    nominal_in_pos_p_node = in_pos.port("p").node_id
    nominal_in_neg_p_node = in_neg.port("p").node_id

    # 3. Inject scenario-5 fault.
    scenario = next(s for s in SCENARIOS if s.id == 5)
    for fault_fn in scenario.fault_fns:
        fault_fn(sim)
    print(f"\n[fault injection] scenario {scenario.id}: "
          f"{scenario.root_cause.root_cause_description_proper}")

    # 4. Simulate faulted state — lamp should be off.
    fault_result = sim.simulate()
    print(f"[faulted] emitting_light: {sorted(fault_result.emitting_light)}")
    assert "main_bulb" not in fault_result.emitting_light, (
        "Main bulb should be OFF in the faulted state (reversed polarity)"
    )

    # 5. Snapshot the fault state.
    sim._fault_snapshot = sim.snapshot()

    # 6. Verify crossed wiring: in_pos.p should now be on the GND node,
    #    in_neg.p should be on the PSU+ node.
    assert in_pos.port("p").node_id == nominal_in_neg_p_node, (
        f"ctrl_cable_in_pos.p should be on GND node {nominal_in_neg_p_node!r} "
        f"after crossing, got {in_pos.port('p').node_id!r}"
    )
    assert in_neg.port("p").node_id == nominal_in_pos_p_node, (
        f"ctrl_cable_in_neg.p should be on PSU+ node {nominal_in_pos_p_node!r} "
        f"after crossing, got {in_neg.port('p').node_id!r}"
    )
    print(
        f"[check] cables crossed: in_pos.p={in_pos.port('p').node_id!r}, "
        f"in_neg.p={in_neg.port('p').node_id!r} ✓"
    )

    # 7. test_repair: temporarily uncross both cables and check the lamp.
    lamp_on = sim.test_repair({"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
    print(f"\n[test_repair] lamp_on={lamp_on}")
    assert lamp_on, (
        "Repairing both ctrl_cable_in_pos and ctrl_cable_in_neg should "
        "restore the lamp — test_repair returned False"
    )

    # 8. After test_repair the circuit must be back in the fault state.
    assert in_pos.port("p").node_id == nominal_in_neg_p_node, (
        "test_repair must restore the fault state — ctrl_cable_in_pos.p "
        "should be on GND node again"
    )
    assert in_neg.port("p").node_id == nominal_in_pos_p_node, (
        "test_repair must restore the fault state — ctrl_cable_in_neg.p "
        "should be on PSU+ node again"
    )
    print("[check] fault state restored after test_repair ✓")


def test_scenario5_partial_then_correct():
    """
    Repairing crossed cables one at a time must yield partial → correct,
    not wrong → correct.

    This validates that _is_wrong_node() in serviceAgentSpiceSim correctly
    marks crossed cables (connected but to the wrong node) as broken, so
    the first single-cable repair is recognised as a partial fix rather than
    being dismissed as wrong.

    The test bypasses the full service-agent stack and calls test_repair
    directly, simulating what verify_hypothesis does:

      Round 1: test_repair({"ctrl_cable_in_pos"}) → False  (lamp still off)
               _is_wrong_node(ctrl_cable_in_pos) after restore → True
               → outcome would be "partial"
               → apply_repairs({"ctrl_cable_in_pos"}) persists the fix

      Round 2: test_repair({"ctrl_cable_in_neg"},
                            already_repaired_ids={"ctrl_cable_in_pos"}) → True
               → outcome would be "correct"
    """
    from diagnosable_systems_simulation.world.components import Cable as _Cable

    def _is_wrong_node(comp) -> bool:
        if not isinstance(comp, _Cable):
            return False
        orig = getattr(comp, "_orig_connections", {})
        return any(
            p.is_connected()
            and orig.get(p.name) is not None
            and p.node_id != orig[p.name]
            for p in comp.ports
        )

    sim = build_three_cubes_system(extra_tools={"multimeter"})
    sim.add_logger(_make_stdout_logger("SpiceRunner2"))

    nominal_result = sim.simulate()
    sim._nominal_emitting_light = nominal_result.emitting_light

    scenario = next(s for s in SCENARIOS if s.id == 5)
    for fault_fn in scenario.fault_fns:
        fault_fn(sim)

    sim._fault_snapshot = sim.snapshot()
    sim.simulate()

    in_pos = sim.component("ctrl_cable_in_pos")
    in_neg = sim.component("ctrl_cable_in_neg")

    # --- Round 1: repair ctrl_cable_in_pos alone ---
    lamp_on = sim.test_repair({"ctrl_cable_in_pos"})
    print(f"\n[round 1] lamp_on={lamp_on}")
    assert not lamp_on, "Single-cable repair should not restore the lamp"

    # After test_repair the snapshot is restored: both cables are crossed again.
    broken_after_r1 = {
        cid for cid, c in sim.all_components().items()
        if _is_wrong_node(c)
    }
    print(f"[round 1] wrong-node cables after restore: {broken_after_r1}")
    assert "ctrl_cable_in_pos" in broken_after_r1, (
        "ctrl_cable_in_pos should still be detected as wrong-node after restore"
    )
    # The candidate intersects still_broken → would be "partial", not "wrong"
    actually_faulty_r1 = {"ctrl_cable_in_pos"} & broken_after_r1
    assert actually_faulty_r1, "Round 1 should yield 'partial' (actually_faulty non-empty)"
    print("[check] round 1 outcome would be 'partial' ✓")

    # Persist round-1 repair (mirrors apply_repairs in verify_hypothesis).
    sim.apply_repairs({"ctrl_cable_in_pos"})
    sim.restore_snapshot(sim._fault_snapshot, exclude_ids={"ctrl_cable_in_pos"})

    # ctrl_cable_in_pos should now be fixed; ctrl_cable_in_neg still crossed.
    assert not _is_wrong_node(in_pos), (
        "ctrl_cable_in_pos should be repaired after apply_repairs"
    )
    assert _is_wrong_node(in_neg), (
        "ctrl_cable_in_neg should still be crossed"
    )

    # --- Round 2: repair ctrl_cable_in_neg, with in_pos already repaired ---
    lamp_on = sim.test_repair(
        {"ctrl_cable_in_neg"},
        already_repaired_ids={"ctrl_cable_in_pos"},
    )
    print(f"\n[round 2] lamp_on={lamp_on}")
    assert lamp_on, (
        "Repairing ctrl_cable_in_neg (with ctrl_cable_in_pos already fixed) "
        "should restore the lamp"
    )
    print("[check] round 2 outcome would be 'correct' ✓")


if __name__ == "__main__":
    test_scenario5_repair_crossed_cables()
    test_scenario5_partial_then_correct()