"""
Smoke test: scenario 10 on the ten-cubes system (SPICE backend).

Scenario 10: ctrl6_cable_in_pos is disconnected AND all control-module
green LEDs (and their indicator resistors) are physically removed.

Steps
-----
1. Build the ten-cubes system and attach a stdout logger (prints each netlist).
2. Simulate nominal state; capture _nominal_emitting_light.
3. Inject scenario-10 faults.
4. Snapshot the fault state (_fault_snapshot).
5. Verify that ctrl6_cable_in_pos.n is floating (fault present).
6. Call test_repair({ctrl6_cable_in_pos}) and assert lamp_on is True.
7. Verify that after test_repair the circuit is back in the fault state
   (port is floating again — no persistent side-effects).

Run with:
    python -m pytest Tests/test_scenario10_spice.py -s
or:
    python Tests/test_scenario10_spice.py
"""
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diagnosable_systems_simulation.systems.ten_cubes.factory import build_ten_cubes_system
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

def test_scenario10_repair_cable():
    """
    After injecting scenario-10 faults (missing LEDs + resistors, disconnected
    ctrl6_cable_in_pos) repairing only ctrl6_cable_in_pos must restore the lamp.
    """
    # 1. Build system and attach logger (prints SPICE netlists to stdout).
    sim = build_ten_cubes_system(extra_tools={"multimeter"})
    sim.add_logger(_make_stdout_logger())

    # 2. Nominal simulation — captures which outputs are lit before any fault.
    nominal_result = sim.simulate()
    sim._nominal_emitting_light = nominal_result.emitting_light
    print(f"\n[nominal] emitting_light: {sorted(nominal_result.emitting_light)}")
    assert "main_bulb" in nominal_result.emitting_light, (
        "Main bulb should be lit in nominal state"
    )

    # 3. Inject scenario-10 faults.
    scenario = next(s for s in SCENARIOS if s.id == 10)
    for fault_fn in scenario.fault_fns:
        fault_fn(sim)
    print(f"\n[fault injection] scenario {scenario.id}: "
          f"{scenario.root_cause.root_cause_description_proper}")

    # 4. Snapshot the fault state.
    sim._fault_snapshot = sim.snapshot()

    # Simulate faulted state — lamp should be off.
    fault_result = sim.simulate()
    print(f"[faulted] emitting_light: {sorted(fault_result.emitting_light)}")
    assert "main_bulb" not in fault_result.emitting_light, (
        "Main bulb should be OFF in the faulted state"
    )

    # 5. Verify the fault: ctrl6_cable_in_pos.n must be floating.
    cable = sim.component("ctrl6_cable_in_pos")
    assert not cable.port("n").is_connected(), (
        "ctrl6_cable_in_pos.n should be floating (fault not injected?)"
    )
    print(f"[check] ctrl6_cable_in_pos.n is_connected={cable.port('n').is_connected()} ✓")

    # 6. test_repair: temporarily reconnect the cable and check the lamp.
    lamp_on = sim.test_repair({"ctrl6_cable_in_pos"})
    print(f"\n[test_repair] lamp_on={lamp_on}")
    assert lamp_on, (
        "Repairing ctrl6_cable_in_pos should restore the lamp — test_repair returned False"
    )

    # 7. After test_repair the circuit must be back in the fault state.
    assert not cable.port("n").is_connected(), (
        "test_repair must restore the fault state on exit — port should be floating again"
    )
    print("[check] fault state restored after test_repair ✓")


if __name__ == "__main__":
    test_scenario10_repair_cable()