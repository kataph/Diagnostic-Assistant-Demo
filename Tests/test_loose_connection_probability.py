"""
Verifies that LooseConnectionCoupling produces a genuine 50/50 lamp-on/lamp-off
distribution across independent simulate() calls.

With the old implementation the coupling bounced within a single solve's
coupling-loop iterations, always ending connected — giving ~100% lamp-on.
With the fixed implementation the coin is flipped once per simulate() call
and the result is held for the entire solve, giving ~50% lamp-on.

The test runs 20 simulations and asserts that both outcomes occur at least
3 times each (probability of failing by chance: < 0.1% for a fair coin).
"""
from __future__ import annotations

import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system
from diagnosable_systems_simulation.electrical_simulation.couplings import _add_loose_connection

N = 20
MIN_EACH = 3  # minimum occurrences of each outcome expected


def _build_loose_system(seed: int):
    random.seed(seed)
    sys_ = build_three_cubes_system(extra_tools={"multimeter"})
    sys_.simulate()
    sys_._nominal_emitting_light = sys_.last_result.emitting_light
    _add_loose_connection(sys_, "psu_cable_pos", "p", p=0.5)
    sys_._fault_snapshot = sys_.snapshot()
    return sys_


def test_loose_connection_is_50_50():
    sys_ = _build_loose_system(seed=0)
    nominal_lit = frozenset(
        cid for cid in sys_._nominal_emitting_light
        if sys_._kg is not None
    )
    # Use nominal_emitting_light directly
    nominal_lit = sys_._nominal_emitting_light

    results = []
    for _ in range(N):
        result = sys_.simulate()
        lamp_on = nominal_lit.issubset(result.emitting_light)
        results.append(lamp_on)

    on_count = sum(results)
    off_count = N - on_count

    assert on_count >= MIN_EACH, (
        f"Expected at least {MIN_EACH} lamp-ON results out of {N}, got {on_count}. "
        f"Coupling may still be always-reconnecting within the solver loop."
    )
    assert off_count >= MIN_EACH, (
        f"Expected at least {MIN_EACH} lamp-OFF results out of {N}, got {off_count}. "
        f"Coupling may not be disconnecting the port at all."
    )
