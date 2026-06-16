"""
Tests for principled hypothesis verification cost in ServiceAgentSpiceSim.

Verifies that:
1. A correct hypothesis naming one degraded component costs 120 (replace cost).
2. A wrong hypothesis naming one non-faulty component also costs 120 (same
   attempted repair cost — the technician still had to try).
3. A correct hypothesis naming two crossed cables costs 2×10=20, not 120,
   because cable reconnection is cheaper than component replacement.
4. A wrong hypothesis naming two non-faulty cables also costs 20.
5. Costs scale with the number of components named in the hypothesis.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

import pytest
from diagnosable_systems_simulation.systems.three_cubes.factory import build_three_cubes_system

from Implementations.serviceAgentSpiceSim import _estimate_repair_cost


def _build_faulted_system(*fault_fns):
    sys_ = build_three_cubes_system(extra_tools={"multimeter"})
    sys_.simulate()
    sys_._nominal_emitting_light = sys_.last_result.emitting_light
    for fn in fault_fns:
        fn(sys_)
    sys_._fault_snapshot = sys_.snapshot()
    return sys_


# ---------------------------------------------------------------------------
# Helpers to find component IDs
# ---------------------------------------------------------------------------

def _first_non_cable_id(sys_) -> str:
    from diagnosable_systems_simulation.world.components import Cable
    return next(
        cid for cid, c in sys_.all_components().items()
        if not isinstance(c, Cable)
    )


def _first_cable_id(sys_) -> str:
    from diagnosable_systems_simulation.world.components import Cable
    return next(
        cid for cid, c in sys_.all_components().items()
        if isinstance(c, Cable)
    )


def _two_cable_ids(sys_) -> list[str]:
    from diagnosable_systems_simulation.world.components import Cable
    return [
        cid for cid, c in sys_.all_components().items()
        if isinstance(c, Cable)
    ][:2]


# ---------------------------------------------------------------------------
# Tests for _estimate_repair_cost (pure cost estimator, no mutation)
# ---------------------------------------------------------------------------

class TestEstimateRepairCost:

    def test_non_faulty_component_costs_120(self):
        """A non-faulty non-cable component: estimate = 120 (replacement cost)."""
        sys_ = _build_faulted_system()
        cid = _first_non_cable_id(sys_)
        cost = _estimate_repair_cost(sys_, {cid})
        assert cost == 120.0

    def test_two_non_faulty_components_cost_240(self):
        """Two non-cable components: estimate scales linearly."""
        sys_ = _build_faulted_system()
        from diagnosable_systems_simulation.world.components import Cable
        non_cables = [
            cid for cid, c in sys_.all_components().items()
            if not isinstance(c, Cable)
        ][:2]
        assert len(non_cables) == 2
        cost = _estimate_repair_cost(sys_, set(non_cables))
        assert cost == 240.0

    def test_faulted_component_costs_120(self):
        """A genuinely degraded component: estimate = 120."""
        from diagnosable_systems_simulation.actions.fault_actions import DegradeComponent
        from Implementations.fault_injections import _apply
        sys_ = _build_faulted_system()
        target_id = "battery"
        _apply(sys_, DegradeComponent({"voltage": 0.0}), {"subject": sys_.component(target_id)})
        cost = _estimate_repair_cost(sys_, {target_id})
        assert cost == 120.0

    def test_non_faulty_cable_costs_zero(self):
        """A cable with no faulted ports: no reconnections needed, cost = 0."""
        sys_ = _build_faulted_system()
        cid = _first_cable_id(sys_)
        cost = _estimate_repair_cost(sys_, {cid})
        assert cost == 0.0

    def test_crossed_cables_cost_per_port(self):
        """Two crossed cables (polarity swap): cost = 10 per wrong-net port reconnection.
        SwapCablePolarities swaps port 'p' between ctrl_cable_in_pos and ctrl_cable_in_neg,
        leaving each with one port on the wrong net → 2 × 10 = 20 total."""
        from Implementations.fault_injections import _apply, _cross_psu_ctrl_cables
        sys_ = _build_faulted_system(_cross_psu_ctrl_cables)
        cable_a_id = "ctrl_cable_in_pos"
        cable_b_id = "ctrl_cable_in_neg"
        cost = _estimate_repair_cost(sys_, {cable_a_id, cable_b_id})
        assert cost == 20.0

    def test_crossed_cables_cheaper_than_component_replacement(self):
        """Crossed cables (cost=20) are cheaper than replacing one component (cost=120)."""
        from Implementations.fault_injections import _cross_psu_ctrl_cables
        sys_ = _build_faulted_system(_cross_psu_ctrl_cables)
        cable_cost = _estimate_repair_cost(sys_, {"ctrl_cable_in_pos", "ctrl_cable_in_neg"})
        component_cost = _estimate_repair_cost(sys_, {"battery"})
        assert cable_cost < component_cost
        assert cable_cost == 20.0
        assert component_cost == 120.0

    def test_estimate_does_not_mutate_state(self):
        """_estimate_repair_cost must not change any circuit state."""
        sys_ = _build_faulted_system()
        from diagnosable_systems_simulation.world.components import Cable
        non_cables = [
            cid for cid, c in sys_.all_components().items()
            if not isinstance(c, Cable)
        ]
        before = {cid: dict(sys_.component(cid)._fault_overlay) for cid in non_cables}
        _estimate_repair_cost(sys_, set(non_cables))
        after = {cid: dict(sys_.component(cid)._fault_overlay) for cid in non_cables}
        assert before == after, "estimate_repair_cost mutated fault overlays"