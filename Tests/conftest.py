"""
Shared helpers and fixtures for all scenario tests.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

from diagnosable_systems_simulation.systems.three_cubes.factory          import build_three_cubes_system
from diagnosable_systems_simulation.systems.ten_cubes.factory            import build_ten_cubes_system
from diagnosable_systems_simulation.systems.asymmetric_chains.factory    import build_asymmetric_chains_system
from diagnosable_systems_simulation.systems.ambient_light_sensor.factory import build_ambient_light_system
from diagnosable_systems_simulation.systems.current_sensor.factory       import build_current_sensor_system
from Implementations.fault_injections import SCENARIOS

_FACTORIES = {
    "3_cubes":              build_three_cubes_system,
    "10_cubes":             build_ten_cubes_system,
    "asymmetric_chains":    build_asymmetric_chains_system,
    "ambient_light_sensor": build_ambient_light_system,
    "current_sensor":       build_current_sensor_system,
}

Step = tuple  # (ActionClass, dict[str, str])


def build_system_for(scenario_id: str):
    from diagnosable_systems_simulation.electrical_simulation.couplings import LooseConnectionCoupling
    sc = next(s for s in SCENARIOS if s.scenario_id == scenario_id)
    sys_ = _FACTORIES[sc.system_name](extra_tools={"multimeter"})
    sys_.simulate()
    if sc.system_config_fn:
        sc.system_config_fn(sys_)
    sys_.simulate()
    sys_._nominal_emitting_light = sys_.last_result.emitting_light
    for fn in (sc.fault_fns or []):
        fn(sys_)
    # Make loose-connection faults deterministic for test repeatability:
    # p=1.0 means the port is always disconnected on the first solver step,
    # so assert_system_broken reliably sees the fault.  snapshot()/restore_snapshot()
    # now save/restore couplings and call reset(), so the p=1.0 setting is
    # preserved across restores and the fault remains deterministic throughout.
    for c in sys_._runner.couplings:
        if isinstance(c, LooseConnectionCoupling):
            c.p = 1.0
    sys_._fault_snapshot = sys_.snapshot()
    return sys_


def assert_system_broken(sys_) -> None:
    """Simulate the faulted system and assert it is not nominal."""
    sys_.simulate()
    assert not sys_.is_system_nominal(), "System should be broken after fault injection"


def run_sequence(sys_, steps: list[Step]) -> None:
    import inspect
    from diagnosable_systems_simulation.actions.diagnostic_actions import ReplaceComponent
    from diagnosable_systems_simulation.electrical_simulation.couplings import LooseConnectionCoupling
    for action_or_cls, roles in steps:
        roles = dict(roles)
        part_id = roles.pop("part", None)
        if inspect.isclass(action_or_cls):
            action = action_or_cls(part_id) if part_id is not None else action_or_cls()
        else:
            action = action_or_cls
        targets = {role: sys_.component(cid) for role, cid in roles.items()}
        if isinstance(action, ReplaceComponent):
            # Remove loose coupling BEFORE the simulate so the replacement cable stays connected.
            subj_id = targets["subject"].component_id
            sys_._runner.couplings = [
                c for c in sys_._runner.couplings
                if not (isinstance(c, LooseConnectionCoupling) and c.component_id == subj_id)
            ]
        result = sys_.apply_action(action, targets)
        assert result.success, f"{type(action).__name__} failed: {result.message}"
