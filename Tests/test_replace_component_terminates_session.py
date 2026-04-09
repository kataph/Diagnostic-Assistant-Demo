"""
Non-regression test: replacing the sole faulty component via a diagnostic
action (not hypothesis verification) must terminate the session successfully.

Scenario: battery is the only fault (voltage=0.0).
A technician replaces the battery directly through the replace_component
diagnostic action.  Afterwards:
  - The lamp is on (system is restored).
  - decide_finish() detects the restored state and returns (True, None),
    ending the session — which is counted as a success in the metrics.
"""
import asyncio
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "diagnosable-systems-simulation"))

import pytest
from diagnosable_systems_simulation.systems.ten_cubes.factory import build_ten_cubes_system
from diagnosable_systems_simulation.actions.fault_actions import DegradeComponent
from diagnosable_systems_simulation.actions.diagnostic_actions import ReplaceComponent

from Implementations.serviceAgentSpiceSim import ServiceAgentSpiceSim
from environment_classes import SystemDescription


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply(sys_, action, targets):
    action.execute(targets, sys_._graph, sys_.context, None)


def _make_agent() -> ServiceAgentSpiceSim:
    """Construct a ServiceAgentSpiceSim without a real Configuration or OpenAI client."""
    agent = ServiceAgentSpiceSim.__new__(ServiceAgentSpiceSim)
    agent.configuration = None          # not used by decide_finish
    agent.logger = logging.getLogger("test.replace_terminates")
    agent.patience_level = 9            # MAX_NUMBER_OF_ROUNDS - 1 = 10 - 1
    agent.annoyance_level = 0
    agent._repaired_comp_ids = set()
    return agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def faulted_system():
    """10-cubes system with only the battery depleted."""
    sys_ = build_ten_cubes_system(extra_tools={"multimeter"})
    # Record nominal emitting light BEFORE applying the fault.
    sys_.simulate()
    sys_._nominal_emitting_light = sys_.last_result.emitting_light
    # Apply fault: battery voltage → 0 V.
    _apply(sys_, DegradeComponent({"voltage": 0.0}), {"subject": sys_.component("battery")})
    sys_._fault_snapshot = sys_.snapshot()
    return sys_


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReplaceComponentTerminatesSession:
    def test_lamp_off_with_battery_fault(self, faulted_system):
        """Sanity: lamp is off when the battery is depleted."""
        result = faulted_system.simulate()
        nominal = faulted_system._nominal_emitting_light
        assert not (result.emitting_light >= nominal), \
            "Lamp should be off when battery is depleted"

    def test_lamp_on_after_replace_component(self, faulted_system):
        """Replacing the battery directly restores the lamp."""
        action = ReplaceComponent(replacement_part_id="battery_unit")
        battery = faulted_system.component("battery")
        action.execute({"subject": battery}, faulted_system._graph, faulted_system.context, None)

        result = faulted_system.simulate()
        nominal = faulted_system._nominal_emitting_light
        assert result.emitting_light >= nominal, \
            "Lamp must be on after the battery is replaced"

    def test_decide_finish_returns_true_after_replace(self, faulted_system):
        """decide_finish() detects the restored lamp and ends the session."""
        # Replace the battery (diagnostic action, no hypothesis).
        action = ReplaceComponent(replacement_part_id="battery_unit")
        battery = faulted_system.component("battery")
        action.execute({"subject": battery}, faulted_system._graph, faulted_system.context, None)

        agent = _make_agent()
        system_desc = SystemDescription(text_input="", simulated_system=faulted_system)

        finished, root = asyncio.run(agent.decide_finish(system_desc, None, None))
        assert finished, "decide_finish must return True when lamp is on"
        # root may be None — that is fine, the orchestrator marks it as success
        # via the system_restored_via_action log marker.

    def test_decide_finish_returns_false_before_replace(self, faulted_system):
        """decide_finish() does NOT end the session while the battery is still dead."""
        agent = _make_agent()
        system_desc = SystemDescription(text_input="", simulated_system=faulted_system)

        finished, _ = asyncio.run(agent.decide_finish(system_desc, None, None))
        assert not finished, \
            "decide_finish must not end the session while the fault is still active"
