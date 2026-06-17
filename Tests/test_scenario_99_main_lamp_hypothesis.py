"""
Tests for hypothesis verification targeting "main lamp":

1. Scenario 99 (3CC-AS.I.4): loose connection on load_cable_pos.p — NOT on the lamp.
   Verifying the lamp must NOT repair the system (KO expected → test passes if KO).

2. Variant: loose connection directly on main_bulb.p — IS on the lamp.
   Verifying the lamp MUST repair the system (OK expected → test passes if OK).

Both use a real OpenRouter LLM call.
SKIPPED BY DEFAULT — run with: RUN_EXPENSIVE=1 pytest Tests/test_scenario_99_main_lamp_hypothesis.py -v -s
Cost: ~$0.10 per run.
"""
import asyncio
import os
import pytest
from Implementations.fault_injections import load_scenarios_from_csv, loose_connection
from diagnosable_systems_simulation.systems.ambient_light_sensor.factory import build_ambient_light_system
from environment_classes import DiagnosticFaultHypothesis, SystemDescription
from Implementations.serviceAgentSpiceSim import ServiceAgentSpiceSim
from configuration import Configuration

_skip = pytest.mark.skipif(
    not os.getenv("RUN_EXPENSIVE"),
    reason="Costs money (~$0.10) — set RUN_EXPENSIVE=1 to run",
)


def _build_config():
    config = Configuration(
        SABOTEUR_TYPE="FixedScenario",
        SERVICE_TYPE="SpiceSim",
        ASSISTANT_TYPE="LLM",
        TEXT_INPUT_FILE="Knowledge_sources/Unstructured_knowledge_sources/3_cubes_with_ambient_sensor/3cc_as_description.txt",
        SYSTEM_NAME="3CC-AS"
    )
    config.SERVICE_CONFIG = {"model": "openai/gpt-4-turbo"}
    return config


def _build_system_with_fault(fault_fn):
    sys = build_ambient_light_system(extra_tools={"multimeter"})
    sys.simulate()
    sys._nominal_emitting_light = sys.last_result.emitting_light
    fault_fn(sys)
    sys._fault_snapshot = sys.snapshot()
    for _ in range(20):
        sys.simulate()
        if not sys.is_system_nominal():
            break
    return sys


def _lamp_hypothesis():
    return DiagnosticFaultHypothesis(
        suspected_components={"main lamp"},
        explanation="The lamp is not lighting up"
    )


@_skip
def test_scenario_99_lamp_hypothesis_does_not_repair():
    """
    Scenario 99 (3CC-AS.I.4): loose connection on load_cable_pos.p.
    The fault is NOT on the lamp — verifying 'main lamp' must NOT repair the system.
    """
    scenarios = load_scenarios_from_csv()
    scenario_99 = next(s for s in scenarios if s.scenario_id == "3CC-AS.I.4")

    sys = build_ambient_light_system(extra_tools={"multimeter"})
    sys.simulate()
    sys._nominal_emitting_light = sys.last_result.emitting_light
    if scenario_99.system_config_fn:
        scenario_99.system_config_fn(sys)
    for fn in (scenario_99.fault_fns or []):
        fn(sys)
    sys._fault_snapshot = sys.snapshot()

    for _ in range(20):
        sys.simulate()
        if not sys.is_system_nominal():
            break

    assert not sys.is_system_nominal(), "Fault should be active before verification"

    service_agent = ServiceAgentSpiceSim(configuration=_build_config())
    sys_desc = SystemDescription(text_input="3CC-AS system")
    sys_desc.simulated_system = sys

    result = asyncio.run(service_agent.verify_hypothesis(sys_desc, _lamp_hypothesis(), scenario_99.root_cause))

    is_nominal = sys.is_system_nominal()
    print(f"\n=== Scenario 99 — loose on cable (expected KO) ===")
    print(f"Outcome: {result.outcome}, system nominal: {is_nominal}, cost: {result.cost}")

    assert not is_nominal, (
        f"UNEXPECTED OK: System became nominal after repairing the lamp, "
        f"but the fault was on load_cable_pos. Outcome: '{result.outcome}'. "
        f"Narrative: {result.narrative}"
    )
    print("✓ OK: Correct — repairing the lamp did NOT fix the system (fault is elsewhere)")


@_skip
def test_lamp_port_loose_connection_repaired_by_lamp_hypothesis():
    """
    Variant: loose connection directly on main_bulb.p — fault IS on the lamp.
    Verifying 'main lamp' must repair the system.
    """
    fault_fn = loose_connection("main_bulb", "p", p=0.5)
    sys = _build_system_with_fault(fault_fn)

    assert not sys.is_system_nominal(), "Fault should be active before verification"

    service_agent = ServiceAgentSpiceSim(configuration=_build_config())
    sys_desc = SystemDescription(text_input="3CC-AS system")
    sys_desc.simulated_system = sys

    result = asyncio.run(service_agent.verify_hypothesis(sys_desc, _lamp_hypothesis(), None))

    is_nominal = sys.is_system_nominal()
    print(f"\n=== Variant — loose on main_bulb.p (expected OK) ===")
    print(f"Outcome: {result.outcome}, system nominal: {is_nominal}, cost: {result.cost}")

    assert is_nominal, (
        f"KO: System NOT nominal after repairing the lamp, "
        f"but the fault WAS on main_bulb.p. Outcome: '{result.outcome}'. "
        f"Narrative: {result.narrative}"
    )
    print("✓ OK: Correct — repairing the lamp DID fix the system")
