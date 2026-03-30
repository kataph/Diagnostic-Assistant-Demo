import random
from typing import Optional

from Knowledge_sources.Simulations.three_cubes.factory import build_three_cubes_system
from Knowledge_sources.Simulations.ten_cubes.factory import build_ten_cubes_system

from configuration import Configuration
from environment_classes import RootCauseDescription, Saboteur, SystemDescription

from .scenarios import SCENARIOS, Scenario

# Map SYSTEM_NAME (from Configuration) to (system_name_tag, builder_fn)
# system_name_tag must match Scenario.system_name values in scenarios.py
_BUILDERS = {
    "3CubesSystem":  ("3_cubes",  build_three_cubes_system),
    "10CubesSystem": ("10_cubes", build_ten_cubes_system),
}


class SaboteurSpiceSim(Saboteur):
    """
    Saboteur that injects faults into a live simulation of the target system.

    Steps:
      1. Build a fresh DiagnosableSystem instance for the configured system.
      2. Attach it to SystemDescription.simulated_system so that
         ServiceAgentSpiceSim can execute actions against it.
      3. Pick a scenario (FORCED_SCENARIO_ID >= 0 → that scenario;
         -1 → random simulatable scenario for this system).
      4. Apply all fault_fns in the chosen scenario.
      5. Return the scenario's RootCauseDescription.
    """

    @property
    def description(self) -> str:
        return super().description + f"_system={self.configuration.SYSTEM_NAME}_scenario={self.configuration.FORCED_SCENARIO_ID}"

    async def sabotage(self, description: SystemDescription) -> Optional[RootCauseDescription]:
        system_name = self.configuration.SYSTEM_NAME
        if system_name not in _BUILDERS:
            raise ValueError(
                f"SaboteurSpiceSim: no simulation builder for SYSTEM_NAME={system_name!r}. "
                f"Known systems: {list(_BUILDERS)}"
            )
        system_tag, builder = _BUILDERS[system_name]

        # Select scenario
        simulatable = [s for s in SCENARIOS if s.system_name == system_tag and s.fault_fns is not None]
        if not simulatable:
            raise ValueError(f"No simulatable scenarios found for system_tag={system_tag!r}")

        forced_id = self.configuration.FORCED_SCENARIO_ID
        if forced_id >= 0:
            matches = [s for s in simulatable if s.id == forced_id]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly 1 simulatable scenario with id={forced_id}, found {len(matches)}"
                )
            scenario: Scenario = matches[0]
        else:
            scenario = random.choice(simulatable)

        # Build a fresh simulation instance and attach it
        sim = builder(extra_tools=scenario.world_context.tools_in_hand)
        description.simulated_system = sim

        # Capture the nominal output (before any faults) so verify_hypothesis
        # can check whether output devices are lit after a repair.
        sim.simulate()
        sim._nominal_emitting_light = sim.last_result.emitting_light

        # Apply fault injections
        for fn in scenario.fault_fns:  # type: ignore[union-attr]
            fn(sim)

        # Snapshot the post-fault state so the service agent can reset to it
        # before hypothesis verification (undoing diagnostic side-effects).
        sim._fault_snapshot = sim.snapshot()

        self.logger.info(
            f"SaboteurSpiceSim: injected scenario {scenario.id} "
            f"({scenario.root_cause.one_liner_repr()}) into {system_name} by executiong the actions {[fn.__name__ for fn in scenario.fault_fns]}"
        )
        return scenario.root_cause
