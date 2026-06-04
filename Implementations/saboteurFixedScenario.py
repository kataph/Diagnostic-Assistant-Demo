from typing import Optional

from environment_classes import Saboteur, SystemDescription, RootCauseDescription

from .fault_injections import SCENARIOS


class SaboteurFixedScenario(Saboteur):

    @property
    def description(self):
        return super().description + "_" + f"scenario_id={self.configuration.FORCED_SCENARIO_ID}"

    async def sabotage(self, description: SystemDescription) -> Optional[RootCauseDescription]:
        forced_scenarios = [s for s in SCENARIOS if s.id == self.configuration.FORCED_SCENARIO_ID]
        if len(forced_scenarios) != 1:
            raise ValueError(f"Scenarios with id {self.configuration.FORCED_SCENARIO_ID} are {len(forced_scenarios)}. It should be exactly 1!")
        scenario = forced_scenarios[0]
        self.logger.info(
            f"The system was saboted producing the following root cause: {scenario.root_cause.one_liner_repr()}")
        return scenario.root_cause
