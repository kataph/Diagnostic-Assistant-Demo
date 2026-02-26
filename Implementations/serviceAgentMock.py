from typing import Optional
from asyncio import sleep

from environment_classes import ServiceAgent, SystemDescription, Observation, DiagnosticActionResult, DiagnosticAction, RootCauseDescription, AssistantState

class ServiceAgentMock(ServiceAgent):

    async def collect_initial_observations(
        self,
        system: SystemDescription,
        root_cause_description: Optional[RootCauseDescription]
    ) -> list[Observation]:
        self.mock_counter = 0
        return [Observation(description = chunk) for chunk in system.text_input.split('.')]

    async def execute_action(self, system: SystemDescription, action: DiagnosticAction, root_cause_description: Optional[RootCauseDescription]) -> DiagnosticActionResult:
        await sleep(0.5)
        result = f"Mock outcome for {action.get_name()}"
        print(action, result)
        return DiagnosticActionResult(action=action, outcome=result)

    async def decide_finish(self, system: SystemDescription, state: AssistantState, root_cause_description: Optional[RootCauseDescription]) -> tuple[bool, Optional[RootCauseDescription]]:
        if self.mock_counter > len([Observation(description = system.text_input)]):
            return True, RootCauseDescription(root_cause_description_proper="Mock root cause")
        self.mock_counter += 1
        return False, None