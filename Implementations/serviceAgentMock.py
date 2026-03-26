from typing import Optional
from asyncio import sleep

from environment_classes import DiagnosticFaultHypothesis, HypothesisVerificationResult, HYPOTHESIS_VERIFICATION_COST, ServiceAgent, SystemDescription, Observation, DiagnosticActionResult, DiagnosticAction, RootCauseDescription, AssistantState


class ServiceAgentMock(ServiceAgent):

    async def collect_initial_observations(
        self,
        system: SystemDescription,
        root_cause_description: Optional[RootCauseDescription]
    ) -> list[Observation]:
        self.mock_counter = 0
        mock_observations = [Observation(description=chunk) for chunk in system.text_input.split('.')]
        self.mock_observations_number = len(self.mock_observations)
        return mock_observations

    async def execute_action(self, system: SystemDescription, action: DiagnosticAction, root_cause_description: Optional[RootCauseDescription]) -> DiagnosticActionResult:
        await sleep(0.5)
        result = f"Mock outcome for {action.get_name()}"
        print(action, result)
        return DiagnosticActionResult(action=action, outcome=result)

    async def verify_hypothesis(
        self,
        system: SystemDescription,
        hypothesis: DiagnosticFaultHypothesis,
        root_cause_description: Optional[RootCauseDescription],
    ) -> HypothesisVerificationResult:
        return HypothesisVerificationResult(
            hypothesis=hypothesis,
            outcome="wrong",
            narrative="Mock: hypothesis not verified.",
            cost=HYPOTHESIS_VERIFICATION_COST,
        )

    async def decide_finish(self, system: SystemDescription, state: AssistantState, root_cause_description: Optional[RootCauseDescription]) -> tuple[bool, Optional[RootCauseDescription]]:
        if self.mock_counter > self.mock_observations_number:
            return True, RootCauseDescription(root_cause_description_proper="Mock root cause")
        self.mock_counter += 1
        return False, None
