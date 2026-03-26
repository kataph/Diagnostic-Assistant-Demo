from typing import Optional
from agents import Agent
from pydantic import BaseModel, RootModel

from Utilities.caching import possibly_cached_runner_run
from environment_classes import DiagnosticFaultHypothesis, HypothesisVerificationResult, HYPOTHESIS_VERIFICATION_COST, RootCauseDescription, ServiceAgent, SystemDescription, Observation, DiagnosticActionResult, DiagnosticAction, AssistantState

from Utilities.agents_boilerplate import get_conversation_start


class LightDiagnosticActionResult(BaseModel):
    action_type: str
    action_target: str
    action_outcome: str


# class TesterConstrainedOutput(BaseModel):
#     root_cause_identified: bool
#     diagnostic_actions_results: list[LightDiagnosticActionResult]

#     def __str__(self):
#         return "root_cause_identified: "+str(self.root_cause_identified)+"\ndiagnostic_actions_results:\n    "+"\n    ".join([str(x) for x in self.diagnostic_actions_results])


class TesterConstrainedLightOutput(BaseModel):
    action_outcome: str

    def __str__(self):
        return f"diagnostic_action_outcome: {self.action_outcome}"


class HypothesisVerificationOutput(BaseModel):
    outcome: str   # "correct" | "partial" | "wrong"
    narrative: str


class ObservationList(BaseModel):
    observations: list[str]

    def __str__(self) -> str:
        return str(self.observations)

    def __repr__(self) -> str:
        return self.__str__()


class ServiceAgentLLM(ServiceAgent):

    def __init__(self, configuration):
        super().__init__(configuration)
        self.patience_level = configuration.MAX_NUMBER_OF_ROUNDS - \
            1  # simulates an user patience
        self.annoyance_level = 0  # simulates an user patience
        self.service_model = configuration.SERVICE_MODEL
        self.serviceAgent = Agent(
            name="ConstrainedInputTester",
            instructions="""You are an engineer expert in simulating diagnosis scenarios. You will receive in input a description of an engineered system and a description of the fault that the system is currently suffering from. Your job is to train another engineer in diagnosing the system. To do so, the trainee will hypothesize a single diagnostic action to carry out on the system to determine the root cause of the fault. The diagnostic action will be given as an (action_type, action_target, action_description) tuple. You have to answer your colleague with the results of the diagnostic action.
            Your output must be:
            action_outcome: a sentence describing what the diagnostic action result was.
            Always respond concisely with just the results of the diagnostic actions without further inference or considerations. Also do not reveal directly what the root cause is, only that you answer describing the results of the diagnostic actions. """,
            model=self.service_model,
            output_type=TesterConstrainedLightOutput,
        )
        self.hypothesisVerifierAgent = Agent(
            name="HypothesisVerifier",
            instructions="""You are an engineer expert in simulating diagnosis scenarios. You will receive a description of an engineered system, the actual fault it is suffering from, and a hypothesis proposed by a trainee about which components are faulty. Your job is to simulate what would happen if those components were repaired/replaced.

Respond with:
  outcome: exactly one of "correct", "partial", or "wrong".
    - "correct": ALL faulty components were named in the hypothesis and repairing/replacing them restores full system function.
    - "partial": at least one named component was indeed faulty and has been fixed, but the system is still not working because additional faults remain that were NOT named.
    - "wrong": none of the named components were actually faulty; repairing them changes nothing.
  narrative: a brief sentence describing what the repair attempt revealed.

Do NOT reveal the actual root cause beyond what is needed to justify the outcome.""",
            model=self.service_model,
            output_type=HypothesisVerificationOutput,
        )
        self.serviceAgent_initialObservations = Agent(
            name="ConstrainedInputTester_start",
            instructions="""You are an engineer expert in simulating diagnosis scenarios. You will receive in input a description of an engineered system and a description of the fault that the system is currently suffering from. Your job is to train another engineer in diagnosing the system. To do so, the trainee will hypothesize a single diagnostic action to carry out on the system to determine the root cause of the fault. In order to make the first hypothesis, the trainee needs to be supplied a series of observations about the system: using your knowledge of the system behavior, write down a list relevant remarks about the system. Such remarks an be both about the part of the current behavior of the system that is anomalous or that is nominal. You get to decide what is relevant and what is not. Of course, you MUST NOT leak any information about the root cause to the trainee, as that would defeat the point of the exercise. Only tell the trainee about superficial, immediately-observable symptoms! Also please try not to give hints or suggestions to the trainee, only describe what can objectively be perceived, not what can be deduced.""",
            model=self.service_model,
            output_type=ObservationList,
        )

    @property
    def description(self) -> str:
        return super().description + "_" + f"patience={self.patience_level};service_model={self.service_model}"

    def _get_input_for_initial_observations(self, system: SystemDescription, root_cause_description: RootCauseDescription) -> list[dict]:
        return (get_conversation_start(system) +
                [{
                    "role": "user",
                    "content": [
                        {"type": "input_text",
                         "text": f"The system is experiencing the following fault: \n{root_cause_description.root_cause_description_proper} \nand thusly the following symptoms: \n{root_cause_description.symptoms_descriptions}"}
                    ]
                }])

    async def collect_initial_observations(
        self,
        system: SystemDescription,
        root_cause_description: RootCauseDescription,
    ) -> list[Observation]:

        o: ObservationList = await possibly_cached_runner_run(self.serviceAgent_initialObservations, input=self._get_input_for_initial_observations(system, root_cause_description), cached=self.configuration.USE_CACHE)
        # silly casting issues
        observations = [Observation(description=obs)
                        for obs in o.observations]
        self.logger.info(
            f"First observations about the system: {str(observations)}")
        return observations

    async def execute_action(self, system: SystemDescription, action: DiagnosticAction, root_cause_description: RootCauseDescription) -> DiagnosticActionResult:
        conversation_history_service = (
            self._get_input_for_initial_observations(system, root_cause_description) +
            # For the tester, the conversation history does not grow: it does not need to remember previous tests, and it would also risk losing memory or becoming biased
            [{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"The trainee says:\n{action.get_full_repr()}"
                    }
                ]
            }])

        o: TesterConstrainedLightOutput = await possibly_cached_runner_run(self.serviceAgent, input=conversation_history_service, cached=self.configuration.USE_CACHE)
        self.logger.info(
            f"Agent executed action: {action.get_name()} with outcome {o.action_outcome}")
        return DiagnosticActionResult(action=action, outcome=o.action_outcome)

    async def verify_hypothesis(
        self,
        system: SystemDescription,
        hypothesis: DiagnosticFaultHypothesis,
        root_cause_description: Optional[RootCauseDescription],
    ) -> HypothesisVerificationResult:
        components_str = ", ".join(hypothesis.suspected_components)
        conversation = (
            self._get_input_for_initial_observations(system, root_cause_description) +
            [{
                "role": "user",
                "content": [{"type": "input_text", "text":
                    f"The trainee declares the following components to be faulty: [{components_str}]. "
                    f"Simulate what would happen if those components were repaired/replaced."
                    + (f" Explanation from trainee: {hypothesis.explanation}" if hypothesis.explanation else "")
                }]
            }]
        )
        o: HypothesisVerificationOutput = await possibly_cached_runner_run(
            self.hypothesisVerifierAgent, input=conversation, cached=self.configuration.USE_CACHE
        )
        # Normalise outcome in case the LLM adds extra text
        outcome_raw = o.outcome.strip().lower()
        if "correct" in outcome_raw:
            outcome = "correct"
        elif "partial" in outcome_raw:
            outcome = "partial"
        else:
            outcome = "wrong"
        self.logger.info(
            f"Hypothesis verification: suspected={hypothesis.suspected_components} "
            f"outcome='{outcome}' narrative='{o.narrative}'"
        )
        return HypothesisVerificationResult(
            hypothesis=hypothesis,
            outcome=outcome,
            narrative=o.narrative,
            cost=HYPOTHESIS_VERIFICATION_COST,
        )

    async def decide_finish(self, system: SystemDescription, state: AssistantState, root_cause_description: RootCauseDescription) -> tuple[bool, None]:
        if self.annoyance_level >= self.patience_level:
            self.logger.info(
                "The Service agent decides it is not worth using the tool...")
            return (True, None)
        self.annoyance_level += 1
        return (False, None)
