from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
import logging

from abc import ABC, abstractmethod
from typing import Callable, Iterator, Literal, Optional
from diagnosable_systems_simulation import DiagnosableSystem
from diagnosable_systems_simulation.world.context import WorldContext
from pydantic import BaseModel, ConfigDict, Field, RootModel

from configuration import Configuration
from voice_client import send_prompt, get_user_text
from time import perf_counter

ACTION_COST_MAP = {
    'Replace': 12,
    'Adjust': 4,
    'Test': 2,
    'Observe': 1,
}

HYPOTHESIS_VERIFICATION_COST: int = 120

class SystemDescription(BaseModel):
    model_config = ConfigDict(frozen=False, arbitrary_types_allowed=True)
    text_input: str
    file_id: str | None = None       # OpenAI uploaded-file id (preferred when available)
    image_b64: str | None = None     # Base64-encoded PNG fallback (used when upload fails)
    simulated_system: DiagnosableSystem | None = None


class SymptomDescription(RootModel[str]):
    def __str__(self):
        return f"'{self.root}'"

    def simple_string(self) -> str:
        return self.root


class SymptomDescriptions(RootModel[list[SymptomDescription]]):
    def __iter__(self) -> Iterator[SymptomDescription]:
        return iter(self.root)

    def __str__(self):
        return "\n".join([str(x) for x in self.root])

    def __len__(self):
        return len(self.root)

    def one_line_repr(self) -> str:
        return str(self.root)

    def append(self, item: SymptomDescription) -> None:
        self.root.append(item)


class ObservationDescriptions(RootModel[list[SymptomDescription]]):
    def __iter__(self) -> Iterator[SymptomDescription]:
        return iter(self.root)

    def __str__(self):
        return "\n".join([str(x) for x in self.root])

    def append(self, item: SymptomDescription) -> None:
        self.root.append(item)


class SingleFaultOutput(BaseModel):
    root_cause_description: str
    symptoms_descriptions: SymptomDescriptions


class TesterOutput(BaseModel):
    system_works_again: bool
    diagnostic_actions_results: str


class TextDiagnosticActionResult(BaseModel):
    action_name: str
    action_result_description: str

    def __str__(self):
        return f"{{action_name: {self.action_name}, action_result_description: {self.action_result_description}}}"




diagnosticActionTypes = Literal['Replace', 'Adjust', 'Observe', 'Test']


class DiagnosticAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: diagnosticActionTypes
    target: str
    description: Optional[str] = None
    # Optional requirements on how the outcome of this action should be reported
    # (e.g. "include ANOMALOUS/NOMINAL verdict token").  Intended for post-execution
    # reporting; do not use for action identification/parsing.
    reporting_requirements: Optional[str] = None

    def get_name(self):
        return f"{self.type} -> {self.target}"

    def __str__(self):
        return f"\nname: {self.get_name()},\ndescription: {self.description}"

    def __repr__(self):
        return self.__str__()

    def get_full_repr(self):
        return f"\ntype: {self.type},\ntarget: {self.target},\ndescription: {self.description}"

    def get_cost(self) -> int:
        return ACTION_COST_MAP[self.type]

class SimplifiedOutcome(Enum):
    ANOMALOUS: str = "ANOMALOUS"
    NOMINAL: str = "NOMINAL"
    
class DiagnosticActionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action: DiagnosticAction
    outcome: str
    precise_action_cost: Optional[float] = None
    cost_breakdown: Optional[list[tuple[str, float]]] = None
    simplified_outcome: Optional[SimplifiedOutcome] = None
    raw_results: Optional[list] = None

    def __str__(self):
        return (f"{{action name: {self.action.get_name()}, "+(f"action description: '{self.action.description}'" if self.action.description else "")+f" action result description: '{self.outcome}'}}")


class TextDiagnosticAction(BaseModel):
    action_name: str
    action_description: str

    def __str__(self):
        return f"\naction_name: {self.action_name},\naction_description: {self.action_description}"


class TextDiagnosticActionsList(BaseModel):
    diagnostic_actions: list[TextDiagnosticAction]

    def __str__(self):
        return "\n".join([str(x) for x in self.diagnostic_actions])


class DiagnosticActionsList(BaseModel):
    diagnostic_actions: list[DiagnosticAction]

    def __str__(self):
        return "\n".join([str(x) for x in self.diagnostic_actions])


class Observation(BaseModel):
    description: str
    confidence: Optional[str] = None

    def __str__(self):
        return f"'{self.description}'"

    def __repr__(self):
        return self.__str__()


class SimpleDescription(RootModel[str]):
    def __str__(self):
        return f"'{self.root}'"


class FaultDescription(RootModel[str]):
    def __str__(self):
        return f"'{self.root}'"


class RootCauseDescription(BaseModel):
    root_cause_description_proper: FaultDescription
    symptoms_descriptions: Optional[SymptomDescriptions] = None
    notes: Optional[str] = None

    def __repr__(self):
        return (f"root cause:\n{self.root_cause_description_proper}" +
                (f"\nsymptoms:\n{self.symptoms_descriptions}" if self.symptoms_descriptions else "") +
                (f"\nnotes:\n{self.notes}" if self.notes else ""))

    def one_liner_repr(self):
        symptoms_str = (
            " | ".join(s.simple_string() for s in self.symptoms_descriptions)
            if self.symptoms_descriptions else ""
        )
        return (f"root cause: '{self.root_cause_description_proper}'" +
                (f" symptoms/observable manifestantions: '{symptoms_str}'" if symptoms_str else "") +
                (f" notes: '{self.notes}'" if self.notes else ""))

    def __str__(self):
        return self.__repr__()


FaultFn = Callable[[DiagnosableSystem], None]


@dataclass
class Scenario:
    number: int
    scenario_id: str
    system_name: str
    system_config: str
    root_cause: RootCauseDescription
    fault_fns: Optional[list[FaultFn]]
    world_context: WorldContext
    system_config_fn: Optional[FaultFn] = field(default=None)

    @property
    def id(self) -> int:
        return self.number


class RootCauseHypothesis(BaseModel):
    component: str
    failure_mode: str
    confidence: float = Field(ge=0, le=1)
    proposed_by: str = "assistant"


class DiagnosticFaultHypothesis(BaseModel):
    """
    Emitted by a DiagnosticAssistant (instead of a DiagnosticAction) when it
    has gathered enough evidence to declare which components it believes are
    faulty.  The service agent must then attempt to verify the claim by
    repairing/replacing those components and observing whether system function
    is restored.
    """
    suspected_components: set[str]
    explanation: Optional[str] = None


class HypothesisVerificationResult(BaseModel):
    """
    Returned by ServiceAgent.verify_hypothesis() after attempting to
    repair/replace the components named in a DiagnosticFaultHypothesis.

    outcome:
      "correct" – all faulty components identified; system function fully restored.
      "partial" – at least one suspected component was indeed faulty and has been
                  fixed, but the system is still not working (more faults remain).
      "wrong"   – none of the suspected components turned out to be faulty.
    """
    hypothesis: DiagnosticFaultHypothesis
    outcome: Literal["correct", "partial", "wrong"]
    narrative: str
    cost: float = HYPOTHESIS_VERIFICATION_COST
    cost_breakdown: Optional[list[tuple[str, float]]] = None


# Human interface through either CLI or voice


class HumanIO(ABC):
    @abstractmethod
    async def prompt(self, text: str) -> None:
        """Show a message to the human (CLI or voice)."""
        ...

    @abstractmethod
    async def read_line(self, prompt: Optional[str] = None) -> str:
        """Get a line of input from the human."""
        ...


class CLIHumanIO(HumanIO):
    """
    Classic terminal-based I/O using print/input.
    """

    async def prompt(self, text: str) -> None:
        print(text)

    async def read_line(self, prompt: Optional[str] = None) -> str:
        # Run blocking input() in a thread so we don't block the event loop.
        loop = asyncio.get_running_loop()
        line = await loop.run_in_executor(None, input, prompt or "> ")
        return line


class VoiceHumanIO(HumanIO):
    """
    Voice I/O via the phone+Whisper backend (voice_server.py).
    """

    def __init__(self, session_id: str = "demo1"):
        self.session_id = session_id

    async def prompt(self, text: str) -> None:
        # Send prompt to the voice server; phone will poll /prompt/{session_id}
        await send_prompt(self.session_id, text)

    async def read_line(self, prompt: Optional[str] = None) -> str:
        # Optionally show an extra prompt text; then wait for user speech.
        if prompt:
            await self.prompt(prompt)
        text = await get_user_text(self.session_id)
        return text

###


class DiagnosticPlan(BaseModel, ABC):
    model_config = ConfigDict(extra="allow")  # Allows extra fields

    # actions: list[DiagnosticAction]
    # current_index: int = 0
    # status: PlanStatus = PlanStatus.ONGOING
    # hypotheses: list[RootCauseHypothesis] = Field(default_factory=list)

    @abstractmethod
    def get_next_action(self, logger: logging.Logger = None) -> DiagnosticAction | None:
        ...


class AssistantState(BaseModel):
    # model_config = ConfigDict(extra="allow") # Allows extra fields

    general_system_description: SystemDescription | None = None
    initial_observations: list[Observation] = Field(default_factory=list)
    diagnostic_scenario_memory: list[DiagnosticActionResult] = Field(
        default_factory=list)

    user_finished: bool = False
    user_confirmed_root_cause: Optional[RootCauseDescription] = None

# All diagnostic-scenario-environment-roles classes receive a configuration and may log info about their behavior


class ThingThatLogs(ABC):

    def _setup_logger(self, configuration: Configuration) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(configuration.LOG_LEVEL)  # INFO is 20
        self.logger.propagate = False  # don't leak to root logger / stderr

        self.logger.addHandler(configuration.get_file_handler())

    def __init__(self, configuration: Configuration):
        super().__init__()
        self.configuration = configuration
        self._setup_logger(self.configuration)

    @property
    def description(self) -> str:
        """Self-declared description of an instance. It should contain the name plus eventual essential characteristics for logging"""
        return self.__class__.__name__


# =========
# Abstract DiagnosticAssistant (black box)
# =========

class DiagnosticAssistant(ThingThatLogs):
    """
    Black-box diagnostic assistant.

    External code can:
      - setup() initial state
      - record_action_outcome(last_outcome) to update the assistant state
      - suggest_action() to advance the dialogue
      - finish_session(root_cause) when the *service agent* decides to stop
      - read .state for logging / UI / analysis
    """

    def __init__(self, description: SystemDescription, configuration: Configuration) -> None:
        """
        Initialize assistant internal state with general system informations.
        """
        super().__init__(configuration)
        self.state = AssistantState(general_system_description=description)

    @abstractmethod
    async def setup(self, observations: list[Observation]) -> None:
        """
        Updates assistant internal state with initial observations.
        """

    # @abstractmethod
    async def record_action_outcome(self, last_outcome: DiagnosticActionResult) -> None:
        """
        Update the assistant state given the last outcome. 
        """
        self.state.diagnostic_scenario_memory.append(last_outcome)

    @abstractmethod
    async def suggest_action(self) -> Optional[DiagnosticAction | DiagnosticFaultHypothesis]:
        """
        Return either:
          - a DiagnosticAction to be executed next, or
          - a DiagnosticFaultHypothesis when the assistant is confident enough
            to declare which components it believes are faulty, or
          - None when the assistant has no further ideas.
        """

    async def record_hypothesis_outcome(
        self,
        hypothesis: DiagnosticFaultHypothesis,
        result: HypothesisVerificationResult,
    ) -> None:
        """
        Called after the service agent has verified a fault hypothesis.
        Default implementation just logs; subclasses may override to incorporate
        the feedback into their internal reasoning state.
        """
        self.logger.info(
            f"Hypothesis {hypothesis.suspected_components} verified with "
            f"outcome '{result.outcome}': {result.narrative}"
        )

    def finish_session(self, root_cause: Optional[RootCauseDescription]) -> None:
        """
        Called by the *service agent* when it decides diagnosis is done.
        """
        self.state.user_finished = True
        if root_cause:
            self.state.user_confirmed_root_cause = root_cause
            self.logger.info(
                f"Session ended by user decision. Recorded root cause is >>{self.state.user_confirmed_root_cause.one_liner_repr()}<<")
        else:
            self.logger.info(
                f"Session ended by user decision. Root cause not recorded")


# =========
# SaboteurAgent and ServiceAgent interfaces
# =========

class Saboteur(ThingThatLogs):
    @abstractmethod
    async def sabotage(self, system: SystemDescription) -> Optional[RootCauseDescription]:
        """
        Perform sabotage and optionally return a (virtual) root cause description.
        """


class ServiceAgent(ThingThatLogs):

    @abstractmethod
    async def collect_initial_observations(
        self,
        system: SystemDescription,
        root_cause_description: Optional[RootCauseDescription]
    ) -> list[Observation]:
        ...

    @abstractmethod
    async def execute_action(self,
                             system: SystemDescription,
                             action: DiagnosticAction,
                             root_cause_description: Optional[RootCauseDescription]) -> DiagnosticActionResult:
        ...

    @abstractmethod
    async def verify_hypothesis(
        self,
        system: SystemDescription,
        hypothesis: DiagnosticFaultHypothesis,
        root_cause_description: Optional[RootCauseDescription],
    ) -> HypothesisVerificationResult:
        """
        Attempt to verify the assistant's fault hypothesis by repairing/replacing
        the suspected components and checking whether system function is restored.
        The verification carries a fixed cost of HYPOTHESIS_VERIFICATION_COST.

        Returns a HypothesisVerificationResult with outcome "correct", "partial",
        or "wrong".  If "correct" the orchestrator will end the session; for
        "partial" or "wrong" it feeds the result back to the assistant and
        continues the diagnostic loop.
        """

    @abstractmethod
    async def decide_finish(self, system: SystemDescription, state: AssistantState, root_cause_description: Optional[RootCauseDescription]) -> tuple[bool, Optional[RootCauseDescription], Optional[str]]:
        """
        Return (finished, root_cause, reason) where:
          finished   — True when the agent decides the diagnosis is done.
          root_cause — Optional confirmed root cause description.
          reason     — Optional human-readable string explaining why the session
                       is ending (logged and surfaced in the chat pill). None
                       means no specific reason to record beyond the default.
        """


# =========
# Orchestrator
# =========

async def run_diagnostic_scenario(
    system: SystemDescription,
    saboteur: Saboteur,
    service_agent: ServiceAgent,
    assistant: DiagnosticAssistant,
    scenario_logger: logging.Logger,
    chat_log=None,
    trajectory_log=None,
) -> None:
    scenario_logger.info(
        f"A diagnostic scenario has been started with a saboteur of type {saboteur.description}, a service agent of type {service_agent.description}, and an assistant of type {assistant.description}")

    # 1) Sabotage phase
    sabotage_root = await saboteur.sabotage(system)

    if chat_log and sabotage_root:
        chat_log.saboteur(sabotage_root.one_liner_repr())

    # 2) Initial observations
    if not sabotage_root or not sabotage_root.symptoms_descriptions:
        initial_obs = await service_agent.collect_initial_observations(system, sabotage_root)
    else:  # In this case the initial observations are those already present in the symptoms of the root cause description
        initial_obs = [Observation(description=symp.simple_string(
            )) for symp in sabotage_root.symptoms_descriptions]
    scenario_logger.info(f"Initial observations: {initial_obs}")

    if chat_log:
        obs_lines = "\n".join(f"• {o.description}" for o in initial_obs)
        chat_log.system(f"Initial observations:\n{obs_lines}")

    await assistant.setup(initial_obs)
    scenario_logger.info(
        f"Diagnostic assistant has finished setup. Starting diagnostic loop...")

    # 3) Diagnostic loop
    suggested_actions_history: list[DiagnosticActionResult] = []
    time_vector: list[float] = []
    cost_vector: list[float] = []
    finish_reason: Optional[str] = None
    end_value: str = "surrender"

    while True:
        start = perf_counter()
        suggestion = await assistant.suggest_action()
        end = perf_counter()
        time_vector.append(end - start)

        if isinstance(suggestion, DiagnosticFaultHypothesis):
            if chat_log:
                chat_log.assistant_hypothesis(suggestion.suspected_components, suggestion.explanation)
            # Assistant is declaring a fault hypothesis — service agent verifies it.
            verification = await service_agent.verify_hypothesis(system, suggestion, sabotage_root)
            cost_vector.append(verification.cost)
            breakdown_str = (
                " | cost_breakdown=" + " + ".join(
                    f"{name}:{t:.0f}s" for name, t in verification.cost_breakdown
                )
                if verification.cost_breakdown else ""
            )
            scenario_logger.info(
                f"Hypothesis {suggestion.suspected_components} verified: "
                f"outcome='{verification.outcome}' | cost={verification.cost:.0f}s{breakdown_str} | {verification.narrative}"
            )
            if chat_log:
                chat_log.service_verification(
                    verification.outcome, verification.narrative,
                    cost=verification.cost, cost_breakdown=verification.cost_breakdown,
                )
            if trajectory_log:
                trajectory_log.record_hypothesis(suggestion, verification)
            await assistant.record_hypothesis_outcome(suggestion, verification)

            if verification.outcome == "correct":
                # All faults found; system restored — end session.
                end_value = "success"
                assistant.finish_session(sabotage_root)
                break
            # "partial" or "wrong": fall through to decide_finish so the
            # service agent can stop early (patience) or continue.

        elif isinstance(suggestion, DiagnosticAction):
            if chat_log:
                full_desc = suggestion.description or ""
                if suggestion.reporting_requirements:
                    full_desc = full_desc + ("\n\n" if full_desc else "") + suggestion.reporting_requirements
                chat_log.assistant_action(suggestion.type, suggestion.target, full_desc)
            last_outcome = await service_agent.execute_action(system, suggestion, sabotage_root)
            suggested_actions_history.append(last_outcome)
            cost_vector.append(
                last_outcome.precise_action_cost
                if last_outcome.precise_action_cost is not None
                else last_outcome.action.get_cost()
            )
            if chat_log:
                chat_log.service_result(
                    suggestion.get_name(),
                    last_outcome.outcome,
                    cost=last_outcome.precise_action_cost
                    if last_outcome.precise_action_cost is not None
                    else last_outcome.action.get_cost(),
                    cost_breakdown=last_outcome.cost_breakdown,
                )
            if trajectory_log:
                trajectory_log.record_action(last_outcome)
            await assistant.record_action_outcome(last_outcome)

        else:  # suggestion is None
            scenario_logger.info("Assistant has no further actions to suggest.")
            if chat_log:
                chat_log.system("Assistant has no further actions to suggest.")

        # Ask service agent if it wants to finish.
        user_finished, user_root, finish_reason = await service_agent.decide_finish(system, assistant.state, sabotage_root)
        if user_finished:
            assistant.finish_session(user_root)
            if finish_reason:
                scenario_logger.info(f"Session ended by service agent: {finish_reason}")
                if "system_restored_via_action" in finish_reason:
                    end_value = "success_no_hypothesis"
                elif suggestion is None:
                    end_value = "surrender"
                else:
                    end_value = "timeout"
            break

        if suggestion is None:
            # No further suggestions and agent chose to continue: auto-stop.
            end_value = "surrender"
            scenario_logger.info("No further suggestions and service agent chose to continue — ending session automatically.")
            if chat_log:
                chat_log.system("No further suggestions and service agent chose to continue — session ended automatically.")
            break

    # 4) Summary

    if rc := assistant.state.user_confirmed_root_cause:
        scenario_logger.info(
            f"Service agent confirmed root cause: {rc.root_cause_description_proper}")
    else:
        scenario_logger.info(
            "Service agent ended session without recording a root cause.")
    scenario_logger.info(f"Cost vector: {cost_vector}")
    scenario_logger.info(
        f"Total cost: {sum(cost_vector)} --- Number of suggestions: {len(cost_vector)} ")
    scenario_logger.info(f"Time vector: {time_vector}")
    scenario_logger.info(
        f"Average time: {sum(time_vector)/len(time_vector) if time_vector else 0}")
    scenario_logger.debug(
        f"Diagnostic memory: {assistant.state.diagnostic_scenario_memory} ")

    total_cost = sum(cost_vector)
    rounds = len(cost_vector)

    if chat_log:
        if not assistant.state.user_confirmed_root_cause:
            if finish_reason:
                chat_log.system(f"⏱ {finish_reason}")
            else:
                chat_log.system("⏱ Session ended without identifying the root cause.")
        chat_log.close(f"{rounds} round{'s' if rounds != 1 else ''} · total cost {total_cost:.0f}")

    if trajectory_log:
        trajectory_log.close(end=end_value, total_cost=total_cost, length=rounds)

    # print("\nFull assistant state:")
    # print(assistant.state.model_dump(indent=2))


if __name__ == "__main__":
    pass
