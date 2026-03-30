from agents import Agent
from pydantic import BaseModel, Field
from typing import Literal, Optional
from configuration import Configuration
from environment_classes import AssistantState, DiagnosticAction, DiagnosticAssistant, DiagnosticFaultHypothesis, HypothesisVerificationResult, Observation, SystemDescription, ACTION_COST_MAP, diagnosticActionTypes
from Utilities.agents_boilerplate import get_conversation_start, get_updated_conversation
from Utilities.formatting import format_conversation_history, format_list
from Utilities.caching import possibly_cached_runner_run


class DiagnosticSuggestion(BaseModel):
    """
    Output type for the LLM diagnostic assistant.
    The LLM chooses between suggesting a diagnostic action or declaring a
    fault hypothesis.
    """
    suggestion_type: Literal["action", "hypothesis"]
    # Fields used when suggestion_type == "action":
    action_type: Optional[diagnosticActionTypes] = None
    action_target: Optional[str] = None
    action_description: Optional[str] = None
    # Fields used when suggestion_type == "hypothesis":
    suspected_components: Optional[list[str]] = None
    hypothesis_explanation: Optional[str] = None


class AssistantStateLLM(AssistantState):
    conversation_history: list[dict] = Field(default_factory=list)


class DiagnosticAssistantLLM(DiagnosticAssistant):

    def __init__(self, description: SystemDescription, configuration: Configuration):
        super().__init__(description, configuration)
        self.state = AssistantStateLLM(
            general_system_description=description,
            conversation_history=get_conversation_start(
                self.state.general_system_description),
        )
        self.constrainedoutputdiagnoser_v3 = Agent(
            name="ConstrainedOutputDiagnoser",
            instructions=(
                "You are an expert reliability engineer. You will receive a description of an "
                "engineered system that is suffering from some fault, and possibly a history of "
                "diagnostic actions executed on the system and their results.\n"
                "Your job is to find the root cause of such a fault.\n\n"
                "On each turn you must choose ONE of the two following output modes:\n\n"
                "MODE 1 — suggest a diagnostic action (suggestion_type='action'):\n"
                "  Use this when you need more information before committing to a diagnosis.\n"
                "  Provide action_type, action_target, and action_description.\n"
                "  - action_type: 'Replace' (swap a component), 'Adjust' (tune/repair without "
                "replacing), 'Test' (measure/manipulate with tools), or 'Observe' (visually "
                "inspect without tools).\n"
                "  - action_target: the component to act on.\n"
                "  - action_description: a brief natural-language description of the action.\n"
                "  - suspected_components: must be left None or empty in this case.\n"
                f"  Action costs by type: Replace={ACTION_COST_MAP['Replace']}, "
                f"Adjust={ACTION_COST_MAP['Adjust']}, Test={ACTION_COST_MAP['Test']}, "
                f"Observe={ACTION_COST_MAP['Observe']}.\n\n"
                "MODE 2 — declare a fault hypothesis (suggestion_type='hypothesis'):\n"
                "  Use this ONLY when you are sufficiently confident about which specific "
                "components are faulty. The service agent will then attempt to repair/replace "
                "those components and report whether the system is restored.\n"
                "  Provide suspected_components (a list of component names/IDs you believe are "
                "faulty) and optionally hypothesis_explanation.\n\n"
                "Do not switch to MODE 2 prematurely: a wrong hypothesis has a high fixed cost."
            ),
            model=self.configuration.LLM_ASSISTANT_MODEL,
            output_type=DiagnosticSuggestion,
        )

    @property
    def description(self) -> str:
        return super().description + "_" + self.configuration.LLM_ASSISTANT_MODEL

    async def setup(self, observations: list[Observation]) -> None:
        self.state.initial_observations = observations.copy()
        self.state.conversation_history = get_updated_conversation(
            self.state.conversation_history, f"The system is experiencing the following symptoms:\n{format_list(self.state.initial_observations)}")

    async def record_outcome(self, last_outcome) -> None:
        await super().record_outcome(last_outcome)
        self.state.conversation_history = get_updated_conversation(
            self.state.conversation_history, f"Someone executed a (additional) diagnostic action on the system. The action (type, target_component, description, outcome) was: \nTYPE: {last_outcome.action.type},\nTARGET: {last_outcome.action.target},\nDESCRIPTION: {last_outcome.action.description},\nRESULT: {last_outcome.outcome}\n")

    async def suggest_action(self) -> DiagnosticAction | DiagnosticFaultHypothesis:
        raw: DiagnosticSuggestion = await possibly_cached_runner_run(
            self.constrainedoutputdiagnoser_v3,
            input=self.state.conversation_history,
            cached=self.configuration.USE_CACHE,
        )
        self.logger.debug(
            f"INPUT [tail only with new content. Full content is concatenation of all previous log entries]: \n{format_conversation_history([self.state.conversation_history[-1]])}\n")
        self.logger.info(f"OUTPUT: {str(raw)}")
        if raw.suggestion_type == "action":
            return DiagnosticAction(
                type=raw.action_type,
                target=raw.action_target,
                description=raw.action_description,
            )
        else:
            return DiagnosticFaultHypothesis(
                suspected_components=set(raw.suspected_components or []),
                explanation=raw.hypothesis_explanation,
            )

    async def record_hypothesis_outcome(self, hypothesis, result):
        await super().record_hypothesis_outcome(hypothesis, result)
        components_str = ", ".join(hypothesis.suspected_components)
        self.state.conversation_history = get_updated_conversation(
            self.state.conversation_history,
            f"You declared a fault hypothesis: suspected faulty components = [{components_str}]. "
            f"The service agent attempted to repair/replace them. "
            f"Outcome: '{result.outcome}'. Details: {result.narrative}. "
            + (
                "The system is now fully restored — diagnosis complete."
                if result.outcome == "correct"
                else "The system is NOT yet fully restored. Please continue the diagnosis."
                if result.outcome == "partial"
                else "Your hypothesis was wrong: none of the suspected components were faulty. Please reconsider."
            )
        )

    def finish_session(self, root_cause):
        super().finish_session(root_cause)
