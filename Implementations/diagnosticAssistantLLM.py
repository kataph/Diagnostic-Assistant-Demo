from agents import Agent
from pydantic import BaseModel, Field
from typing import Literal, Optional
from configuration import Configuration
from environment_classes import AssistantState, DiagnosticAction, DiagnosticAssistant, DiagnosticFaultHypothesis, HypothesisVerificationResult, Observation, SystemDescription, diagnosticActionTypes
from Utilities.agents_boilerplate import (
    append_assistant_turn,
    get_conversation_start,
    get_system_description_instructions,
    get_updated_conversation,
)
from Utilities.formatting import format_conversation_history, format_list
from Utilities.caching import possibly_cached_runner_run
import os as _os


def _make_model(model_name: str):
    """Return a model identifier suitable for the Agents SDK Agent constructor.

    When OPENAI_BASE_URL is unset we are talking directly to OpenAI, which
    supports the Responses API — return the model name as a string so the SDK
    uses its full native feature set (structured outputs, file uploads, etc.).

    When OPENAI_BASE_URL is set we are behind a proxy (OpenRouter, Ollama, …)
    that typically only supports /v1/chat/completions.  In that case we
    construct an explicit OpenAIChatCompletionsModel so the SDK never attempts
    the Responses API endpoint.
    """
    base_url = _os.environ.get("OPENAI_BASE_URL")
    if not base_url:
        return model_name
    # Strip the "openai/" routing prefix — it's an Agents SDK convention for
    # native OpenAI, meaningless (or invalid) on third-party proxies.
    proxy_model_name = model_name.removeprefix("openai/")
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from openai import AsyncOpenAI
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=_os.environ.get("OPENAI_API_KEY", ""),
    )
    return OpenAIChatCompletionsModel(model=proxy_model_name, openai_client=client)


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
        # If no_vision is set (e.g. for text-only local models), strip diagram before super().__init__.
        if configuration.ASSISTANT_CONFIG.get("no_vision"):
            description = description.model_copy(update={"file_id": None, "image_b64": None})
        super().__init__(description, configuration)

        # optimized_history=True (default): system description in Agent instructions (cacheable);
        # assistant turns appended after each suggestion. Set {"optimized_history": false} to revert
        # to legacy behaviour (system desc as first user message, no assistant turns).
        self._optimized_history: bool = configuration.ASSISTANT_CONFIG.get("optimized_history", True)

        self.state = AssistantStateLLM(
            general_system_description=description,
            conversation_history=(
                [] if self._optimized_history
                else get_conversation_start(self.state.general_system_description)
            ),
        )

        _task_instructions = (
            "You are an expert reliability engineer. You will receive a description of an "
                "engineered system that is suffering from some fault, and possibly a history of "
                "diagnostic actions executed on the system and their results.\n\n"
                "YOUR GOAL: restore the system main functionality by identifying and fixing "
                "the root cause of the fault. The session ends successfully when the system is "
                "main function is restored (secondary function like indicators and such are not "
                "strictly needed for this). Restoration can happen in two ways:\n"
                "  1. You declare a correct fault hypothesis (MODE 2) and the service agent "
                "repairs the identified components — this is the PREFERRED path. Declaring a "
                "fault hypothesis means believing that the target components are faulty NOW, "
                "not that they have been just repaired.\n"
                "  2. A Replace action (MODE 1) happens to fix the fault component directly.\n\n"
                "On each turn you must choose ONE of the two following output modes:\n\n"
                "MODE 1 — suggest a diagnostic action (suggestion_type='action'):\n"
                "  Use this when you need more information before committing to a diagnosis.\n"
                "  Provide action_type, action_target, and action_description.\n"
                "  - action_type: 'Replace' (swap a component), 'Adjust' (tune/repair without "
                "replacing), 'Test' (measure/manipulate with tools), or 'Observe' (visually "
                "inspect without tools). The ordering from cheapest to most expensive is roughly "
                "Observe < Test < Adjust < Replace — use your common sense to estimate the "
                "specific cost of each action you propose. Note that it is the action_description "
                " not action_type that will be interpreted precisely and translated into a " 
                "sequence of concrete actions to determine what actually happens.\n"
                "  - action_target: Name specific components from the system description (e.g., "
                "'the battery', 'the main lamp', 'the control relay'), NOT module names "
                "(e.g. NOT 'power supply module', NOT 'control module'). Enclosures themselves ARE "
                "valid targets for actions (e.g., 'invert the power supply cube', 'rotate the control "
                "module'). Some actions may have multiple "
                "targets (e.g., swap two cables).\n"
                "  - action_description: A brief description of the action. "
                "Examples: 'Measure voltage at the battery', 'Test continuity of the main lamp', "
                "'Replace the control relay'. It is possible to target multiple components (e.g., 'inspect "
                "all connections of X') and suggest a series of actions ('do this and do that'), "
                "but do try to keep it simple and concise.\n"
                "  - suspected_components: must be left None or empty in this case.\n\n"
                "NOTE ON MODE 1: Replace actions that repair a component are permanent and "
                "carry over for the rest of the session. Non-replacement actions may be reset "
                "between turns (the system is restored to its fault state to keep observations "
                "consistent). Consequently, if your goal is to verify whether replacing a "
                "specific component restores the system, use MODE 2 — that is exactly what "
                "hypothesis testing is designed for.\n\n"
                "MODE 2 — declare a fault hypothesis (suggestion_type='hypothesis'):\n"
                "  Use this when you are sufficiently confident about which specific components "
                "are faulty. The service agent will attempt to repair/replace those components "
                "and report whether the system is restored. This is the preferred way to finish.\n"
                "  Provide suspected_components: a list of actual component/enclosure names as they "
                "appear in the system description (e.g., 'the battery', 'the main lamp', 'the load diode', "
                "'the power supply cube'). Do NOT invent component names or provide difficult to understand descriptions.\n"
                "  A wrong hypothesis carries a significant cost, so do not guess blindly "
                "(in particular avoid targeting whole modules instead of components) — "
                "but also do not over-collect information when you have enough to decide: "
                "note that you may or may not be under a time and/or action number limit, so "
                "be efficent and to the point.\n\n"
                "IMPORTANT NOTE ON MODE 2: if you suspect a configuration error (e.g. two cables "
                "are swapped), then provide all the components involved in the suspected_components "
                "list. ALWAYS provide one or more components in suspected_components (even though "
                "it may happen they do not suffer from an internal fault, but from some configuration "
                "issue)."
        )

        if self._optimized_history:
            _instructions = get_system_description_instructions(
                self.state.general_system_description
            ) + [{"type": "input_text", "text": "\n\n" + _task_instructions}]
        else:
            _instructions = _task_instructions

        self.constrainedoutputdiagnoser_v3 = Agent(
            name="ConstrainedOutputDiagnoser",
            instructions=_instructions,
            model=_make_model(self.configuration.ASSISTANT_CONFIG.get("model", self.configuration.DEFAULT_LLM_MODEL)),
            output_type=DiagnosticSuggestion,
        )

    @property
    def description(self) -> str:
        return super().description + "_" + self.configuration.ASSISTANT_CONFIG.get("model", self.configuration.DEFAULT_LLM_MODEL)

    async def setup(self, observations: list[Observation]) -> None:
        self.state.initial_observations = observations.copy()
        self.state.conversation_history = get_updated_conversation(
            self.state.conversation_history, f"The system is experiencing the following symptoms:\n{format_list(self.state.initial_observations)}")

    async def record_action_outcome(self, last_outcome) -> None:
        await super().record_action_outcome(last_outcome)
        self.state.conversation_history = get_updated_conversation(
            self.state.conversation_history, f"Someone executed a (additional) diagnostic action on the system. The action (type, target_component, description, outcome) was: \nTYPE: {last_outcome.action.type},\nTARGET: {last_outcome.action.target},\nDESCRIPTION: {last_outcome.action.description},\nRESULT: {last_outcome.outcome}\n")

    async def suggest_action(self) -> DiagnosticAction | DiagnosticFaultHypothesis:
        from environment_classes import LLMTruncationError
        try:
            raw: DiagnosticSuggestion = await possibly_cached_runner_run(
                self.constrainedoutputdiagnoser_v3,
                input=self.state.conversation_history,
                cached=self.configuration.USE_CACHE,
            )
        except Exception as exc:
            import json as _json
            # Walk the cause chain — the JSONDecodeError may be wrapped by httpx/openai/agents
            cause = exc
            while cause is not None:
                if isinstance(cause, _json.JSONDecodeError):
                    self.logger.error(f"LLM returned truncated/invalid JSON: {exc}")
                    raise LLMTruncationError(str(exc)) from exc
                cause = getattr(cause, "__cause__", None) or getattr(cause, "__context__", None)
            msg = str(exc)
            if "json_invalid" in msg or "EOF while parsing" in msg or "Invalid JSON" in msg or "Expecting value" in msg:
                self.logger.error(f"LLM returned truncated/invalid JSON: {exc}")
                raise LLMTruncationError(msg) from exc
            raise
        self.logger.debug(
            f"INPUT [tail only with new content. Full content is concatenation of all previous log entries]: \n{format_conversation_history([self.state.conversation_history[-1]])}\n")
        self.logger.info(f"OUTPUT: {str(raw)}")
        if self._optimized_history:
            self.state.conversation_history = append_assistant_turn(
                self.state.conversation_history, str(raw)
            )
        if raw.suggestion_type == "action" and raw.action_target is not None:
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
