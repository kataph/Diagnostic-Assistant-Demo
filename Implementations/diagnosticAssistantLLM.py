from agents import Agent
from pydantic import Field
from configuration import Configuration
from environment_classes import AssistantState, DiagnosticAction, DiagnosticAssistant, Observation, SystemDescription, ACTION_COST_MAP
from Utilities.agents_boilerplate import get_conversation_start, get_updated_conversation, update_conversation
from Utilities.formatting import format_conversation_history, format_list
from Utilities.caching import possibly_cached_runner_run


class AssistantStateLLM(AssistantState):
    conversation_history: list[dict] = Field(default_factory=list)

class DiagnosticAssistantLLM(DiagnosticAssistant):
    
    def __init__(self, description: SystemDescription, configuration: Configuration):
        super().__init__(description, configuration)
        self.state = AssistantStateLLM(
            general_system_description=description,
            conversation_history = get_conversation_start(self.state.general_system_description),
            )
        self.costrainedoutputdiagnoser_v3 = Agent(
        name="CostrainedOutputDiagnoser",
        instructions=("""You are an expert reliability engineer. You will receive in input a description of an engineered system that is suffering from some fault, and possibly a history of diagnostic actions executed on the system and their results.
        Your job is to find the root cause of such a fault. 
        To do so, you can suggest diagnostic actions to be executed on the system, and you will receive in input the results of such diagnostic actions.
        You can repeat this process multiple times. You will have discovered what the root cause is when the system will start working again.
        """ +
        "Your output must be a single diagnostic action." + 
        """A action is a described by the following properties: 
            - type: a field taking one of the following values: 'Replace', 'Adjust', 'Test', 'Observe'. The field value is assigned depending on what the action consist of: the tye 'Replace' is for actions involving swapping a component with another component. 'Adjust' is for adjusting, refitting, tuning, reconfiguring, or repairing a component without replacing it. 'Test' is for testing or measuring a component using tools and/or by actively manipulating it. Finally 'Observe' is for visually inspecting or analyzing a component without tools and without physically manipulating it.
            - target: a part of the system that will be the target of the action
            - description: a brief and succint natural language description of the action
        """+
        f"When suggesting actions, do keep in mind that (a) you can give in output at most one action of a given type and target and (b) different actions have different cost to be executed. In particular, we stipulate that the action cost is function only of action_type: 'Replace' has a cost of {str(ACTION_COST_MAP['Replace'])}, 'Adjust' has a cost of {str(ACTION_COST_MAP['Adjust'])}, 'Test' has a cost of {str(ACTION_COST_MAP['Test'])}, and 'Observe' has a cost of {str(ACTION_COST_MAP['Observe'])}."),
        model=self.configuration.LLM_ASSISTANT_MODEL, # default "gpt-4.1",
        output_type=DiagnosticAction
        )
        
    @property
    def description(self) -> str:
        super().description + "_" + self.configuration.LLM_ASSISTANT_MODEL
        
    async def setup(self, observations: list[Observation]) -> None:
        self.state.initial_observations = observations.copy()
        self.state.conversation_history = get_updated_conversation(self.state.conversation_history, f"The system is experiencing the following symptoms:\n{format_list(self.state.initial_observations)}")
    
    async def record_outcome(self, last_outcome) -> None:
        await super().record_outcome(last_outcome)
        self.state.conversation_history = get_updated_conversation(self.state.conversation_history, f"Someone executed a (additional) diagnostic action on the system. The action (type, target_component, description, outcome) was: \nTYPE: {last_outcome.action.type},\nTARGET: {last_outcome.action.target},\nDESCRIPTION: {last_outcome.action.description},\nRESULT: {last_outcome.outcome}.\n")
        
    async def suggest_action(self) -> DiagnosticAction:
        suggested_action: DiagnosticAction = await possibly_cached_runner_run(self.costrainedoutputdiagnoser_v3, input=self.state.conversation_history, cached=self.configuration.USE_CACHE)
        self.logger.debug(f"INPUT: \n{format_conversation_history(self.state.conversation_history)}\n")
        self.logger.info(f"OUTPUT: \n{str(suggested_action)}\n\n")
        return suggested_action
    
    def finish_session(self, root_cause):
        super().finish_session(root_cause)