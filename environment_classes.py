from __future__ import annotations

import asyncio
import logging

from abc import ABC, abstractmethod
from typing import Iterator, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, RootModel

from configuration import Configuration
from voice_client import send_prompt, get_user_text


class SystemDescription(BaseModel):
  text_input: str
  file_id: str | None = None
  
empty_sys_descr = SystemDescription(text_input="", file_id="")  
            
# class VirtualSystemDescription(SystemDescription):
#   root_cause_description: RootCauseDescription
# class PhysiscalSystemDescription(SystemDescription):
#     pass


class SymptomDescription(RootModel[str]):
  def __str__(self):
     return f"'{self.root}'"

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

ACTION_COST_MAP = {
    'Replace': 10,
    'Adjust': 5,
    'Test': 3,
    'Observe': 1,
}

diagnosticActionTypes = Literal['Replace', 'Adjust', 'Observe', 'Test']

class DiagnosticAction(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    type: diagnosticActionTypes
    target: str
    description: Optional[str] = None
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

class DiagnosticActionResult(BaseModel):
  action: DiagnosticAction
  outcome: str
  simplified_outcome: Optional[Literal['Anomalous', 'Nominal']] = None
  
  def __str__(self):
    return (f"{{action name: {self.action.get_name()}, "+(f"action description: '{self.action.description}'" if self.action.description else "")+f" action result description: '{self.outcome}'}}")

class TesterCostrainedOutputText(BaseModel):
  system_works_again: bool
  diagnostic_actions_results: list[TextDiagnosticActionResult]
  def __str__(self):
     return "system_works_again: "+str(self.system_works_again)+"\ndiagnostic_actions_results:\n    "+"\n    ".join([str(x) for x in self.diagnostic_actions_results])
class TesterCostrainedOutput(BaseModel):
  system_works_again: bool
  diagnostic_actions_results: list[DiagnosticActionResult]
  def __str__(self):
     return "system_works_again: "+str(self.system_works_again)+"\ndiagnostic_actions_results:\n    "+"\n    ".join([str(x) for x in self.diagnostic_actions_results])
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
    root_cause_description_proper : FaultDescription
    symptoms_descriptions: Optional[SymptomDescriptions] = None
    notes: Optional[str] = None
    def __repr__(self):
        return (f"root cause:\n{self.root_cause_description_proper}" +
                f"\nsymptoms:\n{self.symptoms_descriptions}" if self.symptoms_descriptions else "" +
                f"\nnotes:\n{self.notes}" if self.notes else "")
    def one_liner_repr(self):
        return (f"root cause: {self.root_cause_description_proper} " +
                (f"symptoms: '{self.symptoms_descriptions}' " if self.symptoms_descriptions else "") +
                (f"notes: '{self.notes}'" if self.notes else ""))
    def __str__(self):
        return self.__repr__()


class RootCauseHypothesis(BaseModel):
    component: str
    failure_mode: str
    confidence: float = Field(ge=0, le=1)
    proposed_by: str = "assistant"

### Human interface thorugh either CLI or voice
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
        loop = asyncio.get_event_loop()
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


class DiagnosticPlan(BaseModel):
    model_config = ConfigDict(extra="allow") # Allows extra fields
    
    # actions: List[DiagnosticAction]
    # current_index: int = 0
    # status: PlanStatus = PlanStatus.ONGOING
    # hypotheses: List[RootCauseHypothesis] = Field(default_factory=list)
    
    @abstractmethod
    def get_next_action(self) -> DiagnosticAction | None:
        ...


class AssistantState(BaseModel):
    # model_config = ConfigDict(extra="allow") # Allows extra fields

    general_system_description: SystemDescription = None
    initial_observations: List[Observation] = Field(default_factory=list)
    diagnostic_scenario_memory: List[DiagnosticActionResult] = Field(default_factory=list)

    user_finished: bool = False
    user_confirmed_root_cause: Optional[RootCauseDescription] = None
    
    def __str__(self):
        return f"model_config: {self.model}"

# All diagnostic-scenario-environment-roles classes receive a configuration and may log info about their behavior
class ThingThatLogs(ABC):
    
    def _setup_logger(self, configuration: Configuration) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(configuration.LOG_LEVEL) # INFO is 20
        
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
      - record_outcome(last_outcome) to update the assistant state
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
    async def setup(self, observations: List[Observation]) -> None:
        """
        Updates assistant internal state with initial observations.
        """

    # @abstractmethod
    async def record_outcome(self, last_outcome: DiagnosticActionResult) -> None:
        """
        Update the assistant state given the last outcome. 
        """
        self.state.diagnostic_scenario_memory.append(last_outcome)
        
    @abstractmethod
    async def suggest_action(self) -> Optional[DiagnosticAction]:
        """
        Given the assistant internal state, the next suggested action is returned, or None if no action is available.
        """

    def finish_session(self, root_cause: Optional[RootCauseDescription]) -> None:
        """
        Called by the *service agent* when it decides diagnosis is done.
        """
        self.user_finished = True
        if root_cause:
            self.state.user_confirmed_root_cause = root_cause
            self.logger.info(f"Session ended by user decision. Recorded root cause is >>{self.state.user_confirmed_root_cause.one_liner_repr()}<<")
        else:
            self.logger.info(f"Session ended by user decision. Root cause not recorded")
        


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
    ) -> List[Observation]:
        ...

    @abstractmethod
    async def execute_action(self, 
                       system: SystemDescription,
                       action: DiagnosticAction, 
                       root_cause_description: Optional[RootCauseDescription]) -> DiagnosticActionResult:
        ...

    @abstractmethod
    async def decide_finish(self, system: SystemDescription, state: AssistantState, root_cause_description: Optional[RootCauseDescription]) -> tuple[bool, Optional[RootCauseDescription]]:
        """
        Return True with optional FaultDescription when the agent decides the diagnosis is done,
        or False with None to continue.
        """


# =========
# Orchestrator
# =========

async def run_diagnostic_scenario(
    system: SystemDescription,
    saboteur: Saboteur,
    service_agent: ServiceAgent,
    assistant: DiagnosticAssistant,
    scenario_logger: logging.Logger
) -> None:
    scenario_logger.info(f"A diagnostic scenario has been started with a saboteur of type {saboteur.description}, a service agent of type {service_agent.description}, and an assistant of type {assistant.description}")
    
    # 1) Sabotage phase
    sabotage_root = await saboteur.sabotage(system)

    # 2) Initial observations
    initial_obs = await service_agent.collect_initial_observations(system, sabotage_root)
    await assistant.setup(initial_obs)
    scenario_logger.info(f"Diagnostic assistant has finished setup. Starting diagnostic loop...")

    # 3) Diagnostic loop
    last_outcome: Optional[DiagnosticActionResult] = None
    suggested_actions_history: list[DiagnosticActionResult] = []

    while True:
        action = await assistant.suggest_action()
        if action is None:
            print("\nAssistant has no further actions to suggest.")
        else:
            last_outcome = await service_agent.execute_action(system, action, sabotage_root)
            suggested_actions_history.append(last_outcome)
            await assistant.record_outcome(last_outcome)

        # Ask service agent if it wants to finish
        user_finished, user_root = await service_agent.decide_finish(system, assistant.state, sabotage_root)
        if user_finished:
            assistant.finish_session(user_root)
            break

        if action is None:
            # No further actions and agent wants to continue: auto-stop here,
            # or you could add custom behavior.
            print("No actions and agent chose to continue; that is, the user will continue the diagnosis and the assistant cannot help. Ending session automatically.")
            break

    # 4) Summary
    
    if rc:=assistant.state.user_confirmed_root_cause:
        scenario_logger.info(f"Service agent confirmed root cause: {rc.root_cause_description_proper}")
    else:
        scenario_logger.info("Service agent ended session without recording a root cause.")
    cost_vector = [x.action.get_cost() for x in suggested_actions_history]
    scenario_logger.info(f"Cost vector: {cost_vector}")
    scenario_logger.info(f"Total cost: {sum(cost_vector)} --- Number of suggestions: {len(cost_vector)} ")
    scenario_logger.debug(f"Diagnostic memory: {assistant.state.diagnostic_scenario_memory} ")

    # print("\nFull assistant state:")
    # print(assistant.state.model_dump(indent=2))


if __name__ == "__main__":
    pass