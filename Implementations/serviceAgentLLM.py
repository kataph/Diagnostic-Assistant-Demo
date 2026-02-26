from typing import Optional
from agents import Agent
from pydantic import BaseModel, RootModel

from Utilities.caching import possibly_cached_runner_run
from environment_classes import RootCauseDescription, ServiceAgent, SystemDescription, Observation, DiagnosticActionResult, DiagnosticAction, AssistantState

from Utilities.agents_boilerplate import get_conversation_start

class LightDiagnosticActionResult(BaseModel):
    action_type: str
    action_target: str
    action_outcome: str
class TesterCostrainedOutput(BaseModel):
  system_works_again: bool
  diagnostic_actions_results: list[LightDiagnosticActionResult]
  def __str__(self):
     return "system_works_again: "+str(self.system_works_again)+"\ndiagnostic_actions_results:\n    "+"\n    ".join([str(x) for x in self.diagnostic_actions_results])
class TesterCostrainedLightOutput(BaseModel):
  system_works_again: bool
  action_outcome: str
  def __str__(self):
     return "system_works_again: "+str(self.system_works_again)+f"\ndiagnostic_action_outcome: {self.action_outcome}"

class ObservationList(BaseModel):
    observations: list[str]

    def __str__(self) -> str:
        return str(self.observations)

    def __repr__(self) -> str:
        return self.__str__()


serviceAgent = Agent(
  name="CostrainedInputTester",
  instructions="""You are an engineer expert in simulating diagnosis scenarios. You will receive in input a description of an engineered system and a description of the fault that the system is currently suffering from. Your job is to train another engineer in diagnosing the system. To do so, the trainee will hypothesize diagnostic actions to carry out on the system to determine the root cause of the fault. The diagnostic actions are given as a list of (action_type, action_target, action_description) tuples. You have to answer your colleague with the results of the diagnostic actions. When the trainee executes a diagnostic action that repairs the system, record it in the boolean field 'system_works_again' (otherwise such field must be set to false). 
  
  Your output must be as follows:
  (1) system_works_again: a boolean field indicating if ther system has been repaired successfully by the diagnostic 
  (2) a list of (action_type, action_target, action_outcome) tuples, where action_type and action_target fields must be the same as the ones you got in input and action_outcome must be a sentence describing what the diagnostic action result was. 
  Always respond concisely with just the results of the diagnostic actions without further inference or considerations. Also do not reveal directly what the root cause is, only that you answer describing the results of the diagnostic actions. """,
  model="gpt-4.1",
  output_type=TesterCostrainedOutput,
)
serviceAgent_v2 = Agent(
  name="CostrainedInputTester",
  instructions="""You are an engineer expert in simulating diagnosis scenarios. You will receive in input a description of an engineered system and a description of the fault that the system is currently suffering from. Your job is to train another engineer in diagnosing the system. To do so, the trainee will hypothesize a single diagnostic action to carry out on the system to determine the root cause of the fault. The diagnostic action will be given as an (action_type, action_target, action_description) tuple. You have to answer your colleague with the results of the diagnostic action. When the trainee executes a diagnostic action that repairs the system, record it in the boolean field 'system_works_again' (otherwise such field must be set to false). 
  
  Your output must be as follows:
  (1) system_works_again: a boolean field indicating if ther system has been repaired successfully by the diagnostic 
  (2) action_outcome: a sentence a sentence describing what the diagnostic action result was. 
  Always respond concisely with just the results of the diagnostic actions without further inference or considerations. Also do not reveal directly what the root cause is, only that you answer describing the results of the diagnostic actions. """,
  model="gpt-4.1",
  output_type=TesterCostrainedLightOutput,
)
serviceAgent_initialObservations = Agent(
  name="CostrainedInputTester_start",
  instructions="""You are an engineer expert in simulating diagnosis scenarios. You will receive in input a description of an engineered system and a description of the fault that the system is currently suffering from. Your job is to train another engineer in diagnosing the system. To do so, the trainee will hypothesize a single diagnostic action to carry out on the system to determine the root cause of the fault. In order to make the first hypothesis, the trainee needs to be supplied a series of observations about the system: using your knowledge of the system behavior, write down a list relevant remarks about the system. Such remarks an be both about the part of the current behavior of the system that is anomalous or that is nominal. You get to decide what is relevant and what is not. """,
  model="gpt-4.1",
  output_type=ObservationList,
)


class ServiceAgentLLM(ServiceAgent):

    def __init__(self, configuration):
        super().__init__(configuration)
        self.patience_level = configuration.MAX_NUMBER_OF_ROUNDS - 1 # simulates an user patience
        self.annoyance_level = 0 # simulates an user patience
        
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
        
        o: ObservationList = await possibly_cached_runner_run(serviceAgent_initialObservations, input=self._get_input_for_initial_observations(system, root_cause_description), cached=self.configuration.USE_CACHE)
        self.logger.info(f"First observations about the system: {str(o)}")
        return o        

    async def execute_action(self, system: SystemDescription, action: DiagnosticAction, root_cause_description: RootCauseDescription) -> DiagnosticActionResult:
        
        def __init__(self):
            self.system_works_again = False
            
        conversation_history_service = (
            self._get_input_for_initial_observations(system, root_cause_description) + 
            # For the tester, the conversation history does not grow: it does not need to rememer previous tests, and it would also risk losing memory or becoming biased
            [{
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"The trainee says:\n{action.get_full_repr()}"
                    }
                ]
            }])

        o: TesterCostrainedLightOutput = await possibly_cached_runner_run(serviceAgent_v2, input=conversation_history_service, cached=self.configuration.USE_CACHE)
        self.system_works_again = o.system_works_again        
        self.logger.info(f"Agent executed action: {action.get_name()} with outcome {o.action_outcome}")
        self.logger.info(f"Agent thinks that system works again? {o.system_works_again}")
        return DiagnosticActionResult(action=action, outcome=o.action_outcome)        

    async def decide_finish(self, system: SystemDescription, state: AssistantState, root_cause_description: RootCauseDescription) -> tuple[bool, None]:
        # plan = state.plan
        # if plan and plan.status == PlanStatus.EXHAUSTED:
        #     return self._true_root_cause
        # return None
        if self.annoyance_level >= self.patience_level:
            self.logger.info("The Service agent decides it is not worth using the tool...")
            return (True, None) 
        self.annoyance_level += 1
        return (self.system_works_again, None)