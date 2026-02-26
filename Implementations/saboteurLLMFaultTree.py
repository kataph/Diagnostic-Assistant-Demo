import logging
import random
import traceback

from pydantic import BaseModel
from agents import Agent
from PrettyPrint import PrettyPrintTree

from Utilities.agents_boilerplate import get_updated_conversation
from configuration import Configuration
from environment_classes import Saboteur, SystemDescription, SymptomDescriptions, RootCauseDescription
from Utilities.caching import possibly_cached_runner_run, add_disk_cacheing_option_for_methods


class FaultTreeEvent(BaseModel):
    name: str
    description: str

    def __str__(self):
        return f"{{name: {self.name}, description: {self.description}}}"

class AndOrGate(BaseModel):
    gate: str
    input_arguments: list[str]
    output_argument: str

    def __str__(self):
        return f"{{{str(self.input_arguments)} -- {self.gate} --> {self.output_argument}}}"


class FaultTree(BaseModel):
    """A class representing a fault tree. The tree events are listed in the 'events' property. The 'and_or_gates' collects all gates of the tree."""
    events: list[FaultTreeEvent]
    and_or_gates: list[AndOrGate]

    def get_event(self, name: str) -> FaultTreeEvent:
        for event in self.events:
            if event.name == name:
                return event

    def get_children(self, node: FaultTreeEvent) -> list[FaultTreeEvent]:
        for gate in self.and_or_gates:
            if gate.output_argument == node.name:
                return [self.get_event(name) for name in gate.input_arguments]
        return []

    def get_value(self, node: FaultTreeEvent, add_description=False) -> str:
        for gate in self.and_or_gates:
            if gate.output_argument == node.name:
                gate_type = gate.gate
                return node.name + "--" + node.description + "--" + gate_type if add_description else node.name + '--' + gate_type
        return node.name + "--" + node.description if add_description else node.name

    def get_top_event(self) -> FaultTreeEvent | list[FaultTreeEvent]:
        gate_inputs = set(
            name for gate in self.and_or_gates for name in gate.input_arguments)
        top_events = self.events.copy()
        for event in self.events:
            if event.name in gate_inputs:
                top_events.remove(event)
        if len(top_events) > 1:
            print(
                f'WARNING: Length of top events is not 1, something went wrong! top events is {top_events}, complete FT is\n\n{self}\n\n, I am returning all of them!')
            return top_events
        return top_events[0]

    def select_all_basic_events(self) -> list[FaultTreeEvent]:
        # basic events are those that are in no gate output. Returns the basic events as a set of event names
        gate_outputs = set(gate.output_argument for gate in self.and_or_gates)
        basic_events = self.events.copy()
        for event in self.events:
            if event.name in gate_outputs:
                basic_events.remove(event)
        return basic_events

    def pretty_print_FT(self, return_instead_of_print=True) -> None | str:
        return (PrettyPrintTree(self.get_children, self.get_value, return_instead_of_print=return_instead_of_print, color='', border=True)(self.get_top_event())) if self.get_top_event() else print(f"Fault tree malformed, FT dump is\n{self}")
    
    # this function must also be cached, since it stocastic
    @add_disk_cacheing_option_for_methods
    def select_one_random_basic_event(self) -> FaultTreeEvent:
        # basic events are those that are in no gate output. Chooses one at random and returns its name 
        basic_events = self.select_all_basic_events()
        return random.choice(basic_events)
    
    def validate_fault_tree_gpt(self) -> None:
        """
        Validates structural integrity of a fault tree.
        Raises AssertionError if malformed.
        """

        # 1. All gate references must exist as events
        event_names: set[str] = {event.name for event in self.events}

        for gate in self.and_or_gates:
            assert gate.output_argument in event_names, (
                f"Gate output '{gate.output_argument}' not defined as event"
            )

            for inp in gate.input_arguments:
                assert inp in event_names, (
                    f"Gate input '{inp}' not defined as event"
                )

        # 2. Exactly one top event must exist
        top_event: FaultTreeEvent | list[FaultTreeEvent] = self.get_top_event()
        assert isinstance(top_event, FaultTreeEvent), f"More than one top event was found: only one of these should be a top-event: {top_event}"

        # 3. No cycles (DFS)
        visited: set[str] = set()
        stack: set[str] = set()

        def dfs(event: FaultTreeEvent) -> None:
            if event.name in stack:
                raise AssertionError(f"Cycle detected at event '{event.name}'")

            if event.name in visited:
                return

            stack.add(event.name)
            visited.add(event.name)

            children: list[FaultTreeEvent] = self.get_children(event)
            for child in children:
                dfs(child)

            stack.remove(event.name)

        dfs(top_event)

        # 4. All events must be reachable from top event
        reachable: set[str] = set()

        def mark_reachable(event: FaultTreeEvent) -> None:
            if event.name in reachable:
                return

            reachable.add(event.name)

            children: list[FaultTreeEvent] = self.get_children(event)
            for child in children:
                mark_reachable(child)

        mark_reachable(top_event)

        unreachable: set[str] = event_names - reachable
        assert not unreachable, f"Unreachable events found: {unreachable}"

        # 5. Pretty print must not crash
        try:
            self.pretty_print_FT()
        except Exception as e:
            raise AssertionError(f"Pretty print failed: {e}") from e
    

faultTreeGenerator = Agent(
    name="ftgenerator",
    instructions="""You are an expert engineer whose duty is to study the documentation available about an engineered system and to prouce a fault tree. Fault trees are representations of the various parallel and sequential combinations of faults that can result in the occurrence of a predefined undesired event (top event). The fault tree is constructed by identifying the top event and then determining all the possible causes that could lead to that event, breaking them down into more specific sub-events until reaching basic events that cannot be further subdivided. In your case, the top event is the generic failure of the engineered system. You have to output a json file where the (a) all the events of the fault tree are listed (each with a name and description property) and (b) all the AND and OR gates are in a list containing objects such as 
    {
        \"gate\": \"AND\",
        \"input_arguments\": [\"event1\", \"event2\"],
        \"output_argument\": \"event3\"
     }. 
    Ensure that the fault tree is comprehensive, logically structured, and accurately reflects the potential failure modes of the system based on the provided documentation. Use your own knowledge and/or web searches about engineering systems to supplement the information from the documentation where necessary.
    In particular, ensure that there is a unique top event, and that all events appear at least once in a gate.""",
    model="gpt-4.1",
    output_type=FaultTree,
)

symptomGenerator = Agent(
    name="SymtomGenerator",
    instructions="""You are an expert reliability engineer. You will receive in input 
  (i) a description of an engineered system, together with 
  (ii) a specific fault (root cause) that the system is suffering from

  Your job is to output a description of the symptoms that the system will exhibit in the presence of such a fault, as a list of strings.

  Note that the symptoms can be multiple and can affect multiple components of the system. Also the sysmptoms must be observable effects of the fault, i.e., things that can be measured or perceived when interacting with the system. Finally, ensure that the symptoms you select are plausible given the description of the system provided in input, and that there is a causal relationship between the fault and the symptoms you describe. And, most importantely, do not include the fault itself among the symptoms. 
  """,
    model="gpt-4.1",
)

class SymptomGeneratorOutput(BaseModel):
    symptom_descriptions: SymptomDescriptions
        
symptomGenerator_v2 = Agent(
    name="SymtomGenerator",
    instructions="""You are an expert reliability engineer. You will receive in input 
  (i) a description of an engineered system, together with 
  (ii) a specific fault (root cause) that the system is suffering from

  Your job is to output a description of the symptoms that the system will exhibit in the presence of such a fault, as a list of strings.

  Note that the symptoms can be multiple and can affect multiple components of the system. Also the sysmptoms must be observable effects of the fault, i.e., things that can be measured or perceived when interacting with the system. Finally, ensure that the symptoms you select are plausible given the description of the system provided in input, and that there is a causal relationship between the fault and the symptoms you describe. And, most importantely, do not include the fault itself among the symptoms. 
  Please, organize your output as a list of entries, one brief text describing one symptom per entry. 
  """,
    model="gpt-4.1",
    output_type=SymptomGeneratorOutput
)

class SaboteurLLMFaultTree(Saboteur):

    FT: FaultTree = None
    MAX_RETRIES = 3

    async def generate_fault_tree_with_self_correction(
        self,
        faultTreeGenerator: Agent,
        conversation_start: str,
    ) -> FaultTree:
        """
        Generates a FaultTree using the LLM with automatic self-correction.
        Retries generation if validation fails.
        """

        conversation: str = conversation_start
        last_exception: Exception | None = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            exceptions = []
            try:
                ft: FaultTree = await possibly_cached_runner_run(
                    agent=faultTreeGenerator,
                    input=conversation,
                    cached=self.configuration.USE_CACHE,
                )

                # Validate (raises if malformed)
                ft.validate_fault_tree_gpt()

                # Success
                self.logger.info(f"Fault tree generator was successfull at attempt number {attempt}")
                return ft

            except Exception as e:
                exceptions.append(e)
                last_exception = e
                # error_stack: str = traceback.format_exc()
                # self.logger.error(f"Error in FT generation: last exception is {last_exception}, error stack is: {error_stack}, use cache is {self.configuration.USE_CACHE}")
                self.logger.error(f"Error in FT generation: {last_exception}")

                # Append structured correction feedback
                conversation = get_updated_conversation(conversation,
                    "\n\n--- VALIDATION ERROR ---\n"
                    "Your previously generated fault tree is malformed.\n"
                    "Fix the issues described below and regenerate the entire fault tree.\n\n"
                    # f"{error_stack}\n"
                    f"{last_exception}\n"
                    "Return ONLY the corrected FaultTree object.\n"
                )

        # If we reach here → all retries failed
        self.logger.error(f"Fault tree generation failed after {self.MAX_RETRIES} attempts.")
        raise RuntimeError(
            f"Fault tree generation failed after {self.MAX_RETRIES} attempts."
        ) from last_exception

    async def sabotage(self, description: SystemDescription) -> RootCauseDescription:
        input_item = {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": description.text_input
                }
            ]
        }
        if description.file_id:
            input_item['content'].append(
                {
                    "type": "input_image",
                    "file_id": description.file_id
                },
            )
        conversation_start = [
            input_item
        ]
        self.logger.info('The agent is generating the fault tree...: ') 
        self.FT = await self.generate_fault_tree_with_self_correction(
            faultTreeGenerator=faultTreeGenerator,
            conversation_start=conversation_start,
        )
        determined_fault: FaultTreeEvent = self.FT.select_one_random_basic_event(cached = self.configuration.USE_CACHE)
        self.logger.debug('FT generator output: ' +
                    self.FT.pretty_print_FT(True))
        self.logger.info('All basic FT events: ' + 
                    str(self.FT.select_all_basic_events()))
        self.logger.info('Basic event randomly chosen: ' + str(determined_fault))
        
        input_for_symptom_generator = conversation_start + [{
            "role": "user",
            "content": [
                {"type": "input_text",
                 "text": f"The system is experiencing the following fault: \nname: {determined_fault.name} \ndescription: {determined_fault.description}"}
            ]
        }]
        o: SymptomGeneratorOutput = await possibly_cached_runner_run(agent=symptomGenerator_v2, input=input_for_symptom_generator, cached=self.configuration.USE_CACHE)
        symptom_descriptions = o.symptom_descriptions
        self.logger.info(f'Corresponding symptoms ({len(symptom_descriptions)} symptoms): ' + symptom_descriptions.one_line_repr())
        return RootCauseDescription(root_cause_description_proper = str(determined_fault), symptom_descriptions = symptom_descriptions)
