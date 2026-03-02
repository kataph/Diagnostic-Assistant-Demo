from typing import Optional
from environment_classes import AssistantState, CLIHumanIO, RootCauseDescription, ServiceAgent, SystemDescription, Observation, DiagnosticAction, DiagnosticActionResult, VoiceHumanIO
from Utilities.caching import async_disk_cache_CLI

class ServiceAgentHuman(ServiceAgent):
    
    def __init__(self, configuration):
        super().__init__(configuration)
        self.STOP_WORDS = ['stop', 'stop.', 'stop,']
        self.AFFERMATIVE_WORDS = ['y', 'y.', 'yes', 'yes!', 'yes.']
        match self.configuration.INTERFACE_MODE:
            case "cli":
                self.io = CLIHumanIO()
            case "voice":
                self.io = VoiceHumanIO()
            case _:
                raise ValueError(f"Unknow interface mode: {self.configuration.INTERFACE_MODE}")
    
    # @async_disk_cache_CLI #TODO: comment when not saving time...
    async def collect_initial_observations(
        self,
        system: SystemDescription,
        root_cause_description: Optional[RootCauseDescription]
    ) -> list[Observation]:
        # if root_cause_description and root_cause_description.symptoms_descriptions: TODO: this is buggy and conceptually wrong
        #     return root_cause_description.symptoms_descriptions
        await self.io.prompt(f"\n=== Service phase: initial observations for system ===")
        await self.io.prompt("Describe initial observations about the system ('stop' to finish):")

        result: list[Observation] = []
        while True:
            line = (await self.io.read_line("> ")).strip()
            if line.lower() in self.STOP_WORDS:
                break
            if line == "":
                await self.io.prompt("Ignoring empty input...")
                continue
            result.append(
                Observation(description=line)
            )
        self.logger.info(f"The human service agent collected the following initial observations: {str(result)}")
        return result

    async def execute_action(self, system: SystemDescription, action: DiagnosticAction, root_cause_description: Optional[RootCauseDescription]) -> DiagnosticActionResult:
        await self.io.prompt(f"\nSuggested action: {action.get_name()}")
        if action.description:
            await self.io.prompt(f"  -> {action.description}")
        else:
            await self.io.prompt(f"  -> the action has no description")
            
        await self.io.prompt("(Execute the action and describe the outcome)")

        outcome_text = (await self.io.read_line("Outcome> ")).strip()

        dar = DiagnosticActionResult(action=action, outcome=outcome_text)
        self.logger.info(f"The human service agent executed an action and obtained an outcome: {dar}")
        return dar

    async def decide_finish(self, system: SystemDescription, state: AssistantState, root_cause_description: Optional[RootCauseDescription]) -> tuple[bool,Optional[RootCauseDescription]]:
        answer = (await self.io.read_line("\nDo you consider the diagnosis done now? [y/else]\n> ")).strip().lower() 
        if answer not in self.AFFERMATIVE_WORDS:
            return (False, None)
        # Comment 3 lines above and uncomment line below for faster interaction
        # return (False, None)
        
        answer = (await self.io.read_line("\nDo you want to record the putative root cause? [y/else]\n> ")).strip().lower()
        if answer not in self.AFFERMATIVE_WORDS:
            return (True, None)

        return (True, RootCauseDescription(
            root_cause_description_proper = (await self.io.read_line("Describe the root cause you believe has determined the system failure: ")).strip(),
            symptoms_descriptions = [],
            notes = (await self.io.read_line("Optional notes: ")).strip() or None,
        ))