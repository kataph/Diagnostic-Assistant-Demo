from typing import Optional
from environment_classes import AssistantState, RootCauseDescription, ServiceAgent, SystemDescription, Observation, DiagnosticAction, DiagnosticActionResult
from Utilities.caching import async_disk_cache_CLI

class ServiceAgentHuman(ServiceAgent):
    # @async_disk_cache_CLI #TODO: comment when not saving time...
    async def collect_initial_observations(
        self,
        system: SystemDescription,
        root_cause_description: Optional[RootCauseDescription]
    ) -> list[Observation]:
        # if root_cause_description and root_cause_description.symptoms_descriptions: TODO: this is buggy and conceptually wrong
        #     return root_cause_description.symptoms_descriptions
        print(f"\n=== Service phase: initial observations for system ===")
        print("Describe initial observations about the system ('stop' to finish):")

        result: list[Observation] = []
        while True:
            line = input("> ")
            if line.strip().lower() == "stop":
                break
            if line.strip().lower() == "":
                print("Ignoring empty input...")
                continue
            result.append(
                Observation(description=line.strip())
            )
        self.logger.info(f"The human service agent collected the following initial observations: {str(result)}")
        return result

    async def execute_action(self, system: SystemDescription, action: DiagnosticAction, root_cause_description: Optional[RootCauseDescription]) -> DiagnosticActionResult:
        print(f"\nSuggested action: {action.get_name()}")
        if action.description:
            print(f"  -> {action.description}")
        else:
            print(f"  -> the action has no description")
            
        print("(Execute the action and describe the outcome)")

        outcome_text = input("Outcome> ").strip()

        dar = DiagnosticActionResult(action=action, outcome=outcome_text)
        self.logger.info(f"The human service agent executed an action and obtained an outcome: {dar}")
        return dar

    async def decide_finish(self, system: SystemDescription, state: AssistantState, root_cause_description: Optional[RootCauseDescription]) -> tuple[bool,Optional[RootCauseDescription]]:
        answer = input("\nDo you consider the diagnosis done now? [y/else]\n> ").strip().lower() 
        if answer != "y":
            return (False, None)
        
        answer = input("\nDo you want to record the putative root cause? [y/else]\n> ").strip().lower()
        if answer != "y":
            return (True, None)

        return (True, RootCauseDescription(
            root_cause_description_proper = input("Describe the root cause you believe has determined the system failure: ").strip(),
            symptoms_descriptions = [],
            notes = input("Optional notes: ").strip() or None,
        ))