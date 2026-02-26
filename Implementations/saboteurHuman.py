from typing import Optional

from Utilities.caching import async_disk_cache_CLI
from environment_classes import Saboteur, SymptomDescription, SystemDescription, SymptomDescriptions, RootCauseDescription


class SaboteurHuman(Saboteur):

    # @async_disk_cache_CLI #TODO: comment when no need for fast testing
    async def sabotage(self, description: SystemDescription) -> Optional[RootCauseDescription]:
        prompt_for_human_start = (f"Your job is to sabotage the physical system (hopefully it is next to you), in any way you see fit.\n"
        + f"If it is of any help, the following is a textual description of the system:\n     {description.text_input}\n"
        + (f"Consider also the accompanying file (id: {description.file_id})\n" if description.file_id else "\n")
        + "When you are finished, please, press enter.\n")
        input(prompt_for_human_start)
        y_n = ""
        y_n = input("Do you want to record a description of the root cause (for logging purposes)?  y/n\n>").lower()
        while y_n not in ["y","n"]:
            y_n = input("You have to write either 'y' or 'n', try again\n>").strip().lower()
        if y_n == "n":
            self.logger.info(f"The system was saboted producing an unrecorded root cause")
            return None
        print("You will be asked to describe the failure mode at the level of its root cause, and then at the level of its immediately observable symptoms. ")
        prompt_for_human_collect_rc_description = f"""Please, describe briefly what the is the failure mode (root cause) you have sabotaged the system with: """
        prompt_for_human_collect_symptoms_description = f"""Please, describe briefly some imediately observable aspects of the system behavior (you will be able to list multiple of these, when you want to finish write 'stop'): """
        rc_description = input(prompt_for_human_collect_rc_description)
        symptom_descriptions = SymptomDescriptions([])
        while True:
            symptom_description = input(prompt_for_human_collect_symptoms_description)
            match symptom_description:
                case "":
                    print("Not gonna record empty descriptions")
                    continue
                case "stop":
                    print("Breaking loop...")
                    break
                case _:
                    print("Recording description...")
                    symptom_descriptions.append(SymptomDescription(symptom_description))
        rc = RootCauseDescription(root_cause_description_proper = rc_description, symptoms_descriptions = symptom_descriptions)
        self.logger.info(f"The system was saboted producing the following root cause: {rc.one_liner_repr()}")
        return rc
    
    
    
    
