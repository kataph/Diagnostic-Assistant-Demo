from typing import Optional

from environment_classes import Saboteur, SymptomDescription, SystemDescription, SymptomDescriptions, RootCauseDescription


SCENARIOS = [
    (0,
     "3_cubes",
     RootCauseDescription(
         root_cause_description_proper="One of the cables connected to the switch has been detached",
         symptoms_descriptions=SymptomDescriptions([
             SymptomDescription(
                 "The lamp does not turn on when the switch is operated"),
             SymptomDescription(
                 "The green led on top of the power supply module is on"),
             SymptomDescription(
                 "The red led on top of the control module is off"),
         ])
     )),
    (1, "3_cubes",
     RootCauseDescription(
        root_cause_description_proper="Burned lamp filaments",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription(
                "The lamp does not turn on when the switch is operated"),
            SymptomDescription(
                "The green led on top of the power supply module is on"),
            SymptomDescription(
                "The red led on top of the control module is off"),
        ])
     )),
    (2, "3_cubes",
     RootCauseDescription(
        root_cause_description_proper="Battery is depleted",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription(
                "The lamp does not turn on when the switch is operated"),
            SymptomDescription(
                "The green led on top of the power supply module is off"),
            SymptomDescription(
                "The red led on top of the control module is off"),
        ])
     )),
    (3, "3_cubes",
     RootCauseDescription(
        root_cause_description_proper="Battery has been installed incorrectly (it has been installed with inverted polarity)",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription(
                "The lamp does not turn on when the switch is operated"),
            SymptomDescription(
                "The green led on top of the power supply module is off"),
            SymptomDescription(
                "The red led on top of the control module is on"),
        ])
     )),
    (4, "3_cubes",
     RootCauseDescription(
        root_cause_description_proper="The cables between the control module and the load module (or between the power supply module and the control module) are crossed, resulting in reverse voltage being supplied to the load",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription(
                "The lamp does not turn on when the switch is operated"),
            SymptomDescription(
                "The green led on top of the power supply module is on"),
            SymptomDescription(
                "The red led on top of the control module is on"),
        ])
     )),
    (5, "3_cubes",
     RootCauseDescription(
        root_cause_description_proper="Internal open circuit in the switch: the switch is always open.",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription(
                "The lamp does not turn on when the switch is operated"),
            SymptomDescription(
                "The green led on top of the power supply module is on"),
            SymptomDescription(
                "The red led on top of the control module is off"),
        ])
     )),
    (6, "10_cubes",
     RootCauseDescription(
        root_cause_description_proper="Battery exhausted.",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription("The led on the power supply module is off"),
            SymptomDescription("All the leds on the control modules are off"),
            SymptomDescription(
                "The lamp does not turn on when the battery is inserted in the circuit and all the switches are in the on position"),
        ])
     )),
    (7, "10_cubes",
     RootCauseDescription(
        root_cause_description_proper="The switch in the control module 3 is detached from one of the corresponding cables.",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription("The led on the power supply module is on"),
            SymptomDescription(
                "The first two control module leds are on. Starting from the third onwards they are off"),
            SymptomDescription(
                "The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
        ])
     )),
    (8, "10_cubes",
     RootCauseDescription(
        root_cause_description_proper="The switch in the control module 3 is detached from one of the corresponding cables. Also, all the control module leds have been removed.",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription("The led on the power supply module is on"),
            SymptomDescription(
                "All the leds on the control modules are missing"),
            SymptomDescription(
                "The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
        ])
     )),
    (9, "10_cubes",
     RootCauseDescription(
        root_cause_description_proper="The switch in the control module 6 is detached from one of the corresponding cables. Also, all the control module leds have been removed.",
        symptoms_descriptions=SymptomDescriptions([
            SymptomDescription("The led on the power supply module is on"),
            SymptomDescription(
                "All the leds on the control modules are missing"),
            SymptomDescription(
                "The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
        ])
     )),
    (10, "10_cubes",
     RootCauseDescription(
         root_cause_description_proper="The switch in the control module 6 is detached from one of the corresponding cables. Also, all the control module leds have been removed. Also, the service agent does not have a multimiter or other tools at its disposal to take electric measurements.",
         symptoms_descriptions=SymptomDescriptions([
             SymptomDescription("The led on the power supply module is on"),
             SymptomDescription(
                 "All the leds on the control modules are missing"),
             SymptomDescription(
                 "The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
         ])
     )),
    (11, "3_cubes",
     RootCauseDescription(
         root_cause_description_proper="Detached cable from the switch and, at the same time and independently, The cables between the control module and the load module (or between the power supply module and the control module) are crossed, resulting in reverse voltage being supplied to the load",
         symptoms_descriptions=SymptomDescriptions([
             SymptomDescription(
                 "The lamp does not turn on when the switch is operated"),
             SymptomDescription(
                 "The green led on top of the power supply module is on"),
             SymptomDescription(
                 "The red led on top of the control module is on"),
         ])
     )),
    (12, "10_cubes",
     RootCauseDescription(
         root_cause_description_proper="The switch in the control module 3 is detached from one of the corresponding cables. Also, at the same time and independently, battery is exhausted.",
         symptoms_descriptions=SymptomDescriptions([
             SymptomDescription("The led on the power supply module is off"),
             SymptomDescription("All the leds on the control modules are off"),
             SymptomDescription(
                 "The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
         ])
     )),
    (13, "10_cubes",
     RootCauseDescription(
         root_cause_description_proper="The cables that come out from the power supply module are accidently shorted. This discharged the battery violently and now the battery does not supply power anymore. Replacing the battery will not solve the issue.",
         symptoms_descriptions=SymptomDescriptions([
             SymptomDescription("The led on the power supply module is off"),
             SymptomDescription("All the leds on the control modules are off"),
             SymptomDescription(
                 "The lamp does not turn on when a battery is inserted in the circuit and all the switches are in the on position"),
         ])
     )),
    (14, "ambient_light_sensor",
     RootCauseDescription(
         root_cause_description_proper="The lamp turns off about every 20 seconds because the sensor is incorrectly positioned and receives part of the light of the lamp: a sufficient quantity to make the sensor turn off the lamp. This happens about every 20 seconds due to the ambient light sensor inner workings. After the light turns off the sensor records a below-threshold ambient light and turns on the lamp almost immediately. This keeps occurring.",
         symptoms_descriptions=SymptomDescriptions([
             SymptomDescription("The led on the power supply module is on"),
             SymptomDescription(
                 "The lamp turns off about every 20 seconds for about half a second. Then it turns on again. This keeps happening."),
         ])
     )),
]


class SaboteurFixedScenario(Saboteur):

    @property
    def description(self):
        return super().description + "_" + f"scenario_id={self.configuration.FORCED_SCENARIO_ID}"

    async def sabotage(self, description: SystemDescription) -> Optional[RootCauseDescription]:
        forced_scenarios = [scenario for scenario in SCENARIOS if scenario[0] == self.configuration.FORCED_SCENARIO_ID]
        if len(forced_scenarios) != 1:
            raise ValueError(f"Scenarios with id {self.configuration.FORCED_SCENARIO_ID} are {len(forced_scenarios)}. It should be exactly 1!")
        _, system, rc = forced_scenarios[0]
        self.logger.info(
            f"The system was saboted producing the following root cause: {rc.one_liner_repr()}")
        return rc
