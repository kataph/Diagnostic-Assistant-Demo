import asyncio
from typing import Optional

from nl_interface import run as nl_run

from configuration import Configuration
from environment_classes import (
    AssistantState, DiagnosticAction, DiagnosticActionResult,
    DiagnosticFaultHypothesis, HypothesisVerificationResult,
    HYPOTHESIS_VERIFICATION_COST,
    Observation, RootCauseDescription, ServiceAgent, SystemDescription,
)


class ServiceAgentSpiceSim(ServiceAgent):
    """
    Service agent backed by an equation-based electrical simulation.

    The natural-language description of a DiagnosticAction is forwarded
    verbatim to nl_interface.run(), which parses it into concrete
    simulation-level actions, executes them on the live circuit model,
    and returns a verbalized outcome plus an ActionCost.

    Both the KG-level cost (used by the diagnostic assistant's heuristic)
    and the simulation-level cost (time, equipment, resources) are logged
    so that the two cost models can be compared and calibrated.

    Termination: the orchestrator loop terminates automatically when the
    assistant's verify_hypothesis call returns "correct" (all faults found,
    system restored) or when suggest_action() returns None (total exhaustion).
    A patience cap fires after MAX_NUMBER_OF_ROUNDS as a safety net.
    """

    INITIAL_OBSERVATIONS_PROMPT = (
        "Observe all externally visible components of the system "
        "and report their current state."
    )

    def __init__(self, configuration: Configuration):
        super().__init__(configuration)
        self.patience_level = configuration.MAX_NUMBER_OF_ROUNDS - 1
        self.annoyance_level = 0

    @property
    def description(self) -> str:
        return super().description + f"_system={self.configuration.SYSTEM_NAME}"

    # ------------------------------------------------------------------ #
    # ServiceAgent interface                                               #
    # ------------------------------------------------------------------ #

    async def collect_initial_observations(
        self,
        system: SystemDescription,
        root_cause_description: Optional[RootCauseDescription],
    ) -> list[Observation]:
        """
        Observe the externally visible state of the (already-faulted) simulated
        system and return it as a single Observation.

        NOTE: the fault must already be applied to system.simulated_system before
        this is called.  SaboteurSpiceSim is responsible for that.
        """
        if not system.simulated_system:
            raise ValueError("Got in input a system description without a simulation!")
        narrative, cost = await asyncio.to_thread(
            nl_run, self.INITIAL_OBSERVATIONS_PROMPT, system.simulated_system
        )
        self.logger.info(
            f"Initial simulation observation "
            f"(sim_time={cost.time:.1f}s, equipment={cost.equipment}): {narrative}"
        )
        return [Observation(description=narrative)]

    async def execute_action(
        self,
        system: SystemDescription,
        action: DiagnosticAction,
        root_cause_description: Optional[RootCauseDescription],
    ) -> DiagnosticActionResult:
        """
        Forward the action's natural-language description to nl_interface.run()
        and wrap the verbalized result in a DiagnosticActionResult.
        """
        if not system.simulated_system:
            raise ValueError("Got in input a system description without a simulation!")
        nl_prompt = action.description or action.get_name()
        narrative, sim_cost = await asyncio.to_thread(nl_run, nl_prompt, system.simulated_system)

        self.logger.info(
            f"Executed '{action.get_name()}' via simulation | "
            f"kg_cost={action.get_cost()} | "
            f"sim_time={sim_cost.time:.1f}s | "
            f"equipment={sim_cost.equipment} | "
            f"resources={sim_cost.resources_consumed} | "
            f"outcome: {narrative}"
        )
        return DiagnosticActionResult(action=action, outcome=narrative, precise_action_cost=sim_cost.time)

    async def verify_hypothesis(
        self,
        system: SystemDescription,
        hypothesis: DiagnosticFaultHypothesis,
        root_cause_description: Optional[RootCauseDescription],
    ) -> HypothesisVerificationResult:
        """
        Ask nl_interface to attempt to repair/replace the suspected components
        and report whether the system is restored.

        The prompt instructs the simulation to output exactly one of the
        keywords 'correct', 'partial', or 'wrong' so that outcome parsing
        is unambiguous.
        """
        if not system.simulated_system:
            raise ValueError("Got in input a system description without a simulation!")

        components_str = ", ".join(hypothesis.suspected_components)
        verify_prompt = (
            f"Attempt to repair or replace the following suspected faulty components: "
            f"{components_str}. "
            "After the repair attempt, observe the system state carefully. "
            "If the system function is fully restored, write exactly 'correct'. "
            "If at least one of the named components was indeed faulty and has been "
            "replaced/repaired but the system is still not working, write exactly 'partial'. "
            "If the repair attempt had no effect and the system is still broken, "
            "write exactly 'wrong'."
        )
        narrative, sim_cost = await asyncio.to_thread(
            nl_run, verify_prompt, system.simulated_system
        )

        narrative_lower = narrative.lower()
        if "correct" in narrative_lower:
            outcome = "correct"
        elif "partial" in narrative_lower:
            outcome = "partial"
        else:
            outcome = "wrong"

        self.logger.info(
            f"Hypothesis verification via simulation | "
            f"suspected={hypothesis.suspected_components} | "
            f"outcome='{outcome}' | "
            f"sim_time={sim_cost.time:.1f}s | "
            f"narrative: {narrative}"
        )
        return HypothesisVerificationResult(
            hypothesis=hypothesis,
            outcome=outcome,
            narrative=narrative,
            cost=HYPOTHESIS_VERIFICATION_COST,
        )

    async def decide_finish(
        self,
        system: SystemDescription,
        state: AssistantState,
        root_cause_description: Optional[RootCauseDescription],
    ) -> tuple[bool, Optional[RootCauseDescription]]:
        """
        Patience-based safety cap.  Natural termination is driven by the
        assistant emitting a DiagnosticFaultHypothesis that verifies as
        "correct", which the orchestrator handles before calling decide_finish.
        """
        if self.annoyance_level >= self.patience_level:
            self.logger.info(
                "ServiceAgentSpiceSim: patience exhausted, ending session."
            )
            return (True, None)
        self.annoyance_level += 1
        return (False, None)
