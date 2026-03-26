import asyncio
from typing import Optional

from diagnosable_systems_simulation.systems.base_system import DiagnosableSystem
from nl_interface import run as nl_run

from configuration import Configuration
from environment_classes import (
    AssistantState, DiagnosticAction, DiagnosticActionResult,
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

    The agent terminates after configuration.MAX_NUMBER_OF_ROUNDS rounds
    (patience-based). It never returns a confirmed root cause because the
    simulation has no built-in oracle for that judgement; couple it with
    a diagnostic assistant that does its own termination reasoning, or
    replace decide_finish with an LLM-based judge if needed.
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
        If not self.LLM_elaborates_initial_observations and there are no symptoms_descriptions in the root cause, Observe the externally visible state of the (already-faulted) simulated
        system and return it as a single Observation. Otherwise the symptoms are returned as default symptoms.

        NOTE: the fault must already be applied to system.simulated_system before this is
        called.  A SaboteurSimulation companion class is responsible for that.
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

        The action description (set by the diagnostic assistant, possibly
        enriched by HeuristicTestingProcedure) is used as the NL prompt so
        that any context added by the planner is preserved.  Falls back to
        the action name if no description is set.
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

    async def decide_finish(
        self,
        system: SystemDescription,
        state: AssistantState,
        root_cause_description: Optional[RootCauseDescription],
    ) -> tuple[bool, Optional[RootCauseDescription]]:
        """
        Patience-based termination: ends the session after MAX_NUMBER_OF_ROUNDS.
        Returns no confirmed root cause (the simulation has no oracle for that).
        """
        if self.annoyance_level >= self.patience_level:
            self.logger.info(
                "ServiceAgentSpiceSim: patience exhausted, ending session."
            )
            return (True, None)
        self.annoyance_level += 1
        return (False, None)
