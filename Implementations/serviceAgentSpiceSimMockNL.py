"""
ServiceAgentSpiceSimMockNL — ServiceAgentSpiceSim subclass using the mock
NL interface (zero LLM calls). Intended for protocol testing with
DiagnosticAssistantRandomTrajectory.

Only the nl_run calls in execute_action(), verify_hypothesis(), and
collect_initial_observations() are replaced. All circuit simulation logic
(fault snapshots, circuit reset, decide_finish) is inherited unchanged.
"""
from __future__ import annotations

import asyncio
from typing import Optional

from nl_interface.mock_interface import mock_run

from environment_classes import (
    DiagnosticAction, DiagnosticActionResult,
    DiagnosticFaultHypothesis, HypothesisVerificationResult,
    Observation, RootCauseDescription, SystemDescription,
)
from Implementations.serviceAgentSpiceSim import ServiceAgentSpiceSim, _circuit_state_summary


class ServiceAgentSpiceSimMockNL(ServiceAgentSpiceSim):
    """
    Identical to ServiceAgentSpiceSim but uses mock_run() instead of nl_run().
    No LLM calls are made; actions are parsed via keyword matching.
    """

    @property
    def description(self) -> str:
        return super().description + "_mockNL"

    async def collect_initial_observations(
        self,
        system: SystemDescription,
        root_cause_description: Optional[RootCauseDescription],
    ) -> list[Observation]:
        self.annoyance_level = 0
        self._repaired_comp_ids = set()
        if not system.simulated_system:
            raise ValueError("No simulated_system attached to SystemDescription.")
        system.simulated_system.add_logger(self.logger)

        narrative, cost, _, _ = await asyncio.to_thread(
            mock_run,
            self.INITIAL_OBSERVATIONS_PROMPT,
            system.simulated_system,
            "mock",
            "collect_information",
            self.logger,
        )
        self.logger.info(f"MockNL initial observation (sim_time={cost.time:.1f}s): {narrative}")
        return [Observation(description=narrative)]

    async def execute_action(
        self,
        system: SystemDescription,
        action: DiagnosticAction,
        root_cause_description: Optional[RootCauseDescription],
    ) -> DiagnosticActionResult:
        if not system.simulated_system:
            raise ValueError("No simulated_system attached to SystemDescription.")

        nl_prompt = action.description or action.get_name()
        narrative, sim_cost, actions, results = await asyncio.to_thread(
            mock_run,
            nl_prompt,
            system.simulated_system,
            "mock",
            "collect_information",
            self.logger,
            action.reporting_requirements,
        )

        self.logger.info(f"MockNL parsed actions: {actions}")
        self.logger.info(f"MockNL results: {[r for _, _, r in results]}")

        # Track replace_component repairs (same logic as parent)
        for entry, (_, _, res) in zip(actions, results):
            if entry.get("action_id") == "replace_component" and res.success:
                cid = entry.get("subject")
                if cid:
                    self._repaired_comp_ids.add(cid)

        # Post-batch circuit reset (same logic as parent)
        if self.RESET_CIRCUIT_AFTER_BATCH:
            _fs = getattr(system.simulated_system, "_fault_snapshot", None)
            if _fs is not None:
                system.simulated_system.restore_snapshot(
                    _fs, exclude_ids=self._repaired_comp_ids
                )

        fault_snapshot = getattr(system.simulated_system, "_fault_snapshot", None)
        state_summary = _circuit_state_summary(system.simulated_system, fault_snapshot)
        full_outcome = (
            f"{narrative}\nCurrent persistent circuit state [differences with the starting state]:\n"
            + (state_summary or "No difference.")
        )
        self.logger.info(
            f"MockNL executed '{action.get_name()}' | "
            f"sim_time={sim_cost.time:.1f}s | outcome: {full_outcome}"
        )
        breakdown = [(a.action_id, a.cost.time) for a, _, _ in results]
        return DiagnosticActionResult(
            action=action,
            outcome=full_outcome,
            precise_action_cost=sim_cost.time,
            cost_breakdown=breakdown,
            raw_results=results,
        )

    async def verify_hypothesis(
        self,
        system: SystemDescription,
        hypothesis: DiagnosticFaultHypothesis,
        root_cause_description: Optional[RootCauseDescription],
    ) -> HypothesisVerificationResult:
        if not system.simulated_system:
            raise ValueError("No simulated_system attached to SystemDescription.")

        sim = system.simulated_system
        fault_snapshot = getattr(sim, "_fault_snapshot", None)

        components_str = ", ".join(hypothesis.suspected_components)
        verify_map_prompt = (
            f"The technician suspects a fault in: {components_str}. "
            f"Identify the best matching component and call verify_repair on it."
        )
        _, sim_cost, entries, verify_results = await asyncio.to_thread(
            mock_run,
            verify_map_prompt,
            sim,
            "mock",
            "verify",
            self.logger,
        )
        verify_breakdown = [(a.action_id, a.cost.time) for a, _, _ in verify_results]
        candidate_ids: set[str] = {
            entry["subject"] for entry in entries
            if entry.get("action_id") == "verify_repair" and entry.get("subject")
        }

        if not candidate_ids:
            return HypothesisVerificationResult(
                hypothesis=hypothesis,
                outcome="wrong",
                narrative=(
                    f"The suspected component(s) {list(hypothesis.suspected_components)} "
                    f"could not be identified in the system."
                ),
                cost=sim_cost.time,
                cost_breakdown=verify_breakdown or None,
            )

        # Delegate repair check to parent's test_repair logic
        from diagnosable_systems_simulation.world.affordances import Affordance
        from diagnosable_systems_simulation.world.components import Cable as _Cable

        def _is_wrong_node(comp) -> bool:
            if not isinstance(comp, _Cable):
                return False
            orig = getattr(comp, "_orig_connections", {})
            return any(
                p.is_connected()
                and orig.get(p.name) is not None
                and p.node_id != orig[p.name]
                for p in comp.ports
            )

        still_broken_ids = {
            cid for cid, c in sim.all_components().items()
            if (Affordance.RECONNECTABLE in c.affordances.all_active(c, sim.context))
            or c.has_fault()
            or _is_wrong_node(c)
        }
        actually_faulty = candidate_ids & still_broken_ids

        lamp_on = await asyncio.to_thread(
            sim.test_repair, candidate_ids,
            already_repaired_ids=self._repaired_comp_ids,
        )

        if lamp_on:
            outcome = "correct"
        elif actually_faulty:
            outcome = "partial"
        else:
            outcome = "wrong"

        if outcome in ("correct", "partial"):
            self._repaired_comp_ids.update(actually_faulty)
            if lamp_on:
                from diagnosable_systems_simulation.world.components import PhysicalEnclosure as _PE
                self._repaired_comp_ids.update(
                    cid for cid in candidate_ids
                    if isinstance(sim.all_components().get(cid), _PE)
                )
            if fault_snapshot is not None:
                sim.apply_repairs(self._repaired_comp_ids)
                sim.restore_snapshot(fault_snapshot, exclude_ids=self._repaired_comp_ids)

        candidate_names = [
            sim.component(cid).display_name
            for cid in candidate_ids if cid in still_broken_ids
        ] or list(hypothesis.suspected_components)

        if outcome == "correct":
            narrative = f"Repaired {candidate_names}. System fully restored."
        elif outcome == "partial":
            narrative = f"Confirmed faulty: {candidate_names}. System still not working."
        else:
            narrative = f"Repair attempt on {list(hypothesis.suspected_components)} had no effect."

        self.logger.info(
            f"MockNL hypothesis verification | suspected={hypothesis.suspected_components} | "
            f"outcome='{outcome}' | lamp_on={lamp_on}"
        )
        return HypothesisVerificationResult(
            hypothesis=hypothesis,
            outcome=outcome,
            narrative=narrative,
            cost=sim_cost.time,
            cost_breakdown=verify_breakdown or None,
        )
