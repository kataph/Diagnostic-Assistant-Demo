import asyncio
from typing import Optional

from nl_interface import run as nl_run

from configuration import Configuration
from environment_classes import (
    AssistantState, DiagnosticAction, DiagnosticActionResult,
    DiagnosticFaultHypothesis, HypothesisVerificationResult,
    Observation, RootCauseDescription, ServiceAgent, SystemDescription,
)


def _circuit_state_summary(sim, fault_snapshot: "dict | None" = None) -> str:
    """
    Return a compact, system-agnostic snapshot of circuit-state changes
    introduced by the diagnostic agent relative to the post-fault-injection
    baseline (fault_snapshot).

    When fault_snapshot is provided only deviations from that baseline are
    reported — pre-existing faults injected by the saboteur are silently
    omitted.  When fault_snapshot is None the function falls back to showing
    all non-nominal state (original behaviour, used only before the snapshot
    is available).
    """
    snap_ports  = (fault_snapshot or {}).get("port_connections", {})
    snap_states = (fault_snapshot or {}).get("component_states", {})
    snap_ovls   = (fault_snapshot or {}).get("fault_overlays", {})

    lines = []
    for cid, comp in sim.all_components().items():
        params  = comp.current_parameters()
        nominal = comp.nominal_parameters()

        # Switch positions — only if they changed since fault snapshot
        if "is_closed" in nominal:
            current_closed = params.get("is_closed", nominal["is_closed"])
            if fault_snapshot is not None:
                snap_closed = snap_states.get(cid, {}).get("is_closed", current_closed)
                if current_closed == snap_closed:
                    pass  # no change since snapshot — skip
                else:
                    state = "CLOSED" if current_closed else "OPEN"
                    lines.append(f"{comp.display_name}: {state}")
            else:
                state = "CLOSED" if current_closed else "OPEN"
                lines.append(f"{comp.display_name}: {state}")

        # Disconnected ports — only if they were connected in the fault snapshot
        floating = [p.name for p in comp.ports if not p.is_connected()]
        if floating:
            if fault_snapshot is not None:
                snap_comp_ports = snap_ports.get(cid, {})
                new_floating = [
                    pname for pname in floating
                    if snap_comp_ports.get(pname) is not None
                ]
                if new_floating:
                    lines.append(
                        f"{comp.display_name}: port(s) {new_floating} disconnected (floating)"
                    )
            else:
                lines.append(
                    f"{comp.display_name}: port(s) {floating} disconnected (floating)"
                )

        # Fault overlays — only if they changed since fault snapshot
        if comp.has_fault():
            if fault_snapshot is not None:
                if dict(comp._fault_overlay) != snap_ovls.get(cid, {}):
                    lines.append(f"{comp.display_name}: degraded — {comp._fault_overlay}")
            else:
                lines.append(f"{comp.display_name}: degraded — {comp._fault_overlay}")

    if not lines:
        return "No circuit-level changes introduced by diagnostics."
    return "\n".join(f"  • {l}" for l in lines)


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

    # Set to False to disable automatic circuit reset after each action batch.
    # When True, the full circuit state (switches, cables, enclosure orientations,
    # open panels) is restored to the fault-snapshot baseline after every batch so
    # that subsequent observe_component calls always read last_result for the correct
    # operating conditions rather than a transient test state.
    # Components in _repaired_comp_ids are excluded and remain in their repaired state.
    RESET_CIRCUIT_AFTER_BATCH: bool = True

    def __init__(self, configuration: Configuration):
        super().__init__(configuration)
        self.patience_level = configuration.MAX_NUMBER_OF_ROUNDS - 1
        self.annoyance_level = 0
        # Component IDs that have already been repaired in a previous "partial"
        # verification and must not be reset when restoring the fault snapshot.
        self._repaired_comp_ids: set[str] = set()

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
        # Reset per-scenario mutable state so the agent is safe to reuse.
        self.annoyance_level = 0
        self._repaired_comp_ids = set()
        """
        Observe the externally visible state of the (already-faulted) simulated
        system and return it as a single Observation.

        NOTE: the fault must already be applied to system.simulated_system before
        this is called.  SaboteurSpiceSim is responsible for that.
        """
        if not system.simulated_system:
            raise ValueError("Got in input a system description without a simulation!")
        system.simulated_system.add_logger(self.logger)
        self.logger.debug(f"Added ServiceAgentSpiceSim own logger to the simulated_system. Spice circuits just before simulation should now appear in the logs")
        narrative, cost, _, _ = await asyncio.to_thread(
            nl_run, self.INITIAL_OBSERVATIONS_PROMPT, system.simulated_system,
            self.configuration.SERVICE_CONFIG.get("model", self.configuration.SERVICE_MODEL), "collect_information", self.logger
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
        narrative, sim_cost, actions, results = await asyncio.to_thread(
            nl_run, nl_prompt, system.simulated_system,
            self.configuration.SERVICE_CONFIG.get("model", self.configuration.SERVICE_MODEL), "collect_information", self.logger,
            action.reporting_requirements,
        )

        self.logger.info(f"action parsed from text input: {actions}")
        self.logger.info(f"results from the simulation: {[result for action, target, result in results]}")

        # Track components repaired by replace_component diagnostic actions so that
        # verify_hypothesis can exclude them from fault-snapshot restores (they were
        # genuinely fixed mid-session and must not be re-faulted during test_repair).
        for entry, (_, _, res) in zip(actions, results):
            if entry.get("action_id") == "replace_component" and res.success:
                cid = entry.get("subject")
                if cid:
                    self._repaired_comp_ids.add(cid)
                    self.logger.info(
                        f"execute_action: tracked '{cid}' in _repaired_comp_ids "
                        f"after successful replace_component"
                    )

        # ── Post-batch circuit reset ──────────────────────────────────────────
        # Restore the full circuit state (switches, cables, enclosure orientations,
        # open panels) to the fault-snapshot baseline so last_result always reflects
        # normal fault-state operating conditions rather than a transient test state.
        # Components confirmed repaired (in _repaired_comp_ids) are excluded and
        # remain in their repaired state.  Flip RESET_CIRCUIT_AFTER_BATCH = False
        # to disable.
        if self.RESET_CIRCUIT_AFTER_BATCH:
            _fs = getattr(system.simulated_system, "_fault_snapshot", None)
            if _fs is not None:
                system.simulated_system.restore_snapshot(
                    _fs, exclude_ids=self._repaired_comp_ids
                )
                self.logger.info("[post-batch reset] Circuit restored to fault-snapshot baseline.")

        fault_snapshot = getattr(system.simulated_system, "_fault_snapshot", None)
        state_summary = _circuit_state_summary(system.simulated_system, fault_snapshot)
        full_outcome = (f"{narrative}\nCurrent persistent circuit state [differences with the starting state]:\n"
                        + (f"{state_summary}" if state_summary else "No difference. "))
        self.logger.info(
            f"Executed '{action.get_name()}' via simulation | "
            f"kg_cost={action.get_cost()} | "
            f"sim_time={sim_cost.time:.1f}s | "
            f"equipment={sim_cost.equipment} | "
            f"resources={sim_cost.resources_consumed} | "
            f"outcome: {full_outcome}"
        )
        breakdown = [(a.action_id, a.cost.time) for a, _, _ in results]
        return DiagnosticActionResult(
            action=action, outcome=full_outcome,
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
        """
        Verify a hypothesis by directly repairing the suspected components in
        the simulation and checking whether the output devices light up.

        No NL call is made: the outcome is determined entirely from simulation
        state, independent of any narrative feedback.
        """
        if not system.simulated_system:
            raise ValueError("Got in input a system description without a simulation!")

        from diagnosable_systems_simulation.world.affordances import Affordance

        sim = system.simulated_system
        fault_snapshot = getattr(sim, "_fault_snapshot", None)

        # --- Resolve hypothesis text → component IDs via the NL interface ---
        # The NL agent maps the free-text suspected-component description to a
        # concrete component ID by selecting the "verify_repair" action.
        # This sidesteps any name-mismatch between what the diagnostic assistant
        # writes and the actual component display names in the simulation.
        components_str = ", ".join(hypothesis.suspected_components)
        verify_map_prompt = (
            f"The technician suspects a fault in the following component(s): "
            f"{components_str}. "
            f"Identify the best matching component(s) in the system and call "
            f"verify_repair on each one."
        )
        _, sim_cost, entries, verify_results = await asyncio.to_thread(
            nl_run, verify_map_prompt, sim,
            self.configuration.SERVICE_CONFIG.get("model", self.configuration.SERVICE_MODEL), "verify", self.logger
        )
        verify_breakdown = [(a.action_id, a.cost.time) for a, _, _ in verify_results]
        candidate_ids: set[str] = {
            entry["subject"] for entry in entries
            if entry.get("action_id") == "verify_repair" and entry.get("subject")
        }

        # If the NL agent could not map the hypothesis to any component, the
        # suspected components do not exist in the system — verdict is WRONG.
        if not candidate_ids:
            self.logger.warning(
                f"verify_hypothesis: NL mapping produced no candidates for "
                f"{hypothesis.suspected_components}; returning wrong verdict."
            )
            return HypothesisVerificationResult(
                hypothesis=hypothesis,
                outcome="wrong",
                narrative=(
                    f"The suspected component(s) {list(hypothesis.suspected_components)} "
                    f"could not be identified in the system and were not repaired."
                ),
                cost=sim_cost.time,
                cost_breakdown=verify_breakdown or None,
            )

        # --- Identify all broken components in the current fault state --------
        from diagnosable_systems_simulation.world.components import Cable as _Cable

        def _is_wrong_node(comp) -> bool:
            """True for a cable with a port connected to the wrong net."""
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

        # --- Test: repair only the candidates ---------------------------------
        lamp_on = await asyncio.to_thread(
            sim.test_repair,
            candidate_ids,
            already_repaired_ids=self._repaired_comp_ids,
        )

        # --- Determine outcome -----------------------------------------------
        if lamp_on:
            outcome = "correct"
        elif actually_faulty:
            outcome = "partial"
        else:
            outcome = "wrong"

        # --- Persist confirmed repairs ----------------------------------------
        repair_cost_time: float = 0.0
        if outcome in ("correct", "partial"):
            self._repaired_comp_ids.update(actually_faulty)

            # PhysicalEnclosure components have no electrical fault overlay, so
            # actually_faulty will not include them even when their repositioning
            # fixed the system.  Add them explicitly when lamp_on is True.
            if lamp_on:
                from diagnosable_systems_simulation.world.components import PhysicalEnclosure as _PE
                enclosure_candidates = {
                    cid for cid in candidate_ids
                    if isinstance(sim.all_components().get(cid), _PE)
                }
                self._repaired_comp_ids.update(enclosure_candidates)

            if fault_snapshot is not None:
                # test_repair() always restores the circuit to fault state on
                # exit, so confirmed components are currently faulted again.
                # Apply the repairs now so that restore_snapshot(exclude_ids)
                # leaves those components in the repaired state.
                repair_cost_time = sim.apply_repairs(self._repaired_comp_ids).time
                # For repositioned enclosures: set is_inverted=True before the
                # snapshot restore so that restore_snapshot(exclude_ids=...) skips
                # them and leaves is_inverted=True permanently.
                from diagnosable_systems_simulation.world.components import PhysicalEnclosure as _PE2
                for cid in self._repaired_comp_ids:
                    comp = sim.all_components().get(cid)
                    if isinstance(comp, _PE2):
                        comp.is_inverted = True
                sim.restore_snapshot(fault_snapshot, exclude_ids=self._repaired_comp_ids)

        # --- Build narrative --------------------------------------------------
        candidate_names = [
            sim.component(cid).display_name
            for cid in candidate_ids if cid in still_broken_ids
        ] or list(hypothesis.suspected_components)
        if outcome == "correct":
            narrative = (
                f"Repaired {candidate_names}. All output devices are now lit: "
                f"system fully restored."
            )
        elif outcome == "partial":
            narrative = (
                f"Confirmed faulty: {candidate_names}. Repaired them, but the "
                f"system is still not working — further faults remain."
            )
        else:
            narrative = (
                f"Repair attempt on {list(hypothesis.suspected_components)} had no "
                f"effect. The suspected components were not found to be faulty."
            )

        self.logger.info(
            f"Hypothesis verification via simulation | "
            f"suspected={hypothesis.suspected_components} | "
            f"candidate_ids={candidate_ids} | "
            f"outcome='{outcome}' | "
            f"lamp_on={lamp_on} | "
            f"narrative: {narrative}"
        )
        return HypothesisVerificationResult(
            hypothesis=hypothesis,
            outcome=outcome,
            narrative=narrative,
            cost=repair_cost_time,
            cost_breakdown=verify_breakdown or None,
        )

    async def decide_finish(
        self,
        system: SystemDescription,
        state: AssistantState,
        root_cause_description: Optional[RootCauseDescription],
    ) -> tuple[bool, Optional[RootCauseDescription], Optional[str]]:
        """
        Termination check.  In addition to the patience-based safety cap,
        we check whether the system is already restored (lamp on).  This
        happens when the technician replaces the faulty component directly
        via a diagnostic action instead of going through verify_hypothesis.
        Such runs are counted as successes in statistical metrics.
        """
        sim = system.simulated_system
        if sim is not None:
            nominal_lit = getattr(sim, "_nominal_emitting_light", frozenset())
            if nominal_lit:
                result = await asyncio.to_thread(sim.simulate)
                # HACK: for the ambient-light-sensor system the coupling loop
                # may not converge (flickering fault), so a non-converged
                # result with the lamp appearing on is NOT a real fix.
                # Gate the success check on convergence for that system only.
                is_als = getattr(sim, "name", None) == "ambient_light_sensor"
                has_loose = sim.context.extra.get("has_loose_connection", False)
                if (not is_als or result.converged) and not has_loose and result.emitting_light >= nominal_lit:
                    self.logger.info(
                        "ServiceAgentSpiceSim: system_restored_via_action — "
                        "lamp is on, ending session as success."
                    )
                    return (True, None, "Session ended: the system was restored but the root cause was not identified.")

        if self.annoyance_level >= self.patience_level:
            self.logger.info(
                "ServiceAgentSpiceSim: patience exhausted, ending session."
            )
            return (True, None, "Session ended without identifying the root cause (patience cap reached).")
        self.annoyance_level += 1
        return (False, None, None)
