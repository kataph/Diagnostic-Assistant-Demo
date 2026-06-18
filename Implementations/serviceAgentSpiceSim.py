import asyncio
from typing import Optional

from nl_interface import run as nl_run

from configuration import Configuration
from environment_classes import (
    AssistantState, DiagnosticAction, DiagnosticActionResult,
    DiagnosticFaultHypothesis, HypothesisVerificationResult,
    Observation, RootCauseDescription, ServiceAgent, SystemDescription,
)


_RECONNECT_COST = 10.0
_REPLACE_COST = 120.0


def _estimate_repair_cost(sim, component_ids: "set[str]") -> float:
    """
    Estimate the cost a technician would pay to attempt repairing the given
    components, without mutating any circuit state.

    Mirrors the logic of DiagnosableSystem.apply_repairs():
    - Cables: 10 per port that needs reconnection (floating or wrong-net).
    - All other components: 120 (standard replacement cost).

    Used to charge wrong hypotheses the same cost they would have incurred
    had the attempted repair been on genuinely faulty components.
    """
    from diagnosable_systems_simulation.world.components import Cable
    total = 0.0
    for cid in component_ids:
        try:
            comp = sim.component(cid)
        except KeyError:
            continue
        if isinstance(comp, Cable):
            orig = getattr(comp, "_orig_connections", {})
            port_cost = 0.0
            for port_name, node_id in orig.items():
                port = comp.port(port_name)
                if not port.is_connected() or port.node_id != node_id:
                    port_cost += _RECONNECT_COST
            if getattr(comp, "_detached_cable_ports", {}):
                port_cost += _RECONNECT_COST * len(comp._detached_cable_ports)
            # Minimum attempt cost: even a healthy cable costs one reconnect
            # check when a technician handles it as part of a wrong hypothesis.
            total += max(port_cost, _RECONNECT_COST)
        else:
            total += _REPLACE_COST
    return total


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

        # Disconnected ports — only report those deliberately disconnected by the
        # agent (RECONNECTABLE affordance present).  Loose-coupling faults randomly
        # flip a port's connected state on every simulate(); showing them here would
        # produce spurious noise that is not caused by any agent action.
        from diagnosable_systems_simulation.world.affordances import Affordance
        if Affordance.RECONNECTABLE in comp.affordances._dynamic:
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
                            f"{comp.display_name}: port(s) {new_floating} disconnected by agent"
                        )
                else:
                    lines.append(
                        f"{comp.display_name}: port(s) {floating} disconnected by agent"
                    )

        # Fault overlays — only if they changed since fault snapshot
        if comp.has_fault():
            if fault_snapshot is not None:
                if dict(comp._fault_overlay) != snap_ovls.get(cid, {}):
                    lines.append(f"{comp.display_name}: degraded — {comp._fault_overlay}")
            else:
                lines.append(f"{comp.display_name}: degraded — {comp._fault_overlay}")

    if not lines:
        return None
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
        """
        Observe the externally visible state of the (already-faulted) simulated
        system and return it as a single Observation.

        NOTE: the fault must already be applied to system.simulated_system before
        this is called.  SaboteurSpiceSim is responsible for that.
        """
        # Reset per-scenario mutable state so the agent is safe to reuse.
        self.annoyance_level = 0
        self._repaired_comp_ids = set()
        if not system.simulated_system:
            raise ValueError("Got in input a system description without a simulation!")
        system.simulated_system.add_logger(self.logger)
        self.logger.debug(f"Added ServiceAgentSpiceSim own logger to the simulated_system. Spice circuits just before simulation should now appear in the logs")
        narrative, cost, _, _ = await asyncio.to_thread(
            nl_run, self.INITIAL_OBSERVATIONS_PROMPT, system.simulated_system,
            self.configuration.SERVICE_CONFIG.get("model", self.configuration.DEFAULT_SERVICE_MODEL), "collect_information", self.logger
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
            self.configuration.SERVICE_CONFIG.get("model", self.configuration.DEFAULT_SERVICE_MODEL), "collect_information", self.logger,
            action.reporting_requirements,
        )

        self.logger.info(f"action parsed from text input: {actions}")
        self.logger.info(f"results from the simulation: {[result for action, target, result in results]}")

        # Track components repaired by replace_component or reconnect_cable actions
        # so that verify_hypothesis and the post-batch reset can exclude them (they were
        # genuinely fixed mid-session and must not be re-faulted).
        for entry, (_, _, res) in zip(actions, results):
            action_id = entry.get("action_id")
            if action_id in ("replace_component", "reconnect_cable") and res.success:
                cid = entry.get("subject")
                if cid:
                    self._repaired_comp_ids.add(cid)
                    self.logger.info(
                        f"execute_action: tracked '{cid}' in _repaired_comp_ids "
                        f"after successful {action_id}"
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
        full_outcome = narrative
        if state_summary:
            full_outcome += f"\nCurrent persistent circuit state [differences with the starting state]:\n{state_summary}"
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
            self.configuration.SERVICE_CONFIG.get("model", self.configuration.DEFAULT_SERVICE_MODEL), "verify", self.logger
        )
        verify_breakdown = [(a.action_id, a.cost.time) for a, _, _ in verify_results]
        _removed = getattr(sim, "_removed_components", {})
        _all_subjects = {
            entry["subject"] for entry in entries
            if entry.get("action_id") == "verify_repair" and entry.get("subject")
        }
        _all_subjects.discard("_no_component_match")
        candidate_ids: set[str] = _all_subjects - set(_removed)
        _missing_ids: set[str] = _all_subjects & set(_removed)  # removed from system

        # If any candidate is a PhysicalEnclosure that is NOT referenced by any
        # coupling as a shielding/sensor enclosure, it is a plain module
        # (power supply, control, load cube). Modules cannot be "repaired" — they
        # are logical aggregates.  Charge a default penalty
        # and ask the assistant to target a specific component inside.
        from diagnosable_systems_simulation.world.components import PhysicalEnclosure as _PE

        def _is_sensor_enclosure(enc_comp) -> bool:
            """True if this enclosure is referenced by a coupling (e.g. ALS shielding)."""
            for coupling in sim._runner.couplings:
                if enc_comp in getattr(coupling, "shielding_enclosures", []):
                    return True
            return False

        plain_module_candidates = {
            cid for cid in candidate_ids
            if isinstance(sim.all_components().get(cid), _PE)
            and not _is_sensor_enclosure(sim.all_components()[cid])
        }
        if plain_module_candidates:
            module_names = [sim.component(cid).display_name for cid in plain_module_candidates]
            self.logger.warning(
                f"verify_hypothesis: hypothesis targets plain module(s) {module_names} "
                f"instead of specific components — charging default cost and rejecting."
            )
            return HypothesisVerificationResult(
                hypothesis=hypothesis,
                outcome="wrong",
                narrative=(
                    f"The hypothesis targeted module(s) {module_names}, but modules are "
                    f"functional aggregates — they cannot be repaired directly. "
                    f"Please identify the specific component(s) inside the module that are faulty "
                    f"(e.g., a cable, a bulb, a diode, a relay) or manipulate the module enclosure."
                ),
                cost=_REPLACE_COST,
                cost_breakdown=verify_breakdown or None,
            )

        # If the NL agent could not map the hypothesis to any component,
        # the description was not clear enough or did not match any system component.
        if not candidate_ids:
            if _missing_ids:
                missing_names = [_removed[cid] for cid in _missing_ids]
                self.logger.warning(
                    f"verify_hypothesis: all candidates {_missing_ids} are removed components."
                )
                return HypothesisVerificationResult(
                    hypothesis=hypothesis,
                    outcome="wrong",
                    narrative=(
                        f"The suspected component(s) {missing_names} have been physically "
                        f"removed from the system and cannot be repaired. "
                        f"Please identify a component that is still present."
                    ),
                    cost=sim_cost.time,
                    cost_breakdown=verify_breakdown or None,
                )
            self.logger.warning(
                f"verify_hypothesis: NL mapping produced no candidates for "
                f"{hypothesis.suspected_components}; returning wrong verdict."
            )
            return HypothesisVerificationResult(
                hypothesis=hypothesis,
                outcome="wrong",
                narrative=(
                    f"The suspected component description '{', '.join(hypothesis.suspected_components)}' "
                    f"could not be mapped to actual system components. "
                    f"The NL interface was unable to identify which component(s) you meant."
                ),
                cost=sim_cost.time,
                cost_breakdown=verify_breakdown or None,
            )

        # --- Identify all broken components in the current fault state --------
        from diagnosable_systems_simulation.world.components import Cable as _Cable
        from diagnosable_systems_simulation.electrical_simulation.couplings import LooseConnectionCoupling as _LCC

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

        loose_ids = {c.component_id for c in sim._runner.couplings if isinstance(c, _LCC)}

        still_broken_ids = {
            cid for cid, c in sim.all_components().items()
            if (Affordance.RECONNECTABLE in c.affordances.all_active(c, sim.context))
            or c.has_fault()
            or _is_wrong_node(c)
            or cid in loose_ids
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

        # --- Compute repair cost ----------------------------------------------
        # Charge for diagnostic attempt on ALL candidates the NL identified,
        # regardless of which ones were actually faulty. This reflects the true
        # technician cost: inspecting/attempting repair on all suspected components.
        repair_cost_time: float = 0.0
        if outcome == "wrong":
            repair_cost_time = _estimate_repair_cost(sim, candidate_ids)
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
                sim.apply_repairs(self._repaired_comp_ids)
                # Cost is for the diagnostic attempt on all candidates, not just
                # the ones that needed repair.
                repair_cost_time = _estimate_repair_cost(sim, candidate_ids)
                # For repositioned enclosures: set is_rotated=True before the
                # snapshot restore so that restore_snapshot(exclude_ids=...) skips
                # them and leaves is_rotated=True permanently.
                from diagnosable_systems_simulation.world.components import PhysicalEnclosure as _PE2
                for cid in self._repaired_comp_ids:
                    comp = sim.all_components().get(cid)
                    if isinstance(comp, _PE2):
                        comp.is_rotated = True
                sim.restore_snapshot(fault_snapshot, exclude_ids=self._repaired_comp_ids)

        # --- Build narrative --------------------------------------------------
        candidate_names = [
            sim.component(cid).display_name
            for cid in candidate_ids if cid in still_broken_ids
        ] or list(hypothesis.suspected_components)

        missing_note = (
            f" Note: {[_removed[cid] for cid in _missing_ids]} could not be "
            f"repaired — those components have been physically removed from the system."
            if _missing_ids else ""
        )

        # Cost breakdown for transparency
        cost_msg = f" [Cost: {repair_cost_time:.0f}s — diagnostic attempt on {len(candidate_ids)} component(s)]"

        # For wrong hypothesis, show both user input and actual parsed component IDs
        if outcome == "correct":
            narrative = (
                f"Repaired {candidate_names}. All output devices are now lit: "
                f"system fully restored.{cost_msg}{missing_note}"
            )
        elif outcome == "partial":
            narrative = (
                f"Confirmed faulty: {candidate_names}. Repaired them, but the "
                f"system is still not working — further faults remain.{cost_msg}{missing_note}"
            )
        else:
            attempted_display = [
                sim.component(cid).display_name for cid in candidate_ids
            ]
            narrative = (
                f"Repair attempt on {attempted_display} (user input: "
                f"{list(hypothesis.suspected_components)}) had no effect. "
                f"The suspected components were not found to be faulty.{cost_msg}{missing_note}"
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
                # A non-converged result means the circuit is oscillating (e.g. a
                # shorted lamp tripping a current-sensing relay in a loop). Never
                # treat a non-converged simulation as a successful repair.
                has_loose = sim.context.extra.get("has_loose_connection", False)
                if result.converged and not has_loose and result.emitting_light >= nominal_lit:
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
