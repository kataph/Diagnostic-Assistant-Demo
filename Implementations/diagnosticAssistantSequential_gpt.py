# =========
# One concrete implementation: SequentialDiagnosticAssistant
# =========

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from environment_classes import AssistantState, DiagnosticAction, DiagnosticActionResult, DiagnosticAssistant, Observation, RootCauseDescription, RootCauseHypothesis

class PlanStatus(str, Enum):
    ONGOING = "ongoing"
    EXHAUSTED = "exhausted"
    FAILED = "failed"

class ActionVerb(str, Enum):
    OBSERVE = "observe"
    TEST = "test"
    ADJUST = "adjust"
    REPLACE = "replace"
    
class SequentialDiagnosticPlan(BaseModel):
    actions: list[DiagnosticAction]
    current_index: int = 0
    status: PlanStatus = PlanStatus.ONGOING
    hypotheses: list[RootCauseHypothesis] = Field(default_factory=list)


class DiagnosticAssistantSequential_gpt(DiagnosticAssistant):
    """
    A simple reference implementation, embedding the planning/evidence logic
    that previously lived in Planner + DiagnosticAssistant.
    """

    def __init__(self) -> None:
        self._state = AssistantState()

    @property
    def state(self) -> AssistantState:
        return self._state

    # ---- public API required by abstract base ----

    def setup(self, observations: list[Observation]) -> None:
        self._state = AssistantState()
        self._state.observations.extend(observations)
        # "Planning" is internal here
        self._state.plan = self._initial_plan(self._state.observations)

    def suggest_action(self) -> Optional[DiagnosticAction]:
        # Incorporate last outcome if provided
        last_outcome = self.state.diagnostic_scenario_memory[0] if len(self.state.diagnostic_scenario_memory) > 0 else None
        if last_outcome is not None:
            self._state.outcomes.append(last_outcome)
            if self._state.plan is not None:
                self._update_plan_after_outcome(last_outcome)

        plan = self._state.plan
        if not plan or plan.status != PlanStatus.ONGOING:
            return None

        if plan.current_index >= len(plan.actions):
            plan.status = PlanStatus.EXHAUSTED
            return None

        return plan.actions[plan.current_index]

    def finish_session(self, root_cause: Optional[RootCauseDescription]) -> None:
        self._state.user_finished = True
        if root_cause:
            self._state.user_confirmed_root_cause = RootCauseHypothesis(
                component=root_cause.component,
                failure_mode=root_cause.failure_mode,
                confidence=1.0,
                proposed_by="user",
            )

    # ---- internal planning logic specific to this concrete assistant ----

    def _initial_plan(self, observations: list[Observation]) -> SequentialDiagnosticPlan:
        actions = [
            DiagnosticAction(
                verb=ActionVerb.OBSERVE,
                component="power_supply",
                description="Check whether the power LED is ON",
                estimated_cost=1.0,
            ),
            DiagnosticAction(
                verb=ActionVerb.TEST,
                component="control_unit",
                description="Run control unit self-test",
                estimated_cost=2.0,
            ),
        ]
        hypotheses = [
            RootCauseHypothesis(
                component="power_supply",
                failure_mode="no_input_voltage",
                confidence=0.4,
                proposed_by="assistant",
            )
        ]
        return SequentialDiagnosticPlan(actions=actions, hypotheses=hypotheses)

    def _update_plan_after_outcome(self, outcome: DiagnosticActionResult) -> None:
        plan = self._state.plan
        if not plan or plan.status != PlanStatus.ONGOING:
            return

        # Advance index if expected action matches outcome.action
        if (
            plan.current_index < len(plan.actions)
            and plan.actions[plan.current_index] == outcome.action
        ):
            plan.current_index += 1

        # Example: no complex re-planning here, but you could adapt
        if plan.current_index >= len(plan.actions):
            plan.status = PlanStatus.EXHAUSTED

