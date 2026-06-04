"""
TrajectoryLog — writes a structured JSON trajectory alongside the other logs.

One JSON file is produced per scenario run. It contains:
  - scenario_id, total_cost, length, end
  - hypotheses_count (wrong / partial / right)
  - actions: ordered list of every loop event (actions + hypotheses), each with
      intention, implementation (per-sub-action simulation detail), outcome
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from environment_classes import (
        DiagnosticActionResult,
        DiagnosticFaultHypothesis,
        HypothesisVerificationResult,
    )


def _serialize_targets(targets: dict) -> dict[str, str]:
    """Convert {role: Component} to {role: component_id_str}."""
    out = {}
    for role, comp in targets.items():
        cid = getattr(comp, "component_id", None) or getattr(comp, "id", None) or str(comp)
        out[role] = cid
    return out


def _serialize_properties(observation) -> list[dict]:
    if observation is None:
        return []
    props = getattr(observation, "properties", [])
    return [{"name": p.name, "value": p.value, "unit": p.unit} for p in props]


def _build_implementation(raw_results: list | None) -> list[dict]:
    """
    Convert list[tuple[Action, dict[str, Component], ActionResult]] to
    a list of serializable dicts.
    """
    if not raw_results:
        return []
    entries = []
    for action, targets, result in raw_results:
        entries.append({
            "action_id": action.action_id,
            "targets": _serialize_targets(targets),
            "cost_time": getattr(getattr(action, "cost", None), "time", None),
            "success": result.success,
            "message": result.message,
            "properties": _serialize_properties(getattr(result, "observation", None)),
        })
    return entries


class TrajectoryLog:
    """
    Accumulates trajectory data in memory and writes a single JSON file on close().
    """

    def __init__(self, path: str, scenario_id: int) -> None:
        self._path = path
        self._scenario_id = scenario_id
        self._actions: list[dict] = []
        self._hypotheses_count: dict[str, int] = {"wrong": 0, "partial": 0, "right": 0}

    def record_action(self, result: "DiagnosticActionResult") -> None:
        self._actions.append({
            "intention": result.action.get_name(),
            "implementation": _build_implementation(result.raw_results),
            "outcome": result.outcome,
        })

    def record_hypothesis(
        self,
        hypothesis: "DiagnosticFaultHypothesis",
        result: "HypothesisVerificationResult",
    ) -> None:
        key = "right" if result.outcome == "correct" else result.outcome
        self._hypotheses_count[key] += 1

        comps = ", ".join(sorted(hypothesis.suspected_components))
        self._actions.append({
            "intention": f"Hypothesis -> {{{comps}}}",
            "implementation": [],
            "outcome": f"{result.outcome}: {result.narrative}",
        })

    def close(self, end: str, total_cost: float, length: int) -> None:
        data = {
            "scenario_id": self._scenario_id,
            "total_cost": total_cost,
            "length": length,
            "end": end,
            "hypotheses_count": self._hypotheses_count,
            "actions": self._actions,
        }
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
