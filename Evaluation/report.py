"""
Report generation for the adaptive evaluation protocol.
Produces human-readable text summaries (no HTML, no PDF).
"""
from __future__ import annotations

from typing import Optional


def format_iteration_report(
    scenario_number: int,
    batch_index: int,
    n_trajectories: int,
    n_clusters_intent: int,
    n_clusters_execution: int,
    ari_inter_intent: Optional[float],
    ari_inter_execution: Optional[float],
    ari_boot_p05_intent: float,
    ari_boot_p05_execution: float,
    converged_intent: bool,
    converged_execution: bool,
    streak_intent: int,
    streak_execution: int,
    convergence_window: int,
    numerical_metrics: Optional[dict] = None,
    recommendation: str = "CONTINUE",
) -> str:
    lines = [
        f"{'='*60}",
        f"Scenario {scenario_number} | Batch {batch_index} | N={n_trajectories}",
        f"{'='*60}",
        f"  Intent level:    clusters={n_clusters_intent:<3}  "
        f"ARI_inter={_fmt(ari_inter_intent)}  noise_floor={ari_boot_p05_intent:.3f}  "
        f"pass={'✓' if converged_intent else '✗'}  streak={streak_intent}/{convergence_window}",
        f"  Execution level: clusters={n_clusters_execution:<3}  "
        f"ARI_inter={_fmt(ari_inter_execution)}  noise_floor={ari_boot_p05_execution:.3f}  "
        f"pass={'✓' if converged_execution else '✗'}  streak={streak_execution}/{convergence_window}",
    ]
    if numerical_metrics:
        sr = numerical_metrics.get("success_rate", {})
        cost = numerical_metrics.get("total_cost", {})
        lines.append(
            f"  Metrics:  success={_pct(sr)}  cost={_pt(cost)}  "
            f"n_actions={_pt(numerical_metrics.get('n_actions', {}))}"
        )
    lines.append(f"  → {recommendation}")
    return "\n".join(lines)


def format_final_report(
    scenario_number: int,
    total_batches: int,
    total_trajectories: int,
    n_clusters_intent: int,
    n_clusters_execution: int,
    ari_inter_intent: Optional[float],
    ari_inter_execution: Optional[float],
    ari_boot_p05_intent: float,
    ari_boot_p05_execution: float,
    cluster_labels_intent: Optional[dict[int, str]],
    numerical_metrics: dict,
    batch_history: list[dict],
    reason: str = "converged",
) -> str:
    lines = [
        f"{'#'*60}",
        f"FINAL REPORT — Scenario {scenario_number}",
        f"{'#'*60}",
        f"  Reason: {reason}",
        f"  Total batches: {total_batches} | Total trajectories: {total_trajectories}",
        f"  Intent clusters: {n_clusters_intent}  |  Execution clusters: {n_clusters_execution}",
        f"  ARI_inter at convergence: intent={_fmt(ari_inter_intent)}  execution={_fmt(ari_inter_execution)}",
        f"  Noise floor: intent={ari_boot_p05_intent:.3f}  execution={ari_boot_p05_execution:.3f}",
    ]
    if cluster_labels_intent:
        lines.append("  Intent cluster labels:")
        for cid, label in sorted(cluster_labels_intent.items()):
            lines.append(f"    [{cid}] {label}")
    if numerical_metrics:
        lines.append("  Numerical metrics (final batch):")
        for key, val in numerical_metrics.items():
            if isinstance(val, dict):
                pt = val.get("point")
                lo = val.get("ci_lo")
                hi = val.get("ci_hi")
                lines.append(f"    {key}: {pt}  CI=[{lo}, {hi}]")
    return "\n".join(lines)


def _fmt(v: Optional[float]) -> str:
    return f"{v:.3f}" if v is not None else "  N/A"


def _pct(d: dict) -> str:
    pt = d.get("point")
    lo = d.get("ci_lo")
    hi = d.get("ci_hi")
    if pt is None:
        return "N/A"
    return f"{pt:.1%} [{lo:.1%}, {hi:.1%}]"


def _pt(d: dict) -> str:
    pt = d.get("point")
    return f"{pt:.2f}" if pt is not None else "N/A"
