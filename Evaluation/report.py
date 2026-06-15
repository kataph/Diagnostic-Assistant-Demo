"""
Report generation for the adaptive evaluation protocol.
Produces human-readable text summaries (no HTML, no PDF).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from Evaluation.qualitative import QualitativeReport


def format_iteration_report(
    scenario_number: int,
    batch_index: int,
    n_trajectories: int,
    n_clusters_intent: int,
    n_clusters_execution: int,
    ari_inter_intent: Optional[float],
    ari_inter_execution: Optional[float],
    ari_boot_noise_floor_intent: float,
    ari_boot_noise_floor_execution: float,
    converged_intent: bool,
    converged_execution: bool,
    streak_intent: int,
    streak_execution: int,
    convergence_window: int,
    ari_min_threshold: float = 0.0,
    numerical_metrics: Optional[dict] = None,
    recommendation: str = "CONTINUE",
) -> str:
    lines = [
        f"{'='*60}",
        f"Scenario {scenario_number} | Batch {batch_index} | N={n_trajectories}",
        f"{'='*60}",
        f"  Intent level:    clusters={n_clusters_intent:<3}  "
        f"ARI_inter={_fmt(ari_inter_intent)}  noise_floor={ari_boot_noise_floor_intent:.3f}  min_thr={ari_min_threshold:.3f}  "
        f"pass={'✓' if converged_intent else '✗'}  streak={streak_intent}/{convergence_window}",
        f"  Execution level: clusters={n_clusters_execution:<3}  "
        f"ARI_inter={_fmt(ari_inter_execution)}  noise_floor={ari_boot_noise_floor_execution:.3f}  min_thr={ari_min_threshold:.3f}  "
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
    ari_boot_noise_floor_intent: float,
    ari_boot_noise_floor_execution: float,
    cluster_labels_intent: Optional[dict[int, str]],
    numerical_metrics: dict,
    batch_history: list[dict],
    reason: str = "converged",
    qualitative_report: Optional["QualitativeReport"] = None,
) -> str:
    lines = [
        f"{'#'*60}",
        f"FINAL REPORT — Scenario {scenario_number}",
        f"{'#'*60}",
        f"  Reason: {reason}",
        f"  Total batches: {total_batches} | Total trajectories: {total_trajectories}",
        f"  Intent clusters: {n_clusters_intent}  |  Execution clusters: {n_clusters_execution}",
        f"  ARI_inter at convergence: intent={_fmt(ari_inter_intent)}  execution={_fmt(ari_inter_execution)}",
        f"  Noise floor: intent={ari_boot_noise_floor_intent:.3f}  execution={ari_boot_noise_floor_execution:.3f}",
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

    if qualitative_report is not None:
        lines.append("")
        lines.append("-" * 60)
        lines.append("QUALITATIVE ANALYSIS")
        lines.append("-" * 60)

        # Rubric evaluation
        if qualitative_report.rubric_scores:
            lines.append("")
            lines.append("[Rubric Evaluation]")
            for rs in qualitative_report.rubric_scores:
                lines.append(f"  Cluster {rs.cluster_id} — {rs.cluster_label}")
                for dim_name, dim in [
                    ("Actionability",        rs.actionability),
                    ("Diagnostic coherence", rs.diagnostic_coherence),
                    ("Efficiency",           rs.efficiency),
                    ("Consistency",          rs.consistency),
                    ("Evidence usage",       rs.evidence_usage),
                ]:
                    lines.append(f"    {dim_name:<22} {dim.rating:<6} — {dim.rationale}")
                lines.append(f"    Comment: {rs.overall_comment}")

        # Gold standard comparison
        if qualitative_report.gold_comparisons:
            lines.append("")
            lines.append("[Gold Standard Comparison]")
            for gc in qualitative_report.gold_comparisons:
                lines.append(f"  Cluster {gc.cluster_id} — {gc.cluster_label}")
                lines.append(f"    Fault:  {gc.injected_fault}")
                lines.append(f"    Gold:   {gc.gold_diagnosis}")
                lines.append(f"    Match:  {gc.match_level} — {gc.explanation}")

        # Emergent findings
        ef = qualitative_report.emergent_findings
        if ef is not None:
            lines.append("")
            lines.append("[Emergent Findings]")
            for section, items in [
                ("Failure modes",     ef.failure_modes),
                ("Novel strategies",  ef.novel_strategies),
                ("Inefficiencies",    ef.inefficiencies),
                ("Candidate metrics", ef.candidate_metrics),
            ]:
                if items:
                    lines.append(f"  {section}:")
                    for item in items:
                        lines.append(f"    • {item}")

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
