"""
make_tables.py — generate LaTeX evaluation tables from adaptive protocol checkpoints.

Usage:
    python make_tables.py --run-dir Logs/Evaluation/RandomTrajectory/20260605T175829
    python make_tables.py --run-dir Logs/Evaluation/RandomTrajectory/20260605T175829 --assistant LLM
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_checkpoints(run_dir: Path) -> list[dict]:
    """Load all checkpoint.json files under run_dir/<scenario_number>/checkpoint.json."""
    paths = sorted(
        run_dir.glob("*/checkpoint.json"),
        key=lambda p: int(p.parent.name),
    )
    checkpoints = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            checkpoints.append(json.load(f))
    return checkpoints


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(m: dict | None, pct: bool = False) -> str:
    """Format a metric dict {point, ci_lo, ci_hi} as 'point [lo, hi]'."""
    if not m:
        return "---"
    pt, lo, hi = m.get("point"), m.get("ci_lo"), m.get("ci_hi")
    if pt is None:
        return "---"
    if pct:
        return f"{pt:.1%} [{lo:.1%}, {hi:.1%}]"
    return f"{pt:.2f} [{lo:.2f}, {hi:.2f}]"


def _fmt_ari(ari: float | None, floor: float | None) -> str:
    """Format ARI and its bootstrap noise floor as 'ari (floor)'."""
    ari_s   = f"{ari:.3f}"   if ari   is not None else "---"
    floor_s = f"{floor:.3f}" if floor is not None else "---"
    return f"{ari_s} ({floor_s})"


# ---------------------------------------------------------------------------
# Table builder
# ---------------------------------------------------------------------------

def build_table(checkpoints: list[dict]) -> pd.DataFrame:
    rows = []
    for ckpt in checkpoints:
        last = ckpt["batch_history"][-1] if ckpt.get("batch_history") else {}
        nm   = last.get("numerical_metrics", {})
        rows.append({
            "Scenario":              ckpt["scenario_number"],
            "N":                     ckpt.get("n_trajectories", "?"),
            "Clusters I / E":        f"{last.get('n_clusters_intent', '?')} / {last.get('n_clusters_execution', '?')}",
            "ARI intent (floor)":    _fmt_ari(last.get("ari_inter_intent"),    last.get("ari_boot_p05_intent")),
            "ARI exec (floor)":      _fmt_ari(last.get("ari_inter_execution"), last.get("ari_boot_p05_execution")),
            "Success rate":          _fmt(nm.get("success_rate"), pct=True),
            "Correct hyp. rate":     _fmt(nm.get("correct_hypothesis_rate"), pct=True),
            "Total cost":            _fmt(nm.get("total_cost")),
            "Actions":               _fmt(nm.get("n_actions")),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Caption
# ---------------------------------------------------------------------------

def build_caption(assistant_label: str, run_id: str, n_scenarios: int) -> str:
    return (
        f"Evaluation results for the \\texttt{{{assistant_label}}} diagnostic assistant "
        f"(run~\\texttt{{{run_id}}}, {n_scenarios}~scenario(s)). "
        "Each row reports metrics from the final trajectory batch at protocol convergence. "
        "\\textit{N} is the total number of trajectories collected. "
        "\\textit{Clusters I\\,/\\,E} is the number of distinct trajectory clusters "
        "at the intent level (HDBSCAN on sentence-embedding pairwise distances) "
        "and at the execution level (agglomerative hierarchical clustering, average linkage), "
        "respectively. "
        "\\textit{ARI intent} and \\textit{ARI exec} are the inter-batch Adjusted Rand Index "
        "values at convergence for intent-level and execution-level clustering; "
        "the value in parentheses is the 5th-percentile bootstrap noise floor "
        "(1000 resamples with replacement), used as the convergence threshold --- "
        "convergence is declared when ARI exceeds this floor for the required streak. "
        "\\textit{Success rate} and \\textit{Correct hyp.~rate} confidence intervals "
        "use the Wilson score method (95\\%), treating each trial as a Bernoulli event "
        "pooled across all trajectories. "
        "All other metric confidence intervals are percentile bootstrap intervals "
        "(95\\%, 1000 resamples of the mean)."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a LaTeX table from adaptive evaluation protocol checkpoints."
    )
    parser.add_argument(
        "--run-dir", required=True, type=Path,
        help="Path to the protocol run directory "
             "(e.g. Logs/Evaluation/RandomTrajectory/20260605T175829)",
    )
    parser.add_argument(
        "--assistant", default=None,
        help="Assistant label for the caption (defaults to the parent directory name).",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        print(f"ERROR: run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    checkpoints = load_checkpoints(run_dir)
    if not checkpoints:
        print(f"ERROR: no checkpoint.json files found under {run_dir}", file=sys.stderr)
        sys.exit(1)

    assistant_label = args.assistant or run_dir.parent.name
    run_id          = run_dir.name
    n_scenarios     = len(checkpoints)

    df      = build_table(checkpoints)
    caption = build_caption(assistant_label, run_id, n_scenarios)
    label   = f"tab:eval:{assistant_label.lower()}:{run_id}"

    latex = df.to_latex(
        index=False,
        caption=caption,
        label=label,
        escape=True,   # escape data cells; caption is passed through as-is
        float_format="%.2f",
    )

    print(df.to_string(index=False))
    print()
    print(latex)

    try:
        import pyperclip
        pyperclip.copy(latex)
        print("(LaTeX copied to clipboard.)")
    except Exception:
        pass


if __name__ == "__main__":
    main()