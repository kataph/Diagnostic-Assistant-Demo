"""
make_tables.py — generate LaTeX evaluation tables from adaptive protocol checkpoints.

Usage
-----
# Per-run detailed table (ARI, CI intervals, clusters) for one or more runs:
python make_tables.py --run EvidenceKGOptimal:20260613T195135
python make_tables.py --run EvidenceKGOptimal:20260613T195135 --run LLM:20260613T195146

# Auto-detect the latest run for every assistant type found:
python make_tables.py

# Custom checkpoint directory (default: Logs/Evaluation):
python make_tables.py --checkpoint-dir Logs/Evaluation --run EvidenceKGOptimal:20260613T195135

Output (printed to stdout, LaTeX also copied to clipboard if pyperclip is installed)
------
  1. Per-run detailed table  — one row per scenario, one table per run
     (metrics: N trajectories, clusters, ARI, success rate, cost, actions)
  2. Cross-assistant comparison table — one column group per run, one row per scenario
     (metrics: success rate, cost, actions)
  3. System-level aggregate table — mean ± std per system, one column group per run
  4. Text summary
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# System metadata
# ---------------------------------------------------------------------------

SYSTEM_RANGES = {
    "3_cubes":              range(1, 26),
    "10_cubes":             range(26, 51),
    "asymmetric_chains":    range(51, 76),
    "ambient_light_sensor": range(76, 106),
    "current_sensor":       range(106, 136),
}

SYSTEM_LABELS = {
    "3_cubes":              "3-Cubes",
    "10_cubes":             "10-Cubes",
    "asymmetric_chains":    "Asym. Chains",
    "ambient_light_sensor": "Ambient L.S.",
    "current_sensor":       "Current S.",
}


def system_of(scenario_number: int) -> str:
    for sys, r in SYSTEM_RANGES.items():
        if scenario_number in r:
            return sys
    return "unknown"


# ---------------------------------------------------------------------------
# Loading — checkpoint.json (detailed per-run data)
# ---------------------------------------------------------------------------

def load_checkpoints(run_dir: Path) -> list[dict]:
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
# Loading — final_report.txt (summary metrics for comparison tables)
# ---------------------------------------------------------------------------

def _float(s: str) -> float:
    try:
        return float(s.strip())
    except ValueError:
        return float("nan")


def parse_report(path: Path) -> Optional[dict]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"FINAL REPORT — Scenario (\d+)", text)
    if not m:
        return None
    scenario = int(m.group(1))

    reason_m = re.search(r"Reason:\s+(\w+)", text)
    reason = reason_m.group(1) if reason_m else "unknown"

    batches_m = re.search(r"Total batches:\s*(\d+)", text)
    trajs_m   = re.search(r"Total trajectories:\s*(\d+)", text)
    n_batches = int(batches_m.group(1)) if batches_m else 0
    n_trajs   = int(trajs_m.group(1))   if trajs_m   else 0

    def metric(name: str) -> float:
        mm = re.search(rf"{re.escape(name)}:\s*([\d\.nan]+)", text)
        return _float(mm.group(1)) if mm else float("nan")

    return {
        "scenario":     scenario,
        "system":       system_of(scenario),
        "reason":       reason,
        "n_batches":    n_batches,
        "n_trajs":      n_trajs,
        "success_rate": metric("success_rate"),
        "total_cost":   metric("total_cost"),
        "n_actions":    metric("n_actions"),
        "n_hypotheses": metric("n_hypotheses"),
        "n_correct":    metric("n_correct"),
        "n_wrong":      metric("n_wrong"),
        "n_partial":    metric("n_partial"),
    }


def load_run_reports(checkpoint_dir: Path, assistant: str, run_id: str) -> dict[int, dict]:
    run_dir = checkpoint_dir / assistant / run_id
    results = {}
    for p in sorted(run_dir.rglob("final_report.txt")):
        d = parse_report(p)
        if d:
            results[d["scenario"]] = d
    return results


def latest_run_id(checkpoint_dir: Path, assistant: str) -> Optional[str]:
    base = checkpoint_dir / assistant
    if not base.exists():
        return None
    runs = sorted(d.name for d in base.iterdir() if d.is_dir())
    return runs[-1] if runs else None


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_ci(m: dict | None, pct: bool = False) -> str:
    if not m:
        return "---"
    pt, lo, hi = m.get("point"), m.get("ci_lo"), m.get("ci_hi")
    if pt is None:
        return "---"
    if pct:
        return f"{pt:.1%} [{lo:.1%}, {hi:.1%}]"
    return f"{pt:.2f} [{lo:.2f}, {hi:.2f}]"


def _fmt_ari(ari: float | None, floor: float | None) -> str:
    ari_s   = f"{ari:.3f}"   if ari   is not None else "---"
    floor_s = f"{floor:.3f}" if floor is not None else "---"
    return f"{ari_s} ({floor_s})"


def fmt(v: float, pct: bool = False) -> str:
    if v != v:
        return "--"
    return f"{v * 100:.1f}\\%" if pct else f"{v:.2f}"


def fmt_mean_std(vals: list[float], pct: bool = False) -> str:
    clean = [v for v in vals if v == v]
    if not clean:
        return "--"
    mu = statistics.mean(clean)
    sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
    if pct:
        return f"{mu*100:.1f}$\\pm${sd*100:.1f}\\%" if len(clean) > 1 else f"{mu*100:.1f}\\%"
    return f"{mu:.2f}$\\pm${sd:.2f}" if len(clean) > 1 else f"{mu:.2f}"


# ---------------------------------------------------------------------------
# Table 1: detailed per-run table (one table per assistant run)
# ---------------------------------------------------------------------------

def detailed_run_table(run_dir: Path, assistant_label: str, run_id: str) -> str:
    checkpoints = load_checkpoints(run_dir)
    if not checkpoints:
        return f"% No checkpoint.json files found under {run_dir}\n"

    try:
        import pandas as pd
        use_pandas = True
    except ImportError:
        use_pandas = False

    rows = []
    for ckpt in checkpoints:
        last = ckpt["batch_history"][-1] if ckpt.get("batch_history") else {}
        nm   = last.get("numerical_metrics", {})
        rows.append({
            "Scenario":           ckpt["scenario_number"],
            "N":                  ckpt.get("n_trajectories", "?"),
            "Clusters I/E":       f"{last.get('n_clusters_intent', '?')}/{last.get('n_clusters_execution', '?')}",
            "ARI intent (floor)": _fmt_ari(last.get("ari_inter_intent"),    last.get("ari_boot_p05_intent")),
            "ARI exec (floor)":   _fmt_ari(last.get("ari_inter_execution"), last.get("ari_boot_p05_execution")),
            "Success rate":       _fmt_ci(nm.get("success_rate"), pct=True),
            "Correct hyp.":       _fmt_ci(nm.get("correct_hypothesis_rate"), pct=True),
            "Total cost":         _fmt_ci(nm.get("total_cost")),
            "Actions":            _fmt_ci(nm.get("n_actions")),
        })

    caption = (
        f"Evaluation results for \\texttt{{{assistant_label}}} "
        f"(run~\\texttt{{{run_id}}}, {len(rows)}~scenario(s)). "
        "Each row reports metrics at protocol convergence. "
        "\\textit{N}: total trajectories. "
        "\\textit{Clusters I/E}: intent / execution cluster count. "
        "\\textit{ARI}: inter-batch Adjusted Rand Index (noise floor in parentheses). "
        "Confidence intervals: Wilson score for rates, percentile bootstrap for others (95\\%, 1000 resamples)."
    )
    label = f"tab:eval:{assistant_label.lower().replace(' ', '_')}:{run_id}"

    if use_pandas:
        import pandas as pd
        df = pd.DataFrame(rows)
        latex = df.to_latex(index=False, caption=caption, label=label,
                            escape=True, float_format="%.2f")
        text  = df.to_string(index=False)
    else:
        # Fallback: plain text + manual LaTeX
        col_names = list(rows[0].keys())
        widths = [max(len(str(r[c])) for r in rows + [{"": c}]) for c in col_names]  # type: ignore[dict-item]
        text = "  ".join(c.ljust(w) for c, w in zip(col_names, widths))
        text += "\n" + "-" * len(text) + "\n"
        text += "\n".join(
            "  ".join(str(r[c]).ljust(w) for c, w in zip(col_names, widths))
            for r in rows
        )
        col_spec = "r" * len(col_names)
        latex = (
            f"\\begin{{table}}[htbp]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
            + " & ".join(col_names) + " \\\\\n\\midrule\n"
            + "\n".join(" & ".join(str(r[c]) for c in col_names) + " \\\\" for r in rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    return text, latex


# ---------------------------------------------------------------------------
# Table 2: cross-assistant comparison (per-scenario)
# ---------------------------------------------------------------------------

def per_scenario_table(runs: list[tuple[str, dict[int, dict]]]) -> str:
    all_scenarios = sorted(set().union(*(d.keys() for _, d in runs)))
    n = len(runs)
    col_spec  = "rl" + "rrr" * n
    mid_cols  = "".join(f"\\cmidrule(lr){{{3 + i*3}-{5 + i*3}}}" for i in range(n))
    hdr_names = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in runs)
    hdr_cols  = " & ".join("Succ & Cost & Acts" for _ in runs)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{Per-scenario comparison: {', '.join(name for name, _ in runs)}}}",
        "\\label{tab:per_scenario}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"\\multicolumn{{2}}{{c}}{{Scenario}} & {hdr_names} \\\\",
        f"\\cmidrule(lr){{1-2}}{mid_cols}",
        f"\\# & System & {hdr_cols} \\\\",
        "\\midrule",
    ]
    for snum in all_scenarios:
        sys_label = SYSTEM_LABELS.get(system_of(snum), system_of(snum))
        cells = " & ".join(
            f"{fmt(d.get(snum, {}).get('success_rate', float('nan')), pct=True)} & "
            f"{fmt(d.get(snum, {}).get('total_cost',   float('nan')))} & "
            f"{fmt(d.get(snum, {}).get('n_actions',    float('nan')))}"
            for _, d in runs
        )
        lines.append(f"{snum} & {sys_label} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: system-level aggregate
# ---------------------------------------------------------------------------

def aggregate_table(runs: list[tuple[str, dict[int, dict]]]) -> str:
    n = len(runs)
    col_spec  = "l" + "rrr" * n
    mid_cols  = "".join(f"\\cmidrule(lr){{{2 + i*3}-{4 + i*3}}}" for i in range(n))
    hdr_names = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in runs)
    hdr_cols  = " & ".join("Succ & Cost & Acts" for _ in runs)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{System-level aggregate: mean $\\pm$ std across scenarios}",
        "\\label{tab:aggregate}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f" & {hdr_names} \\\\",
        mid_cols,
        f"System & {hdr_cols} \\\\",
        "\\midrule",
    ]
    for sys, rng in SYSTEM_RANGES.items():
        nums = list(rng)
        cells = " & ".join(
            f"{fmt_mean_std([d[n]['success_rate'] for n in nums if n in d], pct=True)} & "
            f"{fmt_mean_std([d[n]['total_cost']   for n in nums if n in d])} & "
            f"{fmt_mean_std([d[n]['n_actions']    for n in nums if n in d])}"
            for _, d in runs
        )
        lines.append(f"{SYSTEM_LABELS[sys]} & {cells} \\\\")

    overall_cells = " & ".join(
        f"\\textbf{{{fmt_mean_std([r['success_rate'] for r in d.values()], pct=True)}}} & "
        f"\\textbf{{{fmt_mean_std([r['total_cost']   for r in d.values()])}}} & "
        f"\\textbf{{{fmt_mean_std([r['n_actions']    for r in d.values()])}}}"
        for _, d in runs
    )
    lines += ["\\midrule", f"\\textbf{{Overall}} & {overall_cells} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Text summary
# ---------------------------------------------------------------------------

def text_summary(runs: list[tuple[str, dict[int, dict]]]) -> str:
    def agg(data: dict[int, dict], key: str) -> tuple[float, float]:
        vals = [d[key] for d in data.values() if d.get(key) == d.get(key)]
        if not vals:
            return float("nan"), float("nan")
        mu = statistics.mean(vals)
        return mu, (statistics.stdev(vals) if len(vals) > 1 else 0.0)

    lines = ["=" * 60, "TEXT SUMMARY", "=" * 60]
    for name, data in runs:
        lines.append(f"\n--- {name} ---")
        lines.append(f"  Scenarios with data: {len(data)}")
        mu, sd = agg(data, "success_rate")
        lines.append(f"  Success rate:  {mu*100:.1f}% ± {sd*100:.1f}%")
        mu, sd = agg(data, "total_cost")
        lines.append(f"  Avg cost:      {mu:.1f} ± {sd:.1f}")
        mu, sd = agg(data, "n_actions")
        lines.append(f"  Avg actions:   {mu:.2f} ± {sd:.2f}")
        converged = sum(1 for d in data.values() if d.get("reason") == "converged")
        lines.append(f"  Converged:     {converged}/{len(data)} scenarios")

    col_w = 18
    lines.append("\n--- Per-system success rate ---")
    header = f"{'System':<22}" + "".join(f"{name:>{col_w}}" for name, _ in runs)
    lines += [header, "-" * len(header)]
    for sys, rng in SYSTEM_RANGES.items():
        nums = list(rng)
        row  = f"{SYSTEM_LABELS[sys]:<22}"
        for _, data in runs:
            vals = [data[n]["success_rate"] for n in nums if n in data]
            row += f"{(f'{statistics.mean(vals)*100:.1f}%') if vals else '--':>{col_w}}"
        lines.append(row)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX evaluation tables from adaptive protocol checkpoints."
    )
    parser.add_argument(
        "--run", action="append", metavar="ASSISTANT:RUN_ID",
        help="Run to include, as AssistantType:run_id (repeatable). "
             "If omitted, auto-detects the latest run per assistant type.",
    )
    parser.add_argument(
        "--checkpoint-dir", default="Logs/Evaluation",
        help="Root directory containing per-assistant run subdirectories (default: Logs/Evaluation).",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)

    if args.run:
        specs: list[tuple[str, str]] = []
        for token in args.run:
            if ":" not in token:
                parser.error(f"--run must be AssistantType:run_id, got {token!r}")
            assistant, run_id = token.split(":", 1)
            specs.append((assistant, run_id))
    else:
        specs = []
        if ckpt.exists():
            for d in sorted(ckpt.iterdir()):
                if d.is_dir():
                    run_id = latest_run_id(ckpt, d.name)
                    if run_id:
                        specs.append((d.name, run_id))
        if not specs:
            print("No runs found. Use --run AssistantType:run_id or run the evaluation protocol first.")
            sys.exit(1)

    all_latex_parts: list[str] = []

    # --- Table 1: detailed per-run tables ---
    print("=" * 60)
    print("DETAILED PER-RUN TABLES")
    print("=" * 60)
    for assistant, run_id in specs:
        run_dir = ckpt / assistant / run_id
        print(f"\n>>> {assistant} / {run_id}")
        if not run_dir.exists():
            print(f"  Directory not found: {run_dir}")
            continue
        text, latex = detailed_run_table(run_dir, assistant, run_id)
        print(text)
        all_latex_parts.append(f"% === Detailed table: {assistant} / {run_id} ===")
        all_latex_parts.append(latex)

    # --- Load final_report.txt data for comparison tables ---
    runs: list[tuple[str, dict[int, dict]]] = []
    for assistant, run_id in specs:
        data = load_run_reports(ckpt, assistant, run_id)
        if data:
            runs.append((assistant, data))
        else:
            print(f"\n(No final_report.txt found for {assistant}/{run_id} — skipping comparison tables for this run)")

    if runs:
        print("\n")
        print(text_summary(runs))

        comparison = per_scenario_table(runs)
        aggregate  = aggregate_table(runs)

        print("\n% === Per-scenario comparison table ===")
        print(comparison)
        print("\n% === System-level aggregate table ===")
        print(aggregate)

        all_latex_parts.append("% === Per-scenario comparison table ===")
        all_latex_parts.append(comparison)
        all_latex_parts.append("% === System-level aggregate table ===")
        all_latex_parts.append(aggregate)

    # --- Clipboard ---
    full_latex = "\n\n".join(all_latex_parts)
    try:
        import pyperclip
        pyperclip.copy(full_latex)
        print("\n(All LaTeX copied to clipboard.)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
