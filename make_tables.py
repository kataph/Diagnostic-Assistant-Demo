"""
make_tables.py — generate evaluation tables from adaptive protocol checkpoints.

Usage
-----
python make_tables.py --run EvidenceKGOptimal:20260613T195135
python make_tables.py --run EvidenceKGOptimal:20260613T195135 --run LLM:20260613T195146
python make_tables.py          # auto-detect latest run per assistant type

Output (printed to stdout; LaTeX copied to clipboard if pyperclip is installed)
------
  1. Per-run detailed table       — one row per scenario
  2. Cross-assistant comparison   — per-scenario, all runs side by side
  3. System-level aggregate       — mean ± std per system
  4. Fault-type aggregate         — mean ± std per fault category
  5. Metric breakdown matrices    — one matrix per metric: systems × fault types
  6. Qualitative aggregate        — rubric rating distributions + gold match levels
                                    broken down by system and fault type
  7. Text summary
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
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

FAULT_TYPE_ORDER = [
    "Simple", "Double", "Intermittent", "Misleading",
    "Coupled", "Limited Observability", "Unforeseen Interaction", "Multiple", "Triple",
]

RUBRIC_DIMS = [
    "actionability", "diagnostic_coherence", "efficiency", "consistency", "evidence_usage"
]

RUBRIC_DIM_LABELS = {
    "actionability":       "Actionability",
    "diagnostic_coherence":"Diag. coherence",
    "efficiency":          "Efficiency",
    "consistency":         "Consistency",
    "evidence_usage":      "Evidence usage",
}

RATING_ORDER = ["high", "medium", "low", "N/A"]

MATCH_LEVELS = ["correct", "different_strategy", "partial", "incorrect", "N/A"]

# ---------------------------------------------------------------------------
# SCENARIOS_MASTER.csv — scenario metadata
# ---------------------------------------------------------------------------

_SCENARIO_META: dict[int, dict] = {}


def _load_scenario_meta(csv_path: Optional[Path] = None) -> dict[int, dict]:
    global _SCENARIO_META
    if _SCENARIO_META:
        return _SCENARIO_META
    if csv_path is None:
        csv_path = Path(__file__).resolve().parent / "SCENARIOS_MASTER.csv"
    if not csv_path.exists():
        return {}
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                num = int(row.get("scenario number", "").strip())
            except ValueError:
                continue
            raw_tags = row.get("category", "").strip()
            tags = [t.strip() for t in raw_tags.split(",") if t.strip()] if raw_tags else ["Simple"]
            _SCENARIO_META[num] = {
                "system":         system_of(num),
                "fault_tags":     tags,        # list[str], e.g. ["Misleading", "Double"]
                "injected_fault": row.get("injected fault", "").strip(),
                "ideal_diagnosis": (
                    list(row.values())[9].strip() if len(row) > 9 else ""
                ),
            }
    return _SCENARIO_META


def scenario_meta(num: int) -> dict:
    meta = _load_scenario_meta()
    return meta.get(num, {"system": system_of(num), "fault_tags": ["Simple"],
                          "injected_fault": "", "ideal_diagnosis": ""})


def system_of(scenario_number: int) -> str:
    for sys, r in SYSTEM_RANGES.items():
        if scenario_number in r:
            return sys
    return "unknown"


def all_fault_types(data: dict[int, dict]) -> list[str]:
    """Return sorted fault type tags that appear in data, respecting FAULT_TYPE_ORDER."""
    present = set()
    for num in data:
        for tag in scenario_meta(num)["fault_tags"]:
            present.add(tag)
    ordered = [t for t in FAULT_TYPE_ORDER if t in present]
    extra   = sorted(present - set(FAULT_TYPE_ORDER))
    return ordered + extra


# ---------------------------------------------------------------------------
# Loading — checkpoint.json
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
# Loading — final_report.txt
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

    reason_m  = re.search(r"Reason:\s+(\w+)", text)
    batches_m = re.search(r"Total batches:\s*(\d+)", text)
    trajs_m   = re.search(r"Total trajectories:\s*(\d+)", text)

    def metric(name: str) -> float:
        mm = re.search(rf"{re.escape(name)}:\s*([\d\.nan]+)", text)
        return _float(mm.group(1)) if mm else float("nan")

    meta = scenario_meta(scenario)
    return {
        "scenario":      scenario,
        "system":        meta["system"],
        "fault_tags":    meta["fault_tags"],
        "injected_fault": meta["injected_fault"],
        "reason":        reason_m.group(1) if reason_m else "unknown",
        "n_batches":     int(batches_m.group(1)) if batches_m else 0,
        "n_trajs":       int(trajs_m.group(1))   if trajs_m   else 0,
        "success_rate":  metric("success_rate"),
        "total_cost":    metric("total_cost"),
        "n_actions":     metric("n_actions"),
        "n_hypotheses":  metric("n_hypotheses"),
        "n_correct":     metric("n_correct"),
        "n_wrong":       metric("n_wrong"),
        "n_partial":     metric("n_partial"),
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
# Loading — qualitative.json
# ---------------------------------------------------------------------------

def load_qualitative(checkpoint_dir: Path, assistant: str, run_id: str) -> dict[int, dict]:
    """Return {scenario_number: qualitative_json_dict} for all scenarios in a run."""
    run_dir = checkpoint_dir / assistant / run_id
    results = {}
    for p in sorted(run_dir.rglob("qualitative.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            num  = data.get("scenario_number")
            if num is not None:
                results[int(num)] = data
        except Exception:
            continue
    return results


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


def _mean(vals: list[float]) -> float:
    clean = [v for v in vals if v == v]
    return statistics.mean(clean) if clean else float("nan")


# ---------------------------------------------------------------------------
# Table 1: detailed per-run table
# ---------------------------------------------------------------------------

def detailed_run_table(run_dir: Path, assistant_label: str, run_id: str) -> tuple[str, str]:
    checkpoints = load_checkpoints(run_dir)
    if not checkpoints:
        return f"% No checkpoint.json files found under {run_dir}\n", ""

    rows = []
    for ckpt in checkpoints:
        last = ckpt["batch_history"][-1] if ckpt.get("batch_history") else {}
        nm   = last.get("numerical_metrics", {})
        snum = ckpt["scenario_number"]
        meta = scenario_meta(snum)
        rows.append({
            "Scenario":           snum,
            "System":             SYSTEM_LABELS.get(meta["system"], meta["system"]),
            "Fault type":         ", ".join(meta["fault_tags"]),
            "N":                  ckpt.get("n_trajectories", "?"),
            "Clusters I/E":       f"{last.get('n_clusters_intent','?')}/{last.get('n_clusters_execution','?')}",
            "ARI intent (floor)": _fmt_ari(last.get("ari_inter_intent"),
                                           last.get("ari_boot_noise_floor_intent") or last.get("ari_boot_p05_intent")),
            "ARI exec (floor)":   _fmt_ari(last.get("ari_inter_execution"),
                                           last.get("ari_boot_noise_floor_execution") or last.get("ari_boot_p05_execution")),
            "Success rate":       _fmt_ci(nm.get("success_rate"), pct=True),
            "Correct hyp.":       _fmt_ci(nm.get("correct_hypothesis_rate"), pct=True),
            "Total cost":         _fmt_ci(nm.get("total_cost")),
            "Actions":            _fmt_ci(nm.get("n_actions")),
        })

    caption = (
        f"Evaluation results for \\texttt{{{assistant_label}}} "
        f"(run~\\texttt{{{run_id}}}, {len(rows)}~scenario(s))."
    )
    label = f"tab:eval:{assistant_label.lower().replace(' ','_')}:{run_id}"

    try:
        import pandas as pd
        df    = pd.DataFrame(rows)
        latex = df.to_latex(index=False, caption=caption, label=label, escape=True)
        text  = df.to_string(index=False)
    except ImportError:
        col_names = list(rows[0].keys())
        widths = [max(len(str(r[c])) for r in [{"": c}] + rows) for c in col_names]  # type: ignore[dict-item]
        text  = "  ".join(c.ljust(w) for c, w in zip(col_names, widths))
        text += "\n" + "-" * len(text) + "\n"
        text += "\n".join("  ".join(str(r[c]).ljust(w) for c, w in zip(col_names, widths)) for r in rows)
        col_spec = "r" * len(col_names)
        latex = (
            f"\\begin{{table}}[htbp]\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{col_spec}}}\\toprule\n"
            + " & ".join(col_names) + " \\\\\n\\midrule\n"
            + "\n".join(" & ".join(str(r[c]) for c in col_names) + " \\\\" for r in rows)
            + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )

    return text, latex


# ---------------------------------------------------------------------------
# Table 2: cross-assistant per-scenario comparison
# ---------------------------------------------------------------------------

def per_scenario_table(runs: list[tuple[str, dict[int, dict]]]) -> str:
    all_scenarios = sorted(set().union(*(d.keys() for _, d in runs)))
    n = len(runs)
    col_spec  = "rll" + "rrr" * n
    mid_cols  = "".join(f"\\cmidrule(lr){{{4 + i*3}-{6 + i*3}}}" for i in range(n))
    hdr_names = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in runs)
    hdr_cols  = " & ".join("Succ & Cost & Acts" for _ in runs)

    lines = [
        "\\begin{table}[htbp]", "\\centering",
        f"\\caption{{Per-scenario comparison: {', '.join(name for name, _ in runs)}}}",
        "\\label{tab:per_scenario}", "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
        f"\\multicolumn{{3}}{{c}}{{Scenario}} & {hdr_names} \\\\",
        f"\\cmidrule(lr){{1-3}}{mid_cols}",
        f"\\# & System & Fault type & {hdr_cols} \\\\", "\\midrule",
    ]
    for snum in all_scenarios:
        meta = scenario_meta(snum)
        sys_label   = SYSTEM_LABELS.get(meta["system"], meta["system"])
        fault_label = ", ".join(meta["fault_tags"])
        cells = " & ".join(
            f"{fmt(d.get(snum, {}).get('success_rate', float('nan')), pct=True)} & "
            f"{fmt(d.get(snum, {}).get('total_cost',   float('nan')))} & "
            f"{fmt(d.get(snum, {}).get('n_actions',    float('nan')))}"
            for _, d in runs
        )
        lines.append(f"{snum} & {sys_label} & {fault_label} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 3: system-level aggregate
# ---------------------------------------------------------------------------

def aggregate_table_by_system(runs: list[tuple[str, dict[int, dict]]]) -> str:
    n = len(runs)
    col_spec  = "l" + "rrr" * n
    mid_cols  = "".join(f"\\cmidrule(lr){{{2 + i*3}-{4 + i*3}}}" for i in range(n))
    hdr_names = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in runs)
    hdr_cols  = " & ".join("Succ & Cost & Acts" for _ in runs)

    lines = [
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{System-level aggregate (mean $\\pm$ std)}",
        "\\label{tab:aggregate_system}",
        f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
        f" & {hdr_names} \\\\", mid_cols,
        f"System & {hdr_cols} \\\\", "\\midrule",
    ]
    for sys, rng in SYSTEM_RANGES.items():
        nums  = list(rng)
        cells = " & ".join(
            f"{fmt_mean_std([d[n]['success_rate'] for n in nums if n in d], pct=True)} & "
            f"{fmt_mean_std([d[n]['total_cost']   for n in nums if n in d])} & "
            f"{fmt_mean_std([d[n]['n_actions']    for n in nums if n in d])}"
            for _, d in runs
        )
        lines.append(f"{SYSTEM_LABELS[sys]} & {cells} \\\\")

    overall = " & ".join(
        f"\\textbf{{{fmt_mean_std([r['success_rate'] for r in d.values()], pct=True)}}} & "
        f"\\textbf{{{fmt_mean_std([r['total_cost']   for r in d.values()])}}} & "
        f"\\textbf{{{fmt_mean_std([r['n_actions']    for r in d.values()])}}}"
        for _, d in runs
    )
    lines += ["\\midrule", f"\\textbf{{Overall}} & {overall} \\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 4: fault-type aggregate
# ---------------------------------------------------------------------------

def aggregate_table_by_fault_type(runs: list[tuple[str, dict[int, dict]]]) -> str:
    all_data = {num: row for _, d in runs for num, row in d.items()}
    fault_types = all_fault_types(all_data)
    n = len(runs)

    col_spec  = "l" + "rrr" * n
    mid_cols  = "".join(f"\\cmidrule(lr){{{2 + i*3}-{4 + i*3}}}" for i in range(n))
    hdr_names = " & ".join(f"\\multicolumn{{3}}{{c}}{{{name}}}" for name, _ in runs)
    hdr_cols  = " & ".join("Succ & Cost & Acts" for _ in runs)

    lines = [
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Fault-type aggregate (mean $\\pm$ std; scenarios may appear in multiple rows)}",
        "\\label{tab:aggregate_fault}",
        f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
        f" & {hdr_names} \\\\", mid_cols,
        f"Fault type & {hdr_cols} \\\\", "\\midrule",
    ]
    for ft in fault_types:
        cells = " & ".join(
            f"{fmt_mean_std([r['success_rate'] for r in d.values() if ft in r['fault_tags']], pct=True)} & "
            f"{fmt_mean_std([r['total_cost']   for r in d.values() if ft in r['fault_tags']])} & "
            f"{fmt_mean_std([r['n_actions']    for r in d.values() if ft in r['fault_tags']])}"
            for _, d in runs
        )
        lines.append(f"{ft} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 5: metric breakdown matrices (system × fault type)
# ---------------------------------------------------------------------------

def metric_matrix_tables(runs: list[tuple[str, dict[int, dict]]]) -> list[str]:
    """One LaTeX table per metric per run: rows=systems, cols=fault types, cells=mean."""
    all_data = {num: row for _, d in runs for num, row in d.items()}
    fault_types = all_fault_types(all_data)

    metrics = [
        ("success_rate", "Success rate", True),
        ("total_cost",   "Total cost",   False),
        ("n_actions",    "Actions",      False),
    ]

    tables = []
    for run_name, data in runs:
        for metric_key, metric_label, is_pct in metrics:
            col_spec = "l" + "r" * len(fault_types)
            lines = [
                "\\begin{table}[htbp]", "\\centering",
                f"\\caption{{{metric_label} by system and fault type — {run_name}}}",
                f"\\label{{tab:matrix:{run_name.lower().replace(' ','_')}:{metric_key}}}",
                f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
                "System & " + " & ".join(ft.replace(" ", "~") for ft in fault_types) + " \\\\",
                "\\midrule",
            ]
            for sys, rng in SYSTEM_RANGES.items():
                nums = list(rng)
                cells = []
                for ft in fault_types:
                    vals = [
                        data[n][metric_key] for n in nums
                        if n in data and ft in data[n]["fault_tags"]
                    ]
                    cells.append(fmt_mean_std(vals, pct=is_pct) if vals else "--")
                lines.append(f"{SYSTEM_LABELS[sys]} & " + " & ".join(cells) + " \\\\")

            # Overall row
            overall_cells = []
            for ft in fault_types:
                vals = [r[metric_key] for r in data.values() if ft in r["fault_tags"]]
                overall_cells.append(fmt_mean_std(vals, pct=is_pct) if vals else "--")
            lines += [
                "\\midrule",
                "\\textbf{Overall} & " + " & ".join(f"\\textbf{{{c}}}" for c in overall_cells) + " \\\\",
                "\\bottomrule", "\\end{tabular}", "\\end{table}",
            ]
            tables.append("\n".join(lines))
    return tables


# ---------------------------------------------------------------------------
# Table 6: qualitative aggregation
# ---------------------------------------------------------------------------

def _collect_rubric_ratings(qual_data: dict[int, dict]) -> dict[str, dict[str, list[str]]]:
    """Returns {dimension: {scenario_num_str: [rating, ...]}} across all clusters."""
    per_dim: dict[str, list[str]] = {d: [] for d in RUBRIC_DIMS}
    for num, q in qual_data.items():
        for rs in q.get("rubric_scores", []):
            for dim in RUBRIC_DIMS:
                dim_val = rs.get(dim)
                if isinstance(dim_val, dict):
                    rating = dim_val.get("rating", "N/A").lower().strip()
                    per_dim[dim].append(rating)
    return per_dim


def _collect_match_levels(qual_data: dict[int, dict]) -> list[str]:
    levels = []
    for q in qual_data.values():
        for gc in q.get("gold_comparisons", []):
            levels.append(gc.get("match_level", "N/A").lower().strip())
    return levels


def _rating_dist_str(ratings: list[str]) -> str:
    if not ratings:
        return "--"
    total = len(ratings)
    parts = []
    for r in ["high", "medium", "low"]:
        c = ratings.count(r)
        if c:
            parts.append(f"{r[0].upper()}:{c}/{total}")
    return " ".join(parts) if parts else "--"


def _match_dist_str(levels: list[str]) -> str:
    if not levels:
        return "--"
    total = len(levels)
    parts = []
    for ml in ["correct", "different_strategy", "partial", "incorrect"]:
        c = levels.count(ml)
        if c:
            abbrev = {"correct": "C", "different_strategy": "D", "partial": "P", "incorrect": "I"}[ml]
            parts.append(f"{abbrev}:{c}/{total}")
    return " ".join(parts) if parts else "--"


def qualitative_aggregate_text(
    runs_qual: list[tuple[str, dict[int, dict]]],
    runs_data: list[tuple[str, dict[int, dict]]],
) -> str:
    """Plain-text qualitative aggregate tables: by system and by fault type."""
    if not runs_qual:
        return ""

    all_data = {num: row for _, d in runs_data for num, row in d.items()}
    fault_types = all_fault_types(all_data)

    lines = ["=" * 70, "QUALITATIVE AGGREGATE", "=" * 70]

    for run_name, qual_data in runs_qual:
        lines.append(f"\n--- {run_name} ---")

        # --- Rubric ratings by dimension ---
        per_dim = _collect_rubric_ratings(qual_data)
        lines.append("\nRubric rating distribution (H:high M:medium L:low):")
        col_w = 12
        header = f"  {'Dimension':<22}" + "".join(f"{'dist':>{col_w}}")
        for dim in RUBRIC_DIMS:
            ratings = per_dim[dim]
            lines.append(f"  {RUBRIC_DIM_LABELS[dim]:<22} {_rating_dist_str(ratings)}")

        # --- Rubric by system ---
        lines.append("\nRubric ratings by system:")
        header = f"  {'':22}" + "".join(f"{RUBRIC_DIM_LABELS[d]:>14}" for d in RUBRIC_DIMS)
        lines.append(header)
        for sys, rng in SYSTEM_RANGES.items():
            nums = set(rng) & set(qual_data.keys())
            if not nums:
                continue
            sub = {n: qual_data[n] for n in nums}
            per_dim_sys = _collect_rubric_ratings(sub)
            row = f"  {SYSTEM_LABELS[sys]:<22}" + "".join(
                f"{_rating_dist_str(per_dim_sys[d]):>14}" for d in RUBRIC_DIMS
            )
            lines.append(row)

        # --- Rubric by fault type ---
        lines.append("\nRubric ratings by fault type:")
        lines.append(header)
        for ft in fault_types:
            nums = {n for n in qual_data if ft in scenario_meta(n)["fault_tags"]}
            if not nums:
                continue
            sub = {n: qual_data[n] for n in nums}
            per_dim_ft = _collect_rubric_ratings(sub)
            row = f"  {ft:<22}" + "".join(
                f"{_rating_dist_str(per_dim_ft[d]):>14}" for d in RUBRIC_DIMS
            )
            lines.append(row)

        # --- Gold match levels ---
        all_levels = _collect_match_levels(qual_data)
        lines.append(f"\nGold match distribution (overall): {_match_dist_str(all_levels)}")
        lines.append("  C=correct  D=different_strategy  P=partial  I=incorrect")

        lines.append("\nGold match by system:")
        for sys, rng in SYSTEM_RANGES.items():
            nums = set(rng) & set(qual_data.keys())
            if not nums:
                continue
            sub = {n: qual_data[n] for n in nums}
            levels = _collect_match_levels(sub)
            lines.append(f"  {SYSTEM_LABELS[sys]:<22} {_match_dist_str(levels)}")

        lines.append("\nGold match by fault type:")
        for ft in fault_types:
            nums = {n for n in qual_data if ft in scenario_meta(n)["fault_tags"]}
            if not nums:
                continue
            sub = {n: qual_data[n] for n in nums}
            levels = _collect_match_levels(sub)
            lines.append(f"  {ft:<22} {_match_dist_str(levels)}")

    return "\n".join(lines)


def qualitative_aggregate_latex(
    runs_qual: list[tuple[str, dict[int, dict]]],
    runs_data: list[tuple[str, dict[int, dict]]],
) -> list[str]:
    """LaTeX tables for qualitative aggregation."""
    if not runs_qual:
        return []

    all_data = {num: row for _, d in runs_data for num, row in d.items()}
    fault_types = all_fault_types(all_data)
    tables = []

    # One table per run: rubric ratings × system, then rubric ratings × fault type
    for run_name, qual_data in runs_qual:
        per_dim_global = _collect_rubric_ratings(qual_data)
        dim_labels     = [RUBRIC_DIM_LABELS[d] for d in RUBRIC_DIMS]

        # --- Rubric by system ---
        col_spec = "l" + "r" * len(RUBRIC_DIMS)
        lines = [
            "\\begin{table}[htbp]", "\\centering",
            f"\\caption{{Rubric rating distribution by system — {run_name} (H/M/L counts)}}",
            f"\\label{{tab:rubric_system:{run_name.lower().replace(' ','_')}}}",
            f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
            "System & " + " & ".join(dl.replace(" ", "~") for dl in dim_labels) + " \\\\",
            "\\midrule",
        ]
        for sys, rng in SYSTEM_RANGES.items():
            nums = set(rng) & set(qual_data.keys())
            if not nums:
                continue
            sub = {n: qual_data[n] for n in nums}
            per_dim_sys = _collect_rubric_ratings(sub)
            cells = " & ".join(_rating_dist_str(per_dim_sys[d]) for d in RUBRIC_DIMS)
            lines.append(f"{SYSTEM_LABELS[sys]} & {cells} \\\\")
        lines += [
            "\\midrule",
            "\\textbf{Overall} & " + " & ".join(
                f"\\textbf{{{_rating_dist_str(per_dim_global[d])}}}" for d in RUBRIC_DIMS
            ) + " \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}",
        ]
        tables.append("\n".join(lines))

        # --- Rubric by fault type ---
        lines = [
            "\\begin{table}[htbp]", "\\centering",
            f"\\caption{{Rubric rating distribution by fault type — {run_name}}}",
            f"\\label{{tab:rubric_fault:{run_name.lower().replace(' ','_')}}}",
            f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
            "Fault type & " + " & ".join(dl.replace(" ", "~") for dl in dim_labels) + " \\\\",
            "\\midrule",
        ]
        for ft in fault_types:
            nums = {n for n in qual_data if ft in scenario_meta(n)["fault_tags"]}
            if not nums:
                continue
            sub = {n: qual_data[n] for n in nums}
            per_dim_ft = _collect_rubric_ratings(sub)
            cells = " & ".join(_rating_dist_str(per_dim_ft[d]) for d in RUBRIC_DIMS)
            lines.append(f"{ft} & {cells} \\\\")
        lines += [
            "\\midrule",
            "\\textbf{Overall} & " + " & ".join(
                f"\\textbf{{{_rating_dist_str(per_dim_global[d])}}}" for d in RUBRIC_DIMS
            ) + " \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}",
        ]
        tables.append("\n".join(lines))

        # --- Gold match by system ---
        lines = [
            "\\begin{table}[htbp]", "\\centering",
            f"\\caption{{Gold standard match distribution — {run_name}}}",
            f"\\label{{tab:gold_match:{run_name.lower().replace(' ','_')}}}",
            "\\begin{tabular}{lll}", "\\toprule",
            "Dimension & Breakdown & N \\\\", "\\midrule",
            "\\textit{By system}\\\\[2pt]",
        ]
        for sys, rng in SYSTEM_RANGES.items():
            nums = set(rng) & set(qual_data.keys())
            if not nums:
                continue
            sub = {n: qual_data[n] for n in nums}
            levels = _collect_match_levels(sub)
            lines.append(f"\\quad {SYSTEM_LABELS[sys]} & {_match_dist_str(levels)} & {len(levels)} \\\\")
        lines.append("\\midrule\\textit{By fault type}\\\\[2pt]")
        for ft in fault_types:
            nums = {n for n in qual_data if ft in scenario_meta(n)["fault_tags"]}
            if not nums:
                continue
            sub = {n: qual_data[n] for n in nums}
            levels = _collect_match_levels(sub)
            lines.append(f"\\quad {ft} & {_match_dist_str(levels)} & {len(levels)} \\\\")
        all_levels = _collect_match_levels(qual_data)
        lines += [
            "\\midrule",
            f"\\textbf{{Overall}} & \\textbf{{{_match_dist_str(all_levels)}}} & \\textbf{{{len(all_levels)}}} \\\\",
            "\\bottomrule", "\\end{tabular}", "\\end{table}",
        ]
        tables.append("\n".join(lines))

    return tables


# ---------------------------------------------------------------------------
# Qualitative run summary — reads file written by the protocol
# ---------------------------------------------------------------------------

def load_qualitative_summary(checkpoint_dir: Path, assistant: str, run_id: str) -> Optional[str]:
    """Read the prose summary written by the protocol after convergence."""
    p = checkpoint_dir / assistant / run_id / "qualitative_summary.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return None


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
        lines.append(f"  Scenarios: {len(data)}")
        mu, sd = agg(data, "success_rate")
        lines.append(f"  Success rate:  {mu*100:.1f}% ± {sd*100:.1f}%")
        mu, sd = agg(data, "total_cost")
        lines.append(f"  Avg cost:      {mu:.1f} ± {sd:.1f}")
        mu, sd = agg(data, "n_actions")
        lines.append(f"  Avg actions:   {mu:.2f} ± {sd:.2f}")
        converged = sum(1 for d in data.values() if d.get("reason") == "converged")
        lines.append(f"  Converged:     {converged}/{len(data)}")

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

    lines.append("\n--- Per-fault-type success rate ---")
    all_data = {num: row for _, d in runs for num, row in d.items()}
    fault_types = all_fault_types(all_data)
    header = f"{'Fault type':<22}" + "".join(f"{name:>{col_w}}" for name, _ in runs)
    lines += [header, "-" * len(header)]
    for ft in fault_types:
        row = f"{ft:<22}"
        for _, data in runs:
            vals = [r["success_rate"] for r in data.values() if ft in r["fault_tags"]]
            row += f"{(f'{statistics.mean(vals)*100:.1f}%') if vals else '--':>{col_w}}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation tables from adaptive protocol checkpoints."
    )
    parser.add_argument(
        "--run", action="append", metavar="ASSISTANT:RUN_ID",
        help="Run to include (repeatable). Omit to auto-detect latest per assistant.",
    )
    parser.add_argument(
        "--checkpoint-dir", default="Logs/Evaluation",
        help="Root directory for checkpoints (default: Logs/Evaluation).",
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

    all_latex: list[str] = []

    # --- Table 1: detailed per-run ---
    print("=" * 60)
    print("DETAILED PER-RUN TABLES")
    print("=" * 60)
    for assistant, run_id in specs:
        run_dir = ckpt / assistant / run_id
        print(f"\n>>> {assistant} / {run_id}")
        if not run_dir.exists():
            print(f"  Not found: {run_dir}")
            continue
        text, latex = detailed_run_table(run_dir, assistant, run_id)
        print(text)
        all_latex.append(f"% === Detailed: {assistant}/{run_id} ===\n{latex}")

    # --- Load report data ---
    runs: list[tuple[str, dict[int, dict]]] = []
    for assistant, run_id in specs:
        data = load_run_reports(ckpt, assistant, run_id)
        if data:
            runs.append((assistant, data))
        else:
            print(f"\n(No final_report.txt for {assistant}/{run_id} — skipping)")

    if not runs:
        _copy_latex("\n\n".join(all_latex))
        return

    print("\n")
    print(text_summary(runs))

    # --- Tables 2–4 ---
    t2 = per_scenario_table(runs)
    t3 = aggregate_table_by_system(runs)
    t4 = aggregate_table_by_fault_type(runs)

    print("\n% === Per-scenario comparison ===")
    print(t2)
    print("\n% === System aggregate ===")
    print(t3)
    print("\n% === Fault-type aggregate ===")
    print(t4)

    all_latex += [
        f"% === Per-scenario comparison ===\n{t2}",
        f"% === System aggregate ===\n{t3}",
        f"% === Fault-type aggregate ===\n{t4}",
    ]

    # --- Table 5: metric matrices ---
    matrices = metric_matrix_tables(runs)
    print("\n% === Metric breakdown matrices (system × fault type) ===")
    for m in matrices:
        print(m)
    all_latex += [f"% === Metric matrix ===\n{m}" for m in matrices]

    # --- Table 6: qualitative ---
    runs_qual: list[tuple[str, dict[int, dict]]] = []
    for assistant, run_id in specs:
        qual_data = load_qualitative(ckpt, assistant, run_id)
        if qual_data:
            runs_qual.append((assistant, qual_data))

    if runs_qual:
        print("\n")
        print(qualitative_aggregate_text(runs_qual, runs))
        qual_latex = qualitative_aggregate_latex(runs_qual, runs)
        print("\n% === Qualitative aggregate tables ===")
        for t in qual_latex:
            print(t)
        all_latex += [f"% === Qualitative aggregate ===\n" + "\n\n".join(qual_latex)]

        # --- Prose qualitative summary (written by protocol, read here) ---
        print("\n" + "=" * 60)
        print("QUALITATIVE PROSE SUMMARY")
        print("=" * 60)
        for assistant, run_id in specs:
            summary = load_qualitative_summary(ckpt, assistant, run_id)
            if summary:
                print(f"\n--- {assistant} / {run_id} ---")
                print(summary)
                all_latex.append(
                    f"% === Qualitative prose summary: {assistant}/{run_id} ===\n"
                    f"% {summary.replace(chr(10), chr(10) + '% ')}"
                )
            else:
                print(f"\n--- {assistant} / {run_id} ---")
                print("  (no qualitative_summary.txt — re-run protocol with --resume to generate)")
    else:
        print("\n(No qualitative.json files found — run with --resume to regenerate)")

    _copy_latex("\n\n".join(all_latex))


def _copy_latex(text: str) -> None:
    try:
        import pyperclip
        pyperclip.copy(text)
        print("\n(All LaTeX copied to clipboard.)")
    except Exception:
        pass


if __name__ == "__main__":
    main()
