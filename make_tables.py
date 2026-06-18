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
  6. Convergence report           — convergence status per scenario
  7. Qualitative aggregate        — rubric rating distributions + gold match levels
                                    broken down by system and fault type
  8. Heatmaps                     — PNG heatmaps for matrix tables, saved to Images/
  9. Text summary
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
    "Coupled", "Limited Observability", "Unforeseen Interaction", "Triple",
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
# Loading — config.json (to extract model name for assistant label)
# ---------------------------------------------------------------------------

def load_run_config(checkpoint_dir: Path, assistant: str, run_id: str) -> dict:
    p = checkpoint_dir / assistant / run_id / "config.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def assistant_label(assistant: str, run_id: str, checkpoint_dir: Path) -> str:
    """Return a label like 'LLM (gpt-4.1-mini)' by reading config.json."""
    cfg = load_run_config(checkpoint_dir, assistant, run_id)
    model = cfg.get("assistant_config", {}).get("model", "")
    if model:
        # Strip provider prefix (openai/, google/, etc.)
        short = model.split("/")[-1]
        return f"{assistant} ({short})"
    return assistant


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
        return "--"
    pt, lo, hi = m.get("point"), m.get("ci_lo"), m.get("ci_hi")
    if pt is None:
        return "--"
    if pct:
        return f"{pt:.1%} [{lo:.1%}, {hi:.1%}]"
    return f"{pt:.2f} [{lo:.2f}, {hi:.2f}]"


def _fmt_ari(ari: float | None, floor: float | None) -> str:
    ari_s   = f"{ari:.3f}"   if ari   is not None else "--"
    floor_s = f"{floor:.3f}" if floor is not None else "--"
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
    label = f"tab:eval:{assistant_label.lower().replace(' ','_').replace('(','').replace(')','').replace('/','_')}:{run_id}"

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
# Main results table  (system × model, success/actions/cost, best bolded)
# ---------------------------------------------------------------------------

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score confidence interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def main_results_table(
    runs: list[tuple[str, dict[int, dict]]],
    checkpoint_dir: Path,
    specs: list[tuple[str, str]],
) -> str:
    """
    Main results table for the paper.
    Rows: one per system + Overall.
    Columns: for each model — Success% [CI], Mean actions, Mean cost.
    Best value per row bolded.
    Success rate CI computed from raw trajectory counts via Wilson interval.

    Also loads raw trajectory end values from checkpoint.json for accurate
    success counts (rather than relying on final_report.txt point estimates).
    """
    import math

    # Load raw trajectory-level data per (assistant, run_id) for Wilson CIs
    raw_counts: dict[str, dict[int, dict]] = {}  # label -> scenario -> {n, n_succ}
    for assistant, run_id in specs:
        run_dir = checkpoint_dir / assistant / run_id
        label = next(
            (name for name, _ in runs
             if name == assistant_label(assistant, run_id, checkpoint_dir)),
            assistant,
        )
        sc_counts: dict[int, dict] = {}
        for ckpt_path in sorted(run_dir.glob("*/checkpoint.json"),
                                key=lambda p: int(p.parent.name)):
            try:
                ckpt = json.loads(ckpt_path.read_text())
            except Exception:
                continue
            snum = ckpt.get("scenario_number")
            if snum is None:
                continue
            n_total = 0
            n_succ = 0
            for tpath in ckpt.get("trajectory_paths", []):
                try:
                    t = json.loads(Path(tpath).read_text())
                except Exception:
                    continue
                end = t.get("end", "")
                if end == "llm_truncation":
                    continue
                n_total += 1
                if end in ("success", "success_no_hypothesis"):
                    n_succ += 1
            sc_counts[snum] = {"n": n_total, "n_succ": n_succ}
        raw_counts[label] = sc_counts

    n_runs = len(runs)
    col_spec = "l" + "rrr" * n_runs
    mid_cols = "".join(f"\\cmidrule(lr){{{2 + i*3}-{4 + i*3}}}" for i in range(n_runs))
    hdr_names = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\textbf{{{name}}}}}" for name, _ in runs
    )
    hdr_cols = " & ".join("Succ (95\\%~CI) & Acts & Cost" for _ in runs)

    lines = [
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Main results: success rate (Wilson 95\\% CI), mean actions, and mean cost "
        "per system and model. Best value per row in bold.}",
        "\\label{tab:main_results}", "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
        f" & {hdr_names} \\\\",
        mid_cols,
        f"\\textbf{{System}} & {hdr_cols} \\\\",
        "\\midrule",
    ]

    systems_iter = list(SYSTEM_RANGES.items()) + [("_overall", None)]

    for sys, rng in systems_iter:
        is_overall = sys == "_overall"
        sys_label = "\\textbf{Overall}" if is_overall else SYSTEM_LABELS[sys]
        if is_overall:
            lines.append("\\midrule")
        nums = None if is_overall else list(rng)

        # Collect per-run values
        row_succs: list[str] = []
        row_acts: list[float] = []
        row_costs: list[float] = []

        for name, data in runs:
            if is_overall:
                sc_list = list(data.keys())
            else:
                sc_list = [n for n in nums if n in data]

            # Success rate via Wilson
            counts = raw_counts.get(name, {})
            n_total = sum(counts.get(s, {}).get("n", 0) for s in sc_list)
            n_succ  = sum(counts.get(s, {}).get("n_succ", 0) for s in sc_list)
            p, lo, hi = _wilson_ci(n_succ, n_total)

            if math.isnan(p):
                row_succs.append("--")
            else:
                row_succs.append(f"{p*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]")

            acts  = [data[s]["n_actions"]  for s in sc_list if s in data and data[s]["n_actions"]  == data[s]["n_actions"]]
            costs = [data[s]["total_cost"] for s in sc_list if s in data and data[s]["total_cost"] == data[s]["total_cost"]]
            row_acts.append(statistics.mean(acts)   if acts  else float("nan"))
            row_costs.append(statistics.mean(costs) if costs else float("nan"))

        # Bold best values
        valid_acts  = [v for v in row_acts  if not math.isnan(v)]
        valid_costs = [v for v in row_costs if not math.isnan(v)]
        best_act  = min(valid_acts)  if valid_acts  else None
        best_cost = min(valid_costs) if valid_costs else None

        # For success, best = highest p (parse from string)
        succ_vals = []
        for s in row_succs:
            try:
                succ_vals.append(float(s.split()[0]))
            except Exception:
                succ_vals.append(float("nan"))
        valid_succ = [v for v in succ_vals if not math.isnan(v)]
        best_succ = max(valid_succ) if valid_succ else None

        cells = []
        for i in range(n_runs):
            s_str = row_succs[i]
            a_val = row_acts[i]
            c_val = row_costs[i]

            # Bold success if best
            if best_succ is not None and succ_vals[i] == best_succ:
                s_cell = f"\\textbf{{{s_str}}}\\%"
            else:
                s_cell = f"{s_str}\\%" if s_str != "--" else "--"

            a_cell = f"{a_val:.2f}" if not math.isnan(a_val) else "--"
            c_cell = f"{c_val:.0f}" if not math.isnan(c_val) else "--"

            if best_act is not None and a_val == best_act:
                a_cell = f"\\textbf{{{a_cell}}}"
            if best_cost is not None and c_val == best_cost:
                c_cell = f"\\textbf{{{c_cell}}}"

            cells.append(f"{s_cell} & {a_cell} & {c_cell}")

        lines.append(f"{sys_label} & " + " & ".join(cells) + " \\\\")

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
                f"\\caption{{{metric_label} by system and fault type -- {run_name}}}",
                f"\\label{{tab:matrix:{run_name.lower().replace(' ','_').replace('(','').replace(')','').replace('/','_')}:{metric_key}}}",
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
# Table 6: convergence report
# ---------------------------------------------------------------------------

def convergence_report_latex(
    specs: list[tuple[str, str]],
    checkpoint_dir: Path,
    run_labels: dict[tuple[str,str], str],
) -> str:
    """LaTeX table: one row per scenario, columns = converged/stopped/batches per run."""
    all_checkpoints: dict[tuple[str,str], dict[int, dict]] = {}
    all_scenarios: set[int] = set()
    for assistant, run_id in specs:
        run_dir = checkpoint_dir / assistant / run_id
        ckpts = load_checkpoints(run_dir)
        mapping = {c["scenario_number"]: c for c in ckpts}
        all_checkpoints[(assistant, run_id)] = mapping
        all_scenarios.update(mapping.keys())

    n = len(specs)
    col_spec = "rll" + "cccc" * n
    mid_cols = "".join(f"\\cmidrule(lr){{{4 + i*4}-{7 + i*4}}}" for i in range(n))
    hdr_names = " & ".join(
        f"\\multicolumn{{4}}{{c}}{{{run_labels[(a,r)].replace('_','-')}}}"
        for a, r in specs
    )
    hdr_cols = " & ".join("Conv. & Stop & Batches & Trunc" for _ in specs)

    lines = [
        "\\begin{table}[htbp]", "\\centering",
        "\\caption{Convergence report per scenario}",
        "\\label{tab:convergence}", "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}", "\\toprule",
        f"\\multicolumn{{3}}{{c}}{{Scenario}} & {hdr_names} \\\\",
        f"\\cmidrule(lr){{1-3}}{mid_cols}",
        f"\\# & System & Fault type & {hdr_cols} \\\\",
        "\\midrule",
    ]

    for snum in sorted(all_scenarios):
        meta = scenario_meta(snum)
        sys_label   = SYSTEM_LABELS.get(meta["system"], meta["system"])
        fault_label = ", ".join(meta["fault_tags"])
        cells_parts = []
        for assistant, run_id in specs:
            ckpt = all_checkpoints[(assistant, run_id)].get(snum)
            if ckpt is None:
                cells_parts.append("-- & -- & -- & --")
            else:
                conv  = "\\checkmark" if ckpt.get("converged") else ""
                stop  = "\\checkmark" if ckpt.get("stopped") else ""
                nb    = ckpt.get("batch_index", "?")
                nt    = ckpt.get("n_truncations", 0)
                trunc = str(nt) if nt else ""
                cells_parts.append(f"{conv} & {stop} & {nb} & {trunc}")
        lines.append(f"{snum} & {sys_label} & {fault_label} & " + " & ".join(cells_parts) + " \\\\")

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heatmaps (matplotlib) — saved to Images/, included in LaTeX
# ---------------------------------------------------------------------------

def _raw_matrix(
    data: dict[int, dict],
    fault_types: list[str],
    metric_key: str,
    use_std: bool,
) -> tuple[list[list[float]], list[str], list[str]]:
    """Return (matrix, row_labels, col_labels) with NaN for missing cells."""
    import math
    systems = list(SYSTEM_RANGES.keys())
    row_labels = [SYSTEM_LABELS[s] for s in systems]
    col_labels = fault_types
    matrix: list[list[float]] = []
    for sys in systems:
        nums = list(SYSTEM_RANGES[sys])
        row = []
        for ft in fault_types:
            vals = [
                data[n][metric_key] for n in nums
                if n in data and ft in data[n]["fault_tags"] and data[n][metric_key] == data[n][metric_key]
            ]
            if not vals:
                row.append(float("nan"))
            elif use_std:
                row.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
            else:
                row.append(statistics.mean(vals))
        matrix.append(row)
    return matrix, row_labels, col_labels


def generate_heatmaps(
    runs: list[tuple[str, dict[int, dict]]],
    checkpoint_dir: Path,
    images_dir: Path,
) -> list[tuple[str, str]]:
    """
    Generate heatmap PNGs for each metric matrix.
    Returns list of (latex_figure_code, description) for inclusion.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import math
    except ImportError:
        print("(matplotlib not available — skipping heatmaps)")
        return []

    images_dir.mkdir(parents=True, exist_ok=True)
    all_data = {num: row for _, d in runs for num, row in d.items()}
    fault_types = all_fault_types(all_data)

    metrics = [
        ("success_rate", "Success Rate", True),
        ("total_cost",   "Total Cost",   False),
        ("n_actions",    "Actions",      False),
    ]

    latex_blocks: list[tuple[str, str]] = []

    for run_name, data in runs:
        safe_run = run_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        for metric_key, metric_label, is_pct in metrics:
            fig_paths = []
            for use_std, stat_label, stat_short in [(False, "Mean", "mean"), (True, "Std", "std")]:
                matrix, row_labels, col_labels = _raw_matrix(data, fault_types, metric_key, use_std)

                # Normalise for colour
                flat = [v for row in matrix for v in row if not math.isnan(v)]
                vmin = min(flat) if flat else 0.0
                vmax = max(flat) if flat else 1.0
                if vmax == vmin:
                    vmax = vmin + 1e-6

                fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 0.9), max(2.5, len(row_labels) * 0.6)))
                import numpy as np
                arr = np.array(matrix, dtype=float)
                cmap = "YlOrRd" if not use_std else "Blues"
                im = ax.imshow(arr, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7)
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=7)
                ax.set_title(f"{metric_label} {stat_label} — {run_name}", fontsize=8, pad=4)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                # Annotate cells
                for r, row in enumerate(matrix):
                    for c, val in enumerate(row):
                        if not math.isnan(val):
                            txt = f"{val*100:.0f}%" if is_pct and not use_std else f"{val:.1f}"
                            ax.text(c, r, txt, ha="center", va="center",
                                    fontsize=6, color="black" if val < (vmin + (vmax - vmin) * 0.6) else "white")

                plt.tight_layout()
                fname = f"heatmap_{safe_run}_{metric_key}_{stat_short}.pdf"
                fpath = images_dir / fname
                fig.savefig(fpath, bbox_inches="tight")
                plt.close(fig)
                fig_paths.append((fpath.name, f"{metric_label} {stat_label}"))

            # Emit one figure with two subfigures side by side
            slug = f"fig:heatmap:{safe_run}:{metric_key}"
            latex = (
                "\\begin{figure}[htbp]\n\\centering\n"
                + "".join(
                    f"  \\includegraphics[width=0.48\\linewidth]{{Images/{fname}}}\n"
                    for fname, _ in fig_paths
                )
                + f"\\caption{{{metric_label} heatmaps (mean left, std right) -- {run_name}}}\n"
                f"\\label{{{slug}}}\n"
                "\\end{figure}\n"
            )
            latex_blocks.append((latex, f"Heatmap: {run_name} / {metric_label}"))

    # --- Diff heatmaps: each non-baseline model vs. baseline ---
    if len(runs) >= 2:
        baseline_name, baseline_data = runs[0]
        safe_base = baseline_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

        for run_name, data in runs[1:]:
            safe_run = run_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            for metric_key, metric_label, is_pct in metrics:
                base_matrix, row_labels, col_labels = _raw_matrix(baseline_data, fault_types, metric_key, use_std=False)
                comp_matrix, _, _                   = _raw_matrix(data,          fault_types, metric_key, use_std=False)

                import numpy as np
                base_arr = np.array(base_matrix, dtype=float)
                comp_arr = np.array(comp_matrix, dtype=float)
                diff_arr = comp_arr - base_arr  # positive = model better than baseline

                flat = [v for v in diff_arr.flatten() if not math.isnan(v)]
                if not flat:
                    continue
                abs_max = max(abs(min(flat)), abs(max(flat)), 1e-6)

                fig, ax = plt.subplots(figsize=(max(4, len(col_labels) * 0.9), max(2.5, len(row_labels) * 0.6)))
                im = ax.imshow(diff_arr, aspect="auto",
                               vmin=-abs_max, vmax=abs_max, cmap="RdYlGn")
                ax.set_xticks(range(len(col_labels)))
                ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=7)
                ax.set_yticks(range(len(row_labels)))
                ax.set_yticklabels(row_labels, fontsize=7)
                ax.set_title(f"{metric_label} diff: {run_name} − {baseline_name}", fontsize=8, pad=4)
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Δ (green = model better)", fontsize=6)

                for r, row in enumerate(diff_arr):
                    for c, val in enumerate(row):
                        if not math.isnan(val):
                            txt = f"{val*100:+.0f}%" if is_pct else f"{val:+.1f}"
                            ax.text(c, r, txt, ha="center", va="center",
                                    fontsize=6, color="black")

                plt.tight_layout()
                fname = f"heatmap_diff_{safe_run}_vs_{safe_base}_{metric_key}.pdf"
                fpath = images_dir / fname
                fig.savefig(fpath, bbox_inches="tight")
                plt.close(fig)

                slug  = f"fig:heatmap:diff:{safe_run}_vs_{safe_base}:{metric_key}"
                latex = (
                    "\\begin{figure}[htbp]\n\\centering\n"
                    f"  \\includegraphics[width=0.65\\linewidth]{{Images/{fname}}}\n"
                    f"\\caption{{\\textbf{{Diff heatmap}}: {metric_label} — {run_name} minus {baseline_name}. "
                    "Green = model outperforms baseline; red = baseline outperforms model.}}\n"
                    f"\\label{{{slug}}}\n"
                    "\\end{figure}\n"
                )
                latex_blocks.append((latex, f"Diff heatmap: {run_name} vs {baseline_name} / {metric_label}"))

    return latex_blocks


# ---------------------------------------------------------------------------
# Cost vs. success scatter plot
# ---------------------------------------------------------------------------

# Okabe-Ito palette — colour-blind safe, one colour per model run
_OI_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def generate_cost_vs_success_scatter(
    runs: list[tuple[str, dict[int, dict]]],
    images_dir: Path,
) -> tuple[str, str]:
    """
    One point per (model, system).
    X = mean total_cost, Y = mean success_rate.
    Error bars: std across scenarios within that system.
    Each model gets a distinct colour + marker shape.
    Systems are distinguished by text labels on the points.

    Returns (latex_figure_code, description).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        import numpy as np
        import statistics as _st
    except ImportError:
        return "", ""

    images_dir.mkdir(parents=True, exist_ok=True)

    sys_short = {
        "3_cubes":              "3C",
        "10_cubes":             "10C",
        "asymmetric_chains":    "AC",
        "ambient_light_sensor": "ALS",
        "current_sensor":       "CS",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    legend_handles = []
    for run_idx, (run_name, data) in enumerate(runs):
        color  = _OI_COLORS[run_idx % len(_OI_COLORS)]
        marker = _MARKERS[run_idx % len(_MARKERS)]

        for sys, rng in SYSTEM_RANGES.items():
            nums = [n for n in rng if n in data]
            if not nums:
                continue
            costs   = [data[n]["total_cost"]   for n in nums if data[n]["total_cost"]   == data[n]["total_cost"]]
            succs   = [data[n]["success_rate"]  for n in nums if data[n]["success_rate"] == data[n]["success_rate"]]
            if not costs or not succs:
                continue
            x  = _st.mean(costs)
            y  = _st.mean(succs)
            xe = _st.stdev(costs)  if len(costs) > 1 else 0.0
            ye = _st.stdev(succs)  if len(succs) > 1 else 0.0

            ax.errorbar(x, y, xerr=xe, yerr=ye,
                        fmt=marker, color=color, markersize=7,
                        elinewidth=0.8, capsize=3, alpha=0.85, zorder=3)
            ax.annotate(sys_short[sys], (x, y),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=color, alpha=0.9)

        handle = mlines.Line2D([], [], color=color, marker=marker,
                               linestyle="None", markersize=7, label=run_name)
        legend_handles.append(handle)

    ax.set_xlabel("Mean total cost (technician-seconds)", fontsize=9)
    ax.set_ylabel("Mean success rate", fontsize=9)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Cost vs. success rate per model and system\n(error bars = std across scenarios)", fontsize=9)
    ax.legend(handles=legend_handles, fontsize=8, loc="best")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    plt.tight_layout()

    fname = "scatter_cost_vs_success.pdf"
    fpath = images_dir / fname
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)

    latex = (
        "\\begin{figure}[htbp]\n\\centering\n"
        f"  \\includegraphics[width=0.75\\linewidth]{{Images/{fname}}}\n"
        "\\caption{Cost vs.\\ success rate per model and system. "
        "Each point is one (model, system) pair; error bars show standard deviation "
        "across scenarios. Labels: 3C=3-Cubes, 10C=10-Cubes, AC=Asym.\\ Chains, "
        "ALS=Ambient Light Sensor, CS=Current Sensor.}\n"
        "\\label{fig:scatter:cost_vs_success}\n"
        "\\end{figure}\n"
    )
    return latex, "Scatter: cost vs. success rate"


# ---------------------------------------------------------------------------
# Radar charts — rubric ratings per fault type
# ---------------------------------------------------------------------------

_RATING_NUM = {"low": 1, "medium": 2, "high": 3}


def generate_rubric_radar_charts(
    runs_qual: list[tuple[str, dict[int, dict]]],
    runs_data: list[tuple[str, dict[int, dict]]],
    images_dir: Path,
) -> list[tuple[str, str]]:
    """
    One radar chart per fault type per run.
    Each chart has 5 axes (rubric dims), 3 reference circles (L=1/M=2/H=3),
    and one polygon whose vertices are the mean rating (1–3) per dimension.
    Returns list of (latex_figure_code, description).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("(matplotlib not available -- skipping radar charts)")
        return []

    images_dir.mkdir(parents=True, exist_ok=True)
    all_data = {num: row for _, d in runs_data for num, row in d.items()}
    fault_types = all_fault_types(all_data)

    dim_labels = [RUBRIC_DIM_LABELS[d] for d in RUBRIC_DIMS]
    N = len(RUBRIC_DIMS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    latex_blocks: list[tuple[str, str]] = []

    for run_name, qual_data in runs_qual:
        safe_run = run_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

        # Collect per-fault-type mean ratings
        ft_means: dict[str, list[float]] = {}
        for ft in fault_types:
            nums = {n for n in qual_data if ft in scenario_meta(n)["fault_tags"]}
            if not nums:
                continue
            dim_vals: dict[str, list[float]] = {d: [] for d in RUBRIC_DIMS}
            for n in nums:
                for rs in qual_data[n].get("rubric_scores", []):
                    for dim in RUBRIC_DIMS:
                        dv = rs.get(dim)
                        if isinstance(dv, dict):
                            rating = dv.get("rating", "").lower().strip()
                            num_val = _RATING_NUM.get(rating)
                            if num_val is not None:
                                dim_vals[dim].append(float(num_val))
            means = [
                (sum(dim_vals[d]) / len(dim_vals[d])) if dim_vals[d] else float("nan")
                for d in RUBRIC_DIMS
            ]
            ft_means[ft] = means

        if not ft_means:
            continue

        # One subplot per fault type, arranged in a grid
        n_fts = len(ft_means)
        ncols = min(3, n_fts)
        nrows = (n_fts + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 2.8, nrows * 2.8),
            subplot_kw={"projection": "polar"},
        )
        if n_fts == 1:
            axes = np.array([[axes]])
        elif nrows == 1:
            axes = axes.reshape(1, -1)
        elif ncols == 1:
            axes = axes.reshape(-1, 1)

        # Color cycle
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        for idx, (ft, means) in enumerate(ft_means.items()):
            ax = axes[idx // ncols][idx % ncols]

            # Reference circles at 1, 2, 3
            for level, ls, lc in [(1, ":", "#bbbbbb"), (2, "--", "#999999"), (3, "-", "#666666")]:
                circle_vals = [level] * N + [level]
                ax.plot(angles, circle_vals, linestyle=ls, color=lc, linewidth=0.7, zorder=1)
                ax.fill(angles, circle_vals, alpha=0.0)

            # Data polygon
            vals = means + means[:1]
            color = colors[idx % len(colors)]
            ax.plot(angles, vals, color=color, linewidth=1.8, zorder=3)
            ax.fill(angles, vals, alpha=0.18, color=color, zorder=2)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dim_labels, size=6)
            ax.set_yticks([1, 2, 3])
            ax.set_yticklabels(["L", "M", "H"], size=6)
            ax.set_ylim(0, 3.3)
            ax.set_title(ft, size=7, pad=6, fontweight="bold")
            ax.tick_params(axis="y", labelsize=5)
            # Keep circles, remove spokes and outer spine
            ax.xaxis.grid(False)
            ax.yaxis.grid(True, color="#bbbbbb", linewidth=0.6)
            ax.spines["polar"].set_visible(False)

        # Hide empty subplots
        for idx in range(n_fts, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f"Rubric ratings by fault type -- {run_name}", size=8, y=1.01)
        plt.tight_layout()
        fname = f"radar_{safe_run}_rubric_by_fault_type.pdf"
        fpath = images_dir / fname
        fig.savefig(fpath, bbox_inches="tight")
        plt.close(fig)

        latex = (
            "\\begin{figure}[htbp]\n\\centering\n"
            f"  \\includegraphics[width=\\linewidth]{{Images/{fname}}}\n"
            f"\\caption{{Rubric ratings radar chart by fault type -- {run_name}. "
            "Axes: actionability, diagnostic coherence, efficiency, consistency, evidence usage. "
            "Reference circles: L=1, M=2, H=3.}}\n"
            f"\\label{{fig:radar:{safe_run}:rubric_fault}}\n"
            "\\end{figure}\n"
        )
        latex_blocks.append((latex, f"Radar: {run_name} / rubric by fault type"))

    return latex_blocks


def _run_overall_means(qual_data: dict[int, dict]) -> list[float]:
    """Return mean rating (1-3) per rubric dim across all scenarios."""
    dim_vals: dict[str, list[float]] = {d: [] for d in RUBRIC_DIMS}
    for q in qual_data.values():
        for rs in q.get("rubric_scores", []):
            for dim in RUBRIC_DIMS:
                dv = rs.get(dim)
                if isinstance(dv, dict):
                    v = _RATING_NUM.get(dv.get("rating", "").lower().strip())
                    if v is not None:
                        dim_vals[dim].append(float(v))
    return [
        (sum(dim_vals[d]) / len(dim_vals[d])) if dim_vals[d] else float("nan")
        for d in RUBRIC_DIMS
    ]


def generate_overlay_radar_chart(
    runs_qual: list[tuple[str, dict[int, dict]]],
    images_dir: Path,
) -> tuple[str, str] | None:
    """
    Single polar chart with one polygon per model, aggregated over all scenarios.
    Returns (latex_figure_code, description) or None if no data / matplotlib missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return None

    if not runs_qual:
        return None

    images_dir.mkdir(parents=True, exist_ok=True)
    dim_labels = [RUBRIC_DIM_LABELS[d] for d in RUBRIC_DIMS]
    N = len(RUBRIC_DIMS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw={"projection": "polar"})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for level, ls, lc in [(1, ":", "#cccccc"), (2, "--", "#aaaaaa"), (3, "-", "#888888")]:
        ax.plot(angles, [level] * N + [level], linestyle=ls, color=lc, linewidth=0.7, zorder=1)

    for idx, (run_name, qual_data) in enumerate(runs_qual):
        means = _run_overall_means(qual_data)
        if all(np.isnan(means)):
            continue
        vals = means + means[:1]
        color = colors[idx % len(colors)]
        ax.plot(angles, vals, color=color, linewidth=1.8, label=run_name, zorder=3)
        ax.fill(angles, vals, alpha=0.12, color=color, zorder=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, size=7)
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(["L", "M", "H"], size=6)
    ax.set_ylim(0, 3.3)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color="#bbbbbb", linewidth=0.6)
    ax.spines["polar"].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=7)
    fig.suptitle("Rubric ratings — all scenarios", size=9)
    plt.tight_layout()

    fname = "radar_overlay_all_scenarios.pdf"
    fig.savefig(images_dir / fname, bbox_inches="tight")
    plt.close(fig)

    latex = (
        "\\begin{figure}[htbp]\n\\centering\n"
        f"  \\includegraphics[width=0.6\\linewidth]{{Images/{fname}}}\n"
        "\\caption{Rubric ratings overlay radar chart across all scenarios. "
        "Each polygon is one model; axes are the five rubric dimensions; "
        "reference circles: L=1, M=2, H=3.}\n"
        "\\label{fig:radar:overlay:all}\n"
        "\\end{figure}\n"
    )
    return latex, "Overlay radar: all models / all scenarios"


def generate_delta_radar_charts(
    runs_qual: list[tuple[str, dict[int, dict]]],
    images_dir: Path,
) -> list[tuple[str, str]]:
    """
    For every pair of runs (A, B), produce a delta radar: A - B per dimension.
    Positive = A better, negative = B better. Zero circle drawn for reference.
    Returns list of (latex_figure_code, description).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    if len(runs_qual) < 2:
        return []

    images_dir.mkdir(parents=True, exist_ok=True)
    dim_labels = [RUBRIC_DIM_LABELS[d] for d in RUBRIC_DIMS]
    N = len(RUBRIC_DIMS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    latex_blocks: list[tuple[str, str]] = []
    pairs = [(i, j) for i in range(len(runs_qual)) for j in range(i + 1, len(runs_qual))]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, j in pairs:
        name_a, qual_a = runs_qual[i]
        name_b, qual_b = runs_qual[j]
        means_a = np.array(_run_overall_means(qual_a))
        means_b = np.array(_run_overall_means(qual_b))
        delta = means_a - means_b  # positive = A better

        fig, ax = plt.subplots(figsize=(4.0, 4.0), subplot_kw={"projection": "polar"})

        # Zero reference circle
        ax.plot(angles, [0.0] * N + [0.0], linestyle="-", color="#888888", linewidth=0.9, zorder=1)
        # ±1 guides
        for level, ls in [(1.0, "--"), (-1.0, "--")]:
            ax.plot(angles, [level] * N + [level], linestyle=ls, color="#cccccc", linewidth=0.6, zorder=1)

        vals = delta.tolist() + [delta[0]]
        color = colors[0]
        ax.plot(angles, vals, color=color, linewidth=1.8, zorder=3)
        ax.fill(angles, vals, alpha=0.18, color=color, zorder=2)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dim_labels, size=7)
        ax.set_ylim(-2, 2)
        ax.set_yticks([-1, 0, 1])
        ax.set_yticklabels(["-1", "0", "+1"], size=6)
        ax.xaxis.grid(False)
        ax.yaxis.grid(True, color="#bbbbbb", linewidth=0.6)
        ax.spines["polar"].set_visible(False)
        ax.set_title(f"{name_a} − {name_b}", size=7, pad=8, fontweight="bold")
        plt.tight_layout()

        safe_a = name_a.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        safe_b = name_b.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        fname = f"radar_delta_{safe_a}_vs_{safe_b}.pdf"
        fig.savefig(images_dir / fname, bbox_inches="tight")
        plt.close(fig)

        latex = (
            "\\begin{figure}[htbp]\n\\centering\n"
            f"  \\includegraphics[width=0.55\\linewidth]{{Images/{fname}}}\n"
            f"\\caption{{Delta radar chart: {name_a} $-$ {name_b}. "
            "Positive values indicate {name_a} scores higher on that dimension; "
            "negative values indicate {name_b} scores higher. "
            "Reference circle at 0; dashed guides at $\\pm1$.}}\n"
            f"\\label{{fig:radar:delta:{safe_a}:{safe_b}}}\n"
            "\\end{figure}\n"
        )
        latex_blocks.append((latex, f"Delta radar: {name_a} vs {name_b}"))

    return latex_blocks


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

        per_dim = _collect_rubric_ratings(qual_data)
        lines.append("\nRubric rating distribution (H:high M:medium L:low):")
        for dim in RUBRIC_DIMS:
            ratings = per_dim[dim]
            lines.append(f"  {RUBRIC_DIM_LABELS[dim]:<22} {_rating_dist_str(ratings)}")

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

    for run_name, qual_data in runs_qual:
        per_dim_global = _collect_rubric_ratings(qual_data)
        dim_labels     = [RUBRIC_DIM_LABELS[d] for d in RUBRIC_DIMS]
        safe = run_name.lower().replace(" ", "_").replace("(","").replace(")","").replace("/","_")

        col_spec = "l" + "r" * len(RUBRIC_DIMS)
        lines = [
            "\\begin{table}[htbp]", "\\centering",
            f"\\caption{{Rubric rating distribution by system -- {run_name} (H/M/L counts)}}",
            f"\\label{{tab:rubric_system:{safe}}}",
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

        lines = [
            "\\begin{table}[htbp]", "\\centering",
            f"\\caption{{Rubric rating distribution by fault type -- {run_name}}}",
            f"\\label{{tab:rubric_fault:{safe}}}",
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

        lines = [
            "\\begin{table}[htbp]", "\\centering",
            f"\\caption{{Gold standard match distribution -- {run_name}}}",
            f"\\label{{tab:gold_match:{safe}}}",
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
        lines.append(f"  Success rate:  {mu*100:.1f}% +/- {sd*100:.1f}%")
        mu, sd = agg(data, "total_cost")
        lines.append(f"  Avg cost:      {mu:.1f} +/- {sd:.1f}")
        mu, sd = agg(data, "n_actions")
        lines.append(f"  Avg actions:   {mu:.2f} +/- {sd:.2f}")
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
    parser.add_argument(
        "--images-dir", default="Images",
        help="Directory to save heatmap images (default: Images/).",
    )
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    images_dir = Path(args.images_dir)

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

    # Build assistant labels (with model name) for all specs
    run_labels = {(a, r): assistant_label(a, r, ckpt) for a, r in specs}

    all_latex: list[str] = []

    # --- Table 1: detailed per-run ---
    print("=" * 60)
    print("DETAILED PER-RUN TABLES")
    print("=" * 60)
    for assistant, run_id in specs:
        run_dir = ckpt / assistant / run_id
        label = run_labels[(assistant, run_id)]
        print(f"\n>>> {label} / {run_id}")
        if not run_dir.exists():
            print(f"  Not found: {run_dir}")
            continue
        text, latex = detailed_run_table(run_dir, label, run_id)
        print(text)
        all_latex.append(f"% === Detailed: {label}/{run_id} ===\n{latex}")

    # --- Load report data (use labelled names) ---
    runs: list[tuple[str, dict[int, dict]]] = []
    for assistant, run_id in specs:
        data = load_run_reports(ckpt, assistant, run_id)
        label = run_labels[(assistant, run_id)]
        if data:
            runs.append((label, data))
        else:
            print(f"\n(No final_report.txt for {assistant}/{run_id} -- skipping)")

    if not runs:
        _copy_latex("\n\n".join(all_latex))
        return

    print("\n")
    print(text_summary(runs))

    # --- Main results table ---
    t_main = main_results_table(runs, ckpt, specs)
    print("\n% === MAIN RESULTS TABLE ===")
    print(t_main)
    all_latex.insert(0, f"% === MAIN RESULTS TABLE ===\n{t_main}")

    # --- Tables 2-4 ---
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
    print("\n% === Metric breakdown matrices (system x fault type) ===")
    for m in matrices:
        print(m)
    all_latex += [f"% === Metric matrix ===\n{m}" for m in matrices]

    # --- Heatmaps ---
    heatmap_blocks = generate_heatmaps(runs, ckpt, images_dir)
    if heatmap_blocks:
        print(f"\n% === Heatmaps (saved to {images_dir}/) ===")
        heatmap_latex_parts = []
        for latex, desc in heatmap_blocks:
            print(f"% {desc}")
            print(latex)
            heatmap_latex_parts.append(latex)
        all_latex.append("% === Heatmaps ===\n" + "\n".join(heatmap_latex_parts))

    # --- Scatter: cost vs. success ---
    scatter_latex, scatter_desc = generate_cost_vs_success_scatter(runs, images_dir)
    if scatter_latex:
        print(f"\n% === {scatter_desc} (saved to {images_dir}/) ===")
        print(scatter_latex)
        all_latex.append(f"% === {scatter_desc} ===\n{scatter_latex}")

    # --- Table 6: convergence report ---
    conv_table = convergence_report_latex(specs, ckpt, run_labels)
    print("\n% === Convergence report ===")
    print(conv_table)
    all_latex.append(f"% === Convergence report ===\n{conv_table}")

    # --- Table 7: qualitative ---
    runs_qual: list[tuple[str, dict[int, dict]]] = []
    for assistant, run_id in specs:
        qual_data = load_qualitative(ckpt, assistant, run_id)
        label = run_labels[(assistant, run_id)]
        if qual_data:
            runs_qual.append((label, qual_data))

    if runs_qual:
        print("\n")
        print(qualitative_aggregate_text(runs_qual, runs))
        qual_latex = qualitative_aggregate_latex(runs_qual, runs)
        print("\n% === Qualitative aggregate tables ===")
        for t in qual_latex:
            print(t)
        all_latex += [f"% === Qualitative aggregate ===\n" + "\n\n".join(qual_latex)]

        # --- Radar charts (per fault type, per run) ---
        radar_blocks = generate_rubric_radar_charts(runs_qual, runs, images_dir)
        if radar_blocks:
            print(f"\n% === Rubric radar charts (saved to {images_dir}/) ===")
            radar_latex_parts = []
            for latex, desc in radar_blocks:
                print(f"% {desc}")
                print(latex)
                radar_latex_parts.append(latex)
            all_latex.append("% === Radar charts ===\n" + "\n".join(radar_latex_parts))

        # --- Overlay radar (all models, all scenarios) ---
        overlay = generate_overlay_radar_chart(runs_qual, images_dir)
        if overlay:
            latex, desc = overlay
            print(f"\n% === {desc} ===")
            print(latex)
            all_latex.append("% === Overlay radar ===\n" + latex)

        # --- Delta radar (pairwise model comparisons) ---
        delta_blocks = generate_delta_radar_charts(runs_qual, images_dir)
        if delta_blocks:
            print(f"\n% === Delta radar charts ===")
            delta_latex_parts = []
            for latex, desc in delta_blocks:
                print(f"% {desc}")
                print(latex)
                delta_latex_parts.append(latex)
            all_latex.append("% === Delta radar charts ===\n" + "\n".join(delta_latex_parts))

        print("\n" + "=" * 60)
        print("QUALITATIVE PROSE SUMMARY")
        print("=" * 60)
        for assistant, run_id in specs:
            label = run_labels[(assistant, run_id)]
            summary = load_qualitative_summary(ckpt, assistant, run_id)
            if summary:
                print(f"\n--- {label} / {run_id} ---")
                print(summary)
                all_latex.append(
                    f"% === Qualitative prose summary: {label}/{run_id} ===\n"
                    f"% {summary.replace(chr(10), chr(10) + '% ')}"
                )
            else:
                print(f"\n--- {label} / {run_id} ---")
                print("  (no qualitative_summary.txt -- re-run protocol with --resume to generate)")
    else:
        print("\n(No qualitative.json files found -- run with --resume to regenerate)")

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
