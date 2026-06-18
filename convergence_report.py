"""
convergence_report.py — convergence status for a protocol run.

Three convergence types reported per scenario:
  - Behavioral/intent    : ARI_inter >= noise_floor AND >= ari_min_threshold
                           (this is what the protocol uses to stop)
  - Behavioral/execution : same criterion on execution clusters
                           (this is what the protocol uses to stop)
  - Performance          : CI width <= threshold (informational only;
                           the protocol does NOT use this to stop)
                           Rate metrics [0,1] use absolute CI width (hi - lo).
                           Count/cost metrics use relative CI width (width / point).

Distance-to-convergence:
  - Behavioral : max(0, max(noise_floor, ari_threshold) - ARI_inter)
  - Performance: max(0, ci_width - threshold)

Usage:
    python convergence_report.py [run_id] [--assistant LLM]
                                 [--checkpoint-dir Logs/Evaluation]
                                 [--rate-threshold 0.20]
                                 [--rel-threshold 0.20]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Rate metrics use absolute CI width; others use relative CI width
RATE_METRICS = ["success_rate", "correct_hypothesis_rate"]
REL_METRICS  = ["total_cost", "n_actions", "n_hypotheses"]
PERF_METRICS = ["success_rate", "total_cost", "n_actions", "n_hypotheses"]


def _latest_run(base: Path) -> str:
    runs = sorted(d.name for d in base.iterdir() if d.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {base}")
    return runs[-1]


def _load(run_dir: Path, rate_threshold: float, rel_threshold: float) -> list[dict]:
    rows = []
    for scenario_dir in sorted(run_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 9999):
        if not scenario_dir.is_dir() or not scenario_dir.name.isdigit():
            continue
        ckpt = scenario_dir / "checkpoint.json"
        if not ckpt.exists():
            continue
        data = json.loads(ckpt.read_text())
        if not data.get("batch_history"):
            continue
        last = data["batch_history"][-1]
        ari_thr = last.get("ari_min_threshold", 0.30)

        # ── Behavioral ────────────────────────────────────────────────────────
        ari_i   = last.get("ari_inter_intent")
        ari_e   = last.get("ari_inter_execution")
        floor_i = last.get("ari_boot_noise_floor_intent", 0.0)
        floor_e = last.get("ari_boot_noise_floor_execution", 0.0)
        conv_i  = last.get("converged_intent", False)
        conv_e  = last.get("converged_execution", False)

        def _beh_dist(ari, floor):
            if ari is None:
                return None
            return round(max(0.0, max(floor, ari_thr) - ari), 3)

        # ── Performance (informational) ───────────────────────────────────────
        nm = last.get("numerical_metrics", {})
        perf = {}
        for key in PERF_METRICS:
            m = nm.get(key, {})
            lo = m.get("ci_lo")
            hi = m.get("ci_hi")
            pt = m.get("point")
            if key in RATE_METRICS:
                # absolute CI width for bounded [0,1] rates
                width = round(hi - lo, 3) if (lo is not None and hi is not None) else None
                dist  = round(max(0.0, width - rate_threshold), 3) if width is not None else None
                perf[key] = {"point": pt, "ci_lo": lo, "ci_hi": hi,
                             "width": width, "dist": dist, "absolute": True}
            else:
                # relative CI width for counts/costs
                rw = m.get("rel_width")
                if rw is None and pt and pt != 0 and lo is not None and hi is not None:
                    rw = (hi - lo) / abs(pt)
                if rw is not None and (math.isnan(rw) or math.isinf(rw)):
                    rw = None
                dist = round(max(0.0, rw - rel_threshold), 3) if rw is not None else None
                perf[key] = {"point": pt, "ci_lo": lo, "ci_hi": hi,
                             "width": round(rw, 3) if rw is not None else None, "dist": dist, "absolute": False}

        # ── Cluster summary ───────────────────────────────────────────────────
        ai = data.get("cluster_assignments_intent", [])
        ae = data.get("cluster_assignments_execution", [])
        def _cluster_summary(assignments):
            if not assignments:
                return {"n_clusters": 0, "noise_frac": None, "sizes": []}
            from collections import Counter
            counts = Counter(assignments)
            noise = counts.get(-1, 0)
            real_sizes = sorted([v for k, v in counts.items() if k != -1], reverse=True)
            return {
                "n_clusters": last.get("n_clusters_intent", len(real_sizes)),
                "noise_frac": round(noise / len(assignments), 3) if assignments else None,
                "sizes": real_sizes,
            }
        ci_summary = _cluster_summary(ai)
        ci_summary["n_clusters"] = last.get("n_clusters_intent", ci_summary["n_clusters"])
        ce_summary = _cluster_summary(ae)
        ce_summary["n_clusters"] = last.get("n_clusters_execution", ce_summary["n_clusters"])

        rows.append({
            "scenario":    int(scenario_dir.name),
            "n":           data.get("n_trajectories", 0),
            "batches":     data.get("batch_index", 0),
            "conv_intent": conv_i,
            "conv_exec":   conv_e,
            "ari_intent":  round(ari_i, 3) if ari_i is not None else None,
            "ari_exec":    round(ari_e, 3) if ari_e is not None else None,
            "floor_intent":  round(floor_i, 3),
            "floor_exec":    round(floor_e, 3),
            "dist_intent": _beh_dist(ari_i, floor_i),
            "dist_exec":   _beh_dist(ari_e, floor_e),
            "ari_thr":     ari_thr,
            "perf":        perf,
            "clusters_intent": ci_summary,
            "clusters_exec":   ce_summary,
        })
    return rows


def _fmt(v, width=6):
    return "N/A".rjust(width) if v is None else f"{v:.3f}".rjust(width)

def _c(c):
    return "✓" if c else "✗"


def _print_behavioral(rows: list[dict]) -> None:
    n_i = sum(r["conv_intent"] for r in rows)
    n_e = sum(r["conv_exec"]   for r in rows)
    total = len(rows)

    print(f"\n{'─'*74}")
    print(f"  BEHAVIORAL CONVERGENCE  (used by protocol to stop)")
    print(f"  intent converged: {n_i}/{total}  |  execution converged: {n_e}/{total}")
    print(f"{'─'*74}")
    print(f"{'Sc':>3}  {'N':>3}  "
          f"{'I-ARI':>6}  {'I-flr':>6}  {'I-dist':>6}  {'I':>2}  "
          f"{'E-ARI':>6}  {'E-flr':>6}  {'E-dist':>6}  {'E':>2}")
    print(f"{'':─>3}  {'':─>3}  "
          f"{'':─>6}  {'':─>6}  {'':─>6}  {'':─>2}  "
          f"{'':─>6}  {'':─>6}  {'':─>6}  {'':─>2}")
    for r in rows:
        print(f"{r['scenario']:>3}  {r['n']:>3}  "
              f"{_fmt(r['ari_intent']):>6}  {r['floor_intent']:>6.3f}  {_fmt(r['dist_intent']):>6}  {_c(r['conv_intent']):>2}  "
              f"{_fmt(r['ari_exec']):>6}  {r['floor_exec']:>6.3f}  {_fmt(r['dist_exec']):>6}  {_c(r['conv_exec']):>2}")

    not_e = sorted([r for r in rows if not r["conv_exec"] and r["dist_exec"] is not None],
                   key=lambda x: x["dist_exec"])
    if not_e:
        print(f"\n  Execution not converged — ranked closest first:")
        for r in not_e:
            needed = max(r["floor_exec"], r["ari_thr"])
            print(f"    Sc {r['scenario']:>3}:  ARI={_fmt(r['ari_exec'])}  floor={r['floor_exec']:.3f}"
                  f"  need≥{needed:.3f}  dist-to-conv={r['dist_exec']:.3f}")


def _print_clusters(rows: list[dict]) -> None:
    print(f"\n{'─'*74}")
    print(f"  CLUSTER SUMMARY  (current state after last batch)")
    print(f"  noise_frac = fraction of trajectories assigned to no cluster (-1)")
    print(f"  stability  = ARI_inter (how much cluster assignments agree across batches)")
    print(f"{'─'*74}")
    print(f"{'Sc':>3}  {'N':>3}  "
          f"{'I-#cls':>6}  {'I-noise':>7}  {'I-stab':>7}  {'I-sizes':<16}  "
          f"{'E-#cls':>6}  {'E-noise':>7}  {'E-stab':>7}  {'E-sizes':<16}")
    print(f"{'':─>3}  {'':─>3}  "
          f"{'':─>6}  {'':─>7}  {'':─>7}  {'':─>16}  "
          f"{'':─>6}  {'':─>7}  {'':─>7}  {'':─>16}")
    for r in rows:
        ci = r["clusters_intent"]
        ce = r["clusters_exec"]

        def _nf(v): return f"{v:.2f}" if v is not None else " N/A"
        def _sizes(s): return "[" + ",".join(str(x) for x in s[:6]) + ("+]" if len(s) > 6 else "]")

        i_stab = _fmt(r["ari_intent"]).strip()
        e_stab = _fmt(r["ari_exec"]).strip()

        print(f"{r['scenario']:>3}  {r['n']:>3}  "
              f"{ci['n_clusters']:>6}  {_nf(ci['noise_frac']):>7}  {i_stab:>7}  {_sizes(ci['sizes']):<16}  "
              f"{ce['n_clusters']:>6}  {_nf(ce['noise_frac']):>7}  {e_stab:>7}  {_sizes(ce['sizes']):<16}")


def _print_performance(rows: list[dict], rate_threshold: float, rel_threshold: float) -> None:
    print(f"\n{'─'*74}")
    print(f"  PERFORMANCE CONVERGENCE  (informational — not used by protocol)")
    print(f"  success_rate: absolute CI width ≤ {rate_threshold}  |  "
          f"count/cost: relative CI width ≤ {rel_threshold}")
    print(f"  format: ✓/✗ width (d=dist-to-conv)")
    print(f"{'─'*74}")

    col = 13
    print(f"{'Sc':>3}  {'N':>3}  " + "  ".join(f"{m[:col]:>{col}}" for m in PERF_METRICS))
    print(f"{'':─>3}  {'':─>3}  " + "  ".join("─" * col for _ in PERF_METRICS))

    for r in rows:
        parts = []
        for key in PERF_METRICS:
            p    = r["perf"].get(key, {})
            w    = p.get("width")
            dist = p.get("dist")
            if w is None:
                parts.append("N/A".rjust(col))
            else:
                mark = "✓" if dist == 0.0 else "✗"
                parts.append(f"{mark}{w:.3f}(d={dist:.2f})".rjust(col))
        print(f"{r['scenario']:>3}  {r['n']:>3}  " + "  ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Convergence report for a protocol run.")
    parser.add_argument("run_id", nargs="?", default=None)
    parser.add_argument("--assistant", default="LLM")
    parser.add_argument("--checkpoint-dir", default="Logs/Evaluation")
    parser.add_argument("--rate-threshold", type=float, default=0.20,
                        help="Absolute CI width threshold for rate metrics like success_rate (default: 0.20)")
    parser.add_argument("--rel-threshold", type=float, default=0.20,
                        help="Relative CI width threshold for count/cost metrics (default: 0.20)")
    args = parser.parse_args()

    base = Path(args.checkpoint_dir) / args.assistant
    run_id = args.run_id or _latest_run(base)
    run_dir = base / run_id

    print(f"\nRun: {run_dir}")
    rows = _load(run_dir, args.rate_threshold, args.rel_threshold)

    _print_behavioral(rows)
    _print_clusters(rows)
    _print_performance(rows, args.rate_threshold, args.rel_threshold)


if __name__ == "__main__":
    main()
