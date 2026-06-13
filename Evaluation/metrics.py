"""
Statistical metrics for the adaptive evaluation protocol.

All functions are pure (no side effects, no I/O).
"""
from __future__ import annotations

import math
import random
from typing import Optional

from sklearn.metrics import adjusted_rand_score


# ---------------------------------------------------------------------------
# ARI helpers
# ---------------------------------------------------------------------------

def ari_inter(
    prev_assignments: list[int],
    curr_assignments: list[int],
    shared_indices: list[int],
) -> float:
    """
    Compute ARI between two clusterings restricted to shared trajectory indices.

    prev_assignments / curr_assignments: full assignment vectors (one label per trajectory).
    shared_indices: positions present in both batches.
    Returns float in [-1, 1]; 1 = identical clusterings.
    """
    if not shared_indices:
        return 0.0
    prev = [prev_assignments[i] for i in shared_indices]
    curr = [curr_assignments[i] for i in shared_indices]
    return float(adjusted_rand_score(prev, curr))


def bootstrap_ari_noise_floor(
    assignments: list[int],
    n_resamples: int = 1000,
    percentile: float = 0.05,
    clustering_fn=None,
) -> float:
    """
    Estimate the ARI noise floor via bootstrap resampling.

    For each resample: draw N indices with replacement from [0, N), compute
    the cluster labels on the resampled subset via ``clustering_fn``, then
    compute ARI against the reference assignment (full dataset).

    clustering_fn(indices: list[int]) -> list[int]
        Should return per-resample cluster labels aligned to the full dataset.
        If None, uses the identity (no re-clustering) — only useful for tests.

    Returns the ``percentile``-th percentile of bootstrap ARI values.
    """
    n = len(assignments)
    if n == 0:
        return 0.0

    ari_values: list[float] = []
    for _ in range(n_resamples):
        indices = [random.randrange(n) for _ in range(n)]
        if clustering_fn is not None:
            boot_labels = clustering_fn(indices)
        else:
            boot_labels = [assignments[i] for i in indices]
        ref_labels = [assignments[i] for i in indices]
        try:
            ari = float(adjusted_rand_score(ref_labels, boot_labels))
        except Exception:
            ari = 0.0
        ari_values.append(ari)

    ari_values.sort()
    idx = int(percentile * n_resamples)
    idx = max(0, min(idx, n_resamples - 1))
    return ari_values[idx]


# ---------------------------------------------------------------------------
# Confidence intervals
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson score confidence interval for a Bernoulli proportion.
    Returns (lo, hi).
    """
    if n == 0:
        return 0.0, 1.0
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p_hat = k / n
    centre = (p_hat + z**2 / (2 * n)) / (1 + z**2 / n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return max(0.0, centre - margin), min(1.0, centre + margin)


def bootstrap_ci(
    values: list[float],
    n_resamples: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for the mean of ``values``.
    Returns (lo, hi, rel_width). rel_width = (hi - lo) / mean.
    """
    if not values:
        return 0.0, 0.0, float("inf")
    n = len(values)
    boot_means = []
    for _ in range(n_resamples):
        resample = [values[random.randrange(n)] for _ in range(n)]
        boot_means.append(sum(resample) / n)
    boot_means.sort()
    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples)
    lo = boot_means[max(0, lo_idx)]
    hi = boot_means[min(n_resamples - 1, hi_idx)]
    mean = sum(values) / n
    rel_width = (hi - lo) / mean if mean != 0 else float("inf")
    return lo, hi, rel_width


# ---------------------------------------------------------------------------
# Numerical metrics (per batch)
# ---------------------------------------------------------------------------

def compute_numerical_metrics(
    trajectories: list[dict],
    bootstrap_samples: int = 1000,
) -> dict:
    """
    Compute per-batch numerical metrics from a list of trajectory dicts.

    Each trajectory dict must have:
      "end": str ("success"|"success_no_hypothesis"|"timeout"|"surrender"|"exception")
      "total_cost": float
      "length": int
      "actions": list of action dicts
      "hypotheses_count": {"wrong": int, "partial": int, "right": int}

    Returns a dict suitable for embedding in the checkpoint batch_history entry.
    """
    if not trajectories:
        return {}

    n = len(trajectories)

    # Success rate (Wilson CI)
    successes = [t for t in trajectories if t.get("end") in ("success", "success_no_hypothesis")]
    k = len(successes)
    sr_lo, sr_hi = wilson_ci(k, n)

    def _metric(values: list[float]) -> dict:
        if not values:
            return {"point": None, "ci_lo": None, "ci_hi": None, "rel_width": None}
        point = sum(values) / len(values)
        lo, hi, rw = bootstrap_ci(values, n_resamples=bootstrap_samples)
        return {"point": round(point, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "rel_width": round(rw, 4)}

    costs   = [float(t.get("total_cost", 0)) for t in trajectories]
    lengths = [float(t.get("length", 0))     for t in trajectories]

    def _hyp(t: dict, key: str) -> float:
        return float(t.get("hypotheses_count", {}).get(key, 0))

    n_hypotheses = [sum(t.get("hypotheses_count", {}).values()) for t in trajectories]
    n_hypotheses = [float(v) for v in n_hypotheses]
    n_correct    = [_hyp(t, "right")   for t in trajectories]
    n_wrong      = [_hyp(t, "wrong")   for t in trajectories]
    n_partial    = [_hyp(t, "partial") for t in trajectories]

    # Correct hypothesis rate: pooled Wilson CI over all hypotheses across trajectories
    k_hyp = int(sum(n_correct))
    n_hyp = int(sum(n_hypotheses))
    chr_lo, chr_hi = wilson_ci(k_hyp, n_hyp)

    return {
        "success_rate":  {
            "point": round(k / n, 4),
            "ci_lo": round(sr_lo, 4),
            "ci_hi": round(sr_hi, 4),
        },
        "correct_hypothesis_rate": {
            "point": round(k_hyp / n_hyp, 4) if n_hyp > 0 else None,
            "ci_lo": round(chr_lo, 4),
            "ci_hi": round(chr_hi, 4),
            "k": k_hyp,
            "n": n_hyp,
        },
        "total_cost":    _metric(costs),
        "n_actions":     _metric(lengths),
        "n_hypotheses":  _metric(n_hypotheses),
        "n_correct":     _metric(n_correct),
        "n_wrong":       _metric(n_wrong),
        "n_partial":     _metric(n_partial),
    }
