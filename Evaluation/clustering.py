"""
Trajectory clustering for the adaptive evaluation protocol.

Intent level  — HDBSCAN on sentence-transformer embeddings of NL action sequences.
Execution level — Agglomerative clustering on normalized Levenshtein distance of
                  symbolic action sequences (action_id + targets).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import threading

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Module-level cache for SentenceTransformer to avoid concurrent model loading
_embedding_model_cache: dict[str, "SentenceTransformer"] = {}
_embedding_model_lock = threading.Lock()

# model used to label clusters
LABELING_MODEL = "gpt-4.1-mini"

# ---------------------------------------------------------------------------
# Sequence extraction from trajectory JSON dicts
# ---------------------------------------------------------------------------

def intent_sequence(trajectory: dict) -> str:
    """NL action sequence joined by ' → '."""
    return " → ".join(a.get("intention", "") for a in trajectory.get("actions", []))


def execution_sequence(trajectory: dict) -> tuple[str, ...]:
    """
    Symbolic sequence: one token per implementation sub-action, encoding
    both action_id and all targets.
    E.g. "observe_component(subject:main_bulb)"
    """
    tokens = []
    for action in trajectory.get("actions", []):
        for impl in action.get("implementation", []):
            aid = impl.get("action_id", "?")
            targets = impl.get("targets", {})
            target_str = ",".join(f"{k}:{v}" for k, v in sorted(targets.items()))
            tokens.append(f"{aid}({target_str})")
    return tuple(tokens)


# ---------------------------------------------------------------------------
# Intent-level clustering (HDBSCAN)
# ---------------------------------------------------------------------------

def embed_intent(
    sequences: list[str],
    embedding_model: str = "all-MiniLM-L6-v2",
    mock_embeddings: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed intent sequences. Returns (dist_matrix, embeddings), both float64."""
    n = len(sequences)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0, 0), dtype=np.float64)

    if mock_embeddings:
        rng = np.random.default_rng(42)
        raw = rng.normal(size=(n, 32)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        embeddings = (raw / np.maximum(norms, 1e-9)).astype(np.float64)
    else:
        from sentence_transformers import SentenceTransformer
        # Load model once, cache it thread-safely to avoid concurrent loading issues
        with _embedding_model_lock:
            if embedding_model not in _embedding_model_cache:
                _embedding_model_cache[embedding_model] = SentenceTransformer(embedding_model)
            model = _embedding_model_cache[embedding_model]
        embeddings = model.encode(
            sequences, show_progress_bar=False, convert_to_numpy=True
        ).astype(np.float64)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-9)
    dist_matrix = np.clip(1.0 - (normed @ normed.T), 0.0, 2.0)
    return dist_matrix, embeddings


def _hdbscan_on_dist(
    dist_matrix: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 1,
) -> tuple[list[int], Optional[np.ndarray]]:
    """Run HDBSCAN on a precomputed cosine distance matrix."""
    import hdbscan
    n = dist_matrix.shape[0]
    if n == 0:
        return [], None
    if n == 1:
        # Single trajectory — assign to cluster 0 with prob 1.0
        return [0], np.array([1.0])
    if dist_matrix.max() == 0.0:
        return [0] * n, np.ones(n)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="precomputed",
    )
    clusterer.fit(dist_matrix)
    return clusterer.labels_.tolist(), clusterer.probabilities_


def cluster_intent(
    sequences: list[str],
    min_cluster_size: int = 3,
    min_samples: int = 1,
    embedding_model: str = "all-MiniLM-L6-v2",
    mock_embeddings: bool = False,
) -> tuple[list[int], Optional[np.ndarray]]:
    """
    Embed intent sequences and cluster with HDBSCAN.

    Returns (labels, membership_probabilities).
    labels[i] == -1 means outlier/noise.
    mock_embeddings=True: use random unit vectors (no model download needed for testing).
    """
    if not sequences:
        return [], None
    dist_matrix = embed_intent(sequences, embedding_model, mock_embeddings)
    return _hdbscan_on_dist(dist_matrix, min_cluster_size, min_samples)


def label_clusters_llm(
    sequences: list[str],
    labels: list[int],
    probabilities: Optional[np.ndarray],
    mock: bool = False,
    labeling_model: str = LABELING_MODEL,
) -> dict[int, str]:
    """
    Generate a short natural-language label for each intent cluster.

    In mock mode returns "Cluster N" without any API call.
    Otherwise calls claude-haiku-4-5-20251001 via the Anthropic SDK.
    """
    cluster_ids = sorted(set(l for l in labels if l >= 0))
    if not cluster_ids:
        return {}

    result: dict[int, str] = {}

    for cid in cluster_ids:
        if mock:
            result[cid] = f"Cluster {cid}"
            continue

        # Find most representative trajectory (highest membership probability)
        members = [i for i, l in enumerate(labels) if l == cid]
        if probabilities is not None and len(members) > 0:
            best_idx = max(members, key=lambda i: probabilities[i])
        else:
            best_idx = members[0]
        representative = sequences[best_idx]

        try:
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=labeling_model,
                max_tokens=40,
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarize this diagnostic action sequence in 3-6 words:\n"
                        + representative
                    ),
                }],
            )
            result[cid] = resp.choices[0].message.content.strip()
        except Exception as e:
            result[cid] = f"Cluster {cid} (label error: {e})"

    return result


# ---------------------------------------------------------------------------
# Execution-level clustering (agglomerative + Levenshtein)
# ---------------------------------------------------------------------------

def _levenshtein(a: tuple, b: tuple) -> int:
    """Edit distance between two tuples of strings."""
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]


def _normalized_levenshtein(a: tuple, b: tuple) -> float:
    denom = max(len(a), len(b), 1)
    return _levenshtein(a, b) / denom


def cluster_execution(
    sequences: list[tuple[str, ...]],
    linkage: str = "average",
    cut_grid: Optional[list[float]] = None,
    deduplicate: bool = False,
) -> tuple[list[int], Optional[np.ndarray], Optional[float], list[tuple[str, ...]]]:
    """
    Cluster symbolic execution sequences via agglomerative + Levenshtein.

    1. Optionally deduplicate sequences (deduplicate=True avoids bias from repeated identical runs).
    2. Build pairwise distance matrix on the working set of sequences.
    3. Agglomerative clustering; select cut height maximising silhouette score.
    4. Map labels back to all trajectories via sequence identity.

    Returns (labels, Z, best_cut_height, clustered_seqs).
    labels is aligned to input ``sequences``. Z and best_cut_height are None when n <= 1.
    """
    from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
    from scipy.spatial.distance import squareform
    from sklearn.metrics import silhouette_score

    if not sequences:
        return [], None, None, []

    if cut_grid is None:
        cut_grid = [i / 20 for i in range(1, 20)]

    unique_seqs: list[tuple[str, ...]]
    if deduplicate:
        unique_seqs = list(dict.fromkeys(sequences))
    else:
        unique_seqs = list(sequences)

    n_unique = len(unique_seqs)
    seq_to_label: dict[tuple, int] = {}

    if n_unique == 1:
        seq_to_label[unique_seqs[0]] = 0
        return [0] * len(sequences), None, None, unique_seqs

    # Pairwise distance matrix on unique sequences
    dist_matrix = np.zeros((n_unique, n_unique))
    for i in range(n_unique):
        for j in range(i + 1, n_unique):
            d = _normalized_levenshtein(unique_seqs[i], unique_seqs[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    Z = scipy_linkage(condensed, method=linkage)

    # Select cut height by silhouette score.
    # Singleton clusters (1 member each) are assigned silhouette = 0.0 — they
    # are valid but uninformative; any cut with natural groupings will score higher.
    best_score = -2.0
    best_labels = np.zeros(n_unique, dtype=int)  # fallback: one cluster for all
    best_cut_height: Optional[float] = None
    for height in cut_grid:
        labels = fcluster(Z, t=height, criterion="distance") - 1  # 0-based
        n_clusters = len(set(labels))
        if n_clusters < 2:
            continue
        # Count samples per cluster; singletons contribute 0 to silhouette
        counts = np.bincount(labels)
        if all(c == 1 for c in counts):
            score = 0.0  # all singletons → neutral score
        else:
            try:
                score = silhouette_score(dist_matrix, labels, metric="precomputed")
            except Exception:
                score = 0.0
        if score > best_score:
            best_score = score
            best_labels = labels
            best_cut_height = height

    for i, seq in enumerate(unique_seqs):
        seq_to_label[seq] = int(best_labels[i])

    return [seq_to_label[s] for s in sequences], Z, best_cut_height, unique_seqs


# ---------------------------------------------------------------------------
# Dendrogram visualisation
# ---------------------------------------------------------------------------

def _shorten_seq(seq: tuple[str, ...], head_tail_n: int = 2) -> str:
    ac_ids = [tok.split("(")[0] for tok in seq]
    ob_ids = [tok.split("(")[1][:-1].replace("subject:","").replace("subjects:","") for tok in seq]
    ids = [ac[:10]+":"+ob[:10] for ac,ob in zip(ac_ids,ob_ids)]
    if len(ids) <= 2 * head_tail_n:
        return " → ".join(ids)
    head = ids[:head_tail_n]
    tail = ids[-head_tail_n:]
    return " → ".join(head) + " → … → " + " → ".join(tail)


def save_dendrogram(
    Z: np.ndarray,
    clustered_seqs: list[tuple[str, ...]],
    best_cut_height: float,
    output_path: Path,
    title: str = "",
    head_tail_n: int = 2,
    min_link_height: float = 0.02,
) -> None:
    from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

    # Apply a visual floor to zero-height merges so duplicate leaves are
    # rendered with a visible branch rather than a flat zero-length line.
    # This only affects the plot — the clustering and cut line are unchanged.
    Z_visual = Z.copy()
    Z_visual[:, 2] = np.maximum(Z_visual[:, 2], min_link_height)

    labels = [_shorten_seq(s, head_tail_n) for s in clustered_seqs]
    n_leaves = len(labels)

    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * n_leaves)))
    scipy_dendrogram(Z_visual, orientation="left", labels=labels, ax=ax, leaf_font_size=8)
    ax.axvline(
        x=best_cut_height, color="red", linestyle="--", linewidth=1,
        label=f"cut = {best_cut_height:.2f}",
    )
    ax.set_title(title)
    ax.set_xlabel("Normalized Levenshtein distance")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def _abbrev_intent(seq: str, n: int = 2, max_tok: int = 22) -> str:
    """Abbreviate an intent sequence to head+tail steps, each capped at max_tok chars."""
    def _trunc(s: str) -> str:
        return s if len(s) <= max_tok else s[:max_tok - 1] + "…"
    parts = [_trunc(tok) for tok in seq.split(" → ")]
    if len(parts) <= 2 * n:
        return " → ".join(parts)
    return " → ".join(parts[:n]) + " → … → " + " → ".join(parts[-n:])


def save_intent_scatter(
    embeddings_2d: np.ndarray,
    labels: list[int],
    intent_seqs: list[str],
    cluster_label_names: Optional[dict[int, str]],
    output_path: Path,
    title: str = "",
) -> None:
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines

    # Colour-blind-safe palette (Okabe-Ito) + distinct marker shapes
    _COLORS  = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2",
                "#D55E00","#CC79A7","#000000","#999999","#44AA99"]
    _MARKERS = ["o","s","^","D","v","P","X","*","h","<"]

    cmap = plt.cm.tab10
    cluster_ids = sorted(set(l for l in labels if l >= 0))

    def _color(label: int):
        if label < 0:
            return "#cccccc"
        return _COLORS[label % len(_COLORS)]

    def _marker(label: int):
        if label < 0:
            return "x"
        return _MARKERS[label % len(_MARKERS)]

    fig, ax = plt.subplots(figsize=(10, 7))
    xs, ys = embeddings_2d[:, 0], embeddings_2d[:, 1]

    for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
        is_unfilled = _marker(lbl) == "x"
        scatter_kw = dict(c=[_color(lbl)], marker=_marker(lbl), s=50, alpha=0.85, zorder=2, linewidths=0.5)
        if not is_unfilled:
            scatter_kw["edgecolors"] = "white"
        ax.scatter(x, y, **scatter_kw)
        text = _abbrev_intent(intent_seqs[i])
        ax.annotate(text, (x, y), fontsize=5, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    # Legend — use Line2D handles so marker shape shows alongside colour
    handles = []
    for cid in cluster_ids:
        name = (cluster_label_names or {}).get(cid, f"Cluster {cid}")
        if len(name) > 40:
            name = name[:39] + "…"
        handles.append(mlines.Line2D([], [], color=_color(cid), marker=_marker(cid),
                                     linestyle="None", markersize=7,
                                     label=f"[{cid}] {name}"))
    if any(l < 0 for l in labels):
        handles.append(mlines.Line2D([], [], color="#cccccc", marker="x",
                                     linestyle="None", markersize=7, label="noise"))
    if handles:
        ax.legend(handles=handles, fontsize=7, loc="best")

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cost distribution per cluster: violin + histogram
# ---------------------------------------------------------------------------

_CLUSTER_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#F0E442",
    "#0072B2", "#D55E00", "#CC79A7", "#000000", "#999999", "#44AA99",
]


def save_cluster_cost_violin(
    costs: list[float],
    labels: list[int],
    cluster_label_names: Optional[dict[int, str]],
    output_path: Path,
    title: str = "",
) -> None:
    """
    Violin plot: one violin per cluster (noise excluded).
    Individual trajectory costs overlaid as a strip of dots (jittered).
    X = cluster label, Y = cost.
    """
    import matplotlib.lines as mlines

    cluster_ids = sorted(set(l for l in labels if l >= 0))
    if not cluster_ids:
        return

    data_by_cluster = {
        cid: [c for c, l in zip(costs, labels) if l == cid]
        for cid in cluster_ids
    }

    fig, ax = plt.subplots(figsize=(max(5, len(cluster_ids) * 1.4), 5))

    positions = list(range(len(cluster_ids)))
    violin_data = [data_by_cluster[cid] for cid in cluster_ids]

    parts = ax.violinplot(violin_data, positions=positions,
                          showmedians=True, showextrema=True, widths=0.7)

    for i, (pc, cid) in enumerate(zip(parts["bodies"], cluster_ids)):
        color = _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
        pc.set_edgecolor(color)

    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.0)

    rng = np.random.default_rng(0)
    for i, (cid, pos) in enumerate(zip(cluster_ids, positions)):
        vals = data_by_cluster[cid]
        if not vals:
            continue
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        color = _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]
        ax.scatter(np.array(pos) + jitter, vals,
                   color=color, s=18, alpha=0.7, zorder=3, linewidths=0)

    tick_labels = [
        (cluster_label_names or {}).get(cid, f"C{cid}") + f"\n(n={len(data_by_cluster[cid])})"
        for cid in cluster_ids
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("Total cost (technician-seconds)", fontsize=9)
    ax.set_title(title or "Cost distribution per cluster", fontsize=9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    noise_count = sum(1 for l in labels if l < 0)
    if noise_count:
        ax.text(0.99, 0.01, f"noise (excluded): {noise_count}",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                color="#999999")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def save_cluster_cost_histogram(
    costs: list[float],
    labels: list[int],
    cluster_label_names: Optional[dict[int, str]],
    output_path: Path,
    title: str = "",
) -> None:
    """
    Bar chart: one bar per cluster, positioned at the cluster's mean cost on
    the x-axis, height = count. Gives an immediate sense of where each cluster
    sits on the cost axis.
    """
    cluster_ids = sorted(set(l for l in labels if l >= 0))
    if not cluster_ids:
        return

    data_by_cluster = {
        cid: [c for c, l in zip(costs, labels) if l == cid and not np.isnan(c)]
        for cid in cluster_ids
    }

    fig, ax = plt.subplots(figsize=(max(5, len(cluster_ids) * 1.4), 4))

    # Determine a sensible bar width relative to cost range
    all_means = [np.mean(data_by_cluster[cid]) for cid in cluster_ids if data_by_cluster[cid]]
    cost_range = max(all_means) - min(all_means) if len(all_means) > 1 else max(all_means, default=1)
    bar_w = max(cost_range * 0.06, 5.0)

    for cid in cluster_ids:
        vals = data_by_cluster[cid]
        if not vals:
            continue
        mean_cost = float(np.mean(vals))
        color = _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]
        name = (cluster_label_names or {}).get(cid, f"Cluster {cid}")
        ax.bar(mean_cost, len(vals), width=bar_w,
               color=color, alpha=0.75, label=f"[{cid}] {name} (n={len(vals)})", zorder=2)
        ax.text(mean_cost, len(vals) + 0.1, f"μ={mean_cost:.0f}",
                ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Mean total cost (technician-seconds)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(title or "Cost per cluster (bar at mean)", fontsize=9)
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    noise_count = sum(1 for l in labels if l < 0)
    if noise_count:
        ax.text(0.99, 0.99, f"noise (excluded): {noise_count}",
                transform=ax.transAxes, fontsize=7, ha="right", va="top",
                color="#999999")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


def save_cluster_cost_boxplot(
    costs: list[float],
    labels: list[int],
    cluster_label_names: Optional[dict[int, str]],
    output_path: Path,
    title: str = "",
) -> None:
    """
    Box-and-whisker plot: one box per cluster showing median, IQR, and outliers.
    Complements the violin (which shows full density) with explicit quantile markers.
    """
    cluster_ids = sorted(set(l for l in labels if l >= 0))
    if not cluster_ids:
        return

    data_by_cluster = {
        cid: [c for c, l in zip(costs, labels) if l == cid and not np.isnan(c)]
        for cid in cluster_ids
    }

    fig, ax = plt.subplots(figsize=(max(5, len(cluster_ids) * 1.4), 5))
    positions = list(range(len(cluster_ids)))

    box_data = [data_by_cluster[cid] for cid in cluster_ids]
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True,
                    widths=0.5, showfliers=True,
                    medianprops=dict(color="#333333", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=4, alpha=0.5))

    for patch, cid in zip(bp["boxes"], cluster_ids):
        color = _CLUSTER_COLORS[cid % len(_CLUSTER_COLORS)]
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    tick_labels = [
        (cluster_label_names or {}).get(cid, f"C{cid}") + f"\n(n={len(data_by_cluster[cid])})"
        for cid in cluster_ids
    ]
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=7)
    ax.set_ylabel("Total cost (technician-seconds)", fontsize=9)
    ax.set_title(title or "Cost boxplot per cluster", fontsize=9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)

    noise_count = sum(1 for l in labels if l < 0)
    if noise_count:
        ax.text(0.99, 0.01, f"noise (excluded): {noise_count}",
                transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                color="#999999")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Cross-scenario aggregate cost plots
# ---------------------------------------------------------------------------

def save_aggregate_cost_plots(
    per_scenario_data: list[dict],
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """
    Aggregate violin + histogram + boxplot across all scenarios in a run.

    per_scenario_data is a list of dicts, one per scenario, each with:
      - "costs":  list[float]  — total_cost per trajectory
      - "labels": list[int]    — intent cluster assignment per trajectory (unused for aggregate)
      - "scenario": int        — scenario number
    """
    all_costs: list[float] = []
    all_scenario_ids: list[int] = []

    for entry in per_scenario_data:
        costs = entry.get("costs", [])
        labels = entry.get("labels", [])
        scen = entry.get("scenario", 0)
        for c, l in zip(costs, labels):
            if l >= 0 and not np.isnan(c):
                all_costs.append(c)
                all_scenario_ids.append(scen)

    if not all_costs:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    scenarios = sorted(set(all_scenario_ids))
    data_by_scen = {s: [c for c, sid in zip(all_costs, all_scenario_ids) if sid == s] for s in scenarios}
    positions = list(range(len(scenarios)))
    violin_data = [data_by_scen[s] for s in scenarios]

    # Violin per scenario
    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 0.6), 5))
    parts = ax.violinplot(violin_data, positions=positions, showmedians=True, widths=0.7)
    for pc in parts["bodies"]:
        pc.set_facecolor("#56B4E9")
        pc.set_alpha(0.45)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("#333333")
    rng = np.random.default_rng(0)
    for pos, s in zip(positions, scenarios):
        vals = data_by_scen[s]
        jitter = rng.uniform(-0.18, 0.18, size=len(vals))
        ax.scatter(np.array(pos) + jitter, vals, s=8, alpha=0.5, color="#0072B2", zorder=3, linewidths=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(s) for s in scenarios], fontsize=6, rotation=90)
    ax.set_xlabel("Scenario", fontsize=9)
    ax.set_ylabel("Total cost (technician-seconds)", fontsize=9)
    ax.set_title(f"{title_prefix}Cost distribution across scenarios", fontsize=9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "aggregate_cost_violin.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    # Boxplot per scenario
    fig, ax = plt.subplots(figsize=(max(6, len(scenarios) * 0.6), 5))
    bp = ax.boxplot(violin_data, positions=positions, patch_artist=True, widths=0.5,
                    showfliers=True,
                    medianprops=dict(color="#333333", linewidth=1.5),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch in bp["boxes"]:
        patch.set_facecolor("#56B4E9")
        patch.set_alpha(0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([str(s) for s in scenarios], fontsize=6, rotation=90)
    ax.set_xlabel("Scenario", fontsize=9)
    ax.set_ylabel("Total cost (technician-seconds)", fontsize=9)
    ax.set_title(f"{title_prefix}Cost boxplot across scenarios", fontsize=9)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "aggregate_cost_boxplot.png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    # Pooled histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(all_costs, bins=30, color="#56B4E9", alpha=0.75, edgecolor="white")
    ax.set_xlabel("Total cost (technician-seconds)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(
        f"{title_prefix}Pooled cost histogram ({len(all_costs)} traj., {len(scenarios)} scenarios)",
        fontsize=9,
    )
    ax.grid(axis="y", linewidth=0.4, alpha=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "aggregate_cost_histogram.png", bbox_inches="tight", dpi=100)
    plt.close(fig)
