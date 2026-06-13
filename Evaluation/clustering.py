"""
Trajectory clustering for the adaptive evaluation protocol.

Intent level  — HDBSCAN on sentence-transformer embeddings of NL action sequences.
Execution level — Agglomerative clustering on normalized Levenshtein distance of
                  symbolic action sequences (action_id + targets).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# model used to label clusters
LABELING_MODEL = "gpt-4.1"

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
        model = SentenceTransformer(embedding_model)
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


def _abbrev_intent(seq: str, n: int = 2) -> str:
    """Abbreviate an intent sequence string to head+tail action_ids."""
    parts = [tok for tok in seq.split(" → ")]
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

    cmap = plt.cm.tab10
    cluster_ids = sorted(set(l for l in labels if l >= 0))

    def _color(label: int):
        if label < 0:
            return "#aaaaaa"
        return cmap(label % 10)

    fig, ax = plt.subplots(figsize=(10, 7))
    xs, ys = embeddings_2d[:, 0], embeddings_2d[:, 1]

    for i, (x, y, lbl) in enumerate(zip(xs, ys, labels)):
        ax.scatter(x, y, c=[_color(lbl)], s=40, alpha=0.8, zorder=2)
        text = _abbrev_intent(intent_seqs[i])
        ax.annotate(text, (x, y), fontsize=5, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points")

    # Legend
    patches = []
    for cid in cluster_ids:
        name = (cluster_label_names or {}).get(cid, f"Cluster {cid}")
        patches.append(mpatches.Patch(color=cmap(cid % 10), label=f"[{cid}] {name}"))
    if any(l < 0 for l in labels):
        patches.append(mpatches.Patch(color="#aaaaaa", label="noise"))
    if patches:
        ax.legend(handles=patches, fontsize=7, loc="best")

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
