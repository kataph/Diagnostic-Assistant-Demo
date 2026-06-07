# Adaptive Evaluation Protocol for Diagnostic LLM Agents
## Specification v4

---

# 0. Global Constants

All constants defined at initialization time, overridable via YAML/JSON config.

| Constant | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 10 | Trajectories collected per iteration per scenario type |
| `INITIAL_BATCHES` | 1 | Batches collected before first evaluation |
| `BOOTSTRAP_SAMPLES` | 1000 | Resamples for intra-batch ARI and numerical CI estimation |
| `INTRA_BATCH_ARI_PERCENTILE` | 0.05 | Lower percentile of bootstrap ARI distribution used as noise floor |
| `CONVERGENCE_WINDOW` | 1 | Consecutive batches passing threshold required for convergence |
| `CI_REL_WIDTH_THRESHOLD` | 0.10 | Maximum relative CI width for numerical metrics (informational) |
| `CI_METHOD_SUCCESS_RATE` | "Wilson" | CI method for Bernoulli success rate |
| `CI_METHOD_OTHER` | "percentile_bootstrap" | CI method for continuous metrics |
| `HDBSCAN_MIN_CLUSTER_SIZE` | 3 | Min points to form a cluster (intent level) |
| `HDBSCAN_MIN_SAMPLES` | 1 | HDBSCAN min_samples (controls outlier sensitivity) |
| `AGGLOM_LINKAGE` | "average" | Linkage for execution-level agglomerative clustering |
| `DENDROGRAM_CUT_GRID` | [0.05..0.95, step 0.05] | Candidate cut heights for execution-level clustering |
| `MAX_BATCHES_PER_SCENARIO` | None | Hard cap on iterations (None = unlimited) |

---

# 1. Overview

The protocol is executed **independently per scenario type**. All clustering, stability tracking, and numerical metrics are local to one scenario type; no cross-scenario pooling occurs.

For each scenario type, trajectories are collected in fixed-size batches until behavioral stability is reached. Two independent clusterings are maintained:

- **Intent level**: what the agent intended to do — free-text action sequences, embedded and clustered with HDBSCAN
- **Execution level**: what the agent actually did — symbolic simulator action sequences, clustered via agglomerative clustering on normalized Levenshtein distance

Stability is assessed via **Adjusted Rand Index (ARI)** between successive clusterings, bootstrapped to distinguish true convergence from sampling noise. Numerical metrics are computed once after convergence.

---

# 2. Recorded Data

## 2.1 Metadata (per run)
- `scenario_type_id`
- `run_id`
- `timestamp`
- `model_config` (dict)
- `environment_config` (dict)

## 2.2 Trajectory (per run)
- `agent_observations`: list of strings
- `agent_free_text_actions`: list of strings
- `expanded_simulator_actions`: list of symbolic action tuples
- `simulator_observations`: list of strings
- `metrics_per_step`: list of dicts
- `outcome`: bool (success/failure)

## 2.3 Numerical Metrics (per run, scalars)
- `success`: bool
- `diagnostic_correctness`: float (defined externally, referenced by scenario spec)
- `total_cost`: float
- `n_actions`: int
- `n_tests`: int
- `n_repairs`: int

---

# 3. Clustering

Both clusterings are **recomputed from scratch on the full dataset** at the end of every batch.

## 3.1 Intent-Level Clustering (HDBSCAN)

**Representation**: serialize each trajectory as an ordered action string:
```
Action_1 → Action_2 → Action_3 → ...
```
Order is implicitly preserved by the embedding model's attention mechanism.

**Embedding**: sentence transformer (configured via `model_config`). One vector per trajectory. Pairwise cosine distance matrix computed over all trajectories.

**Algorithm**: HDBSCAN(`min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE`, `min_samples=HDBSCAN_MIN_SAMPLES`) on the cosine distance matrix.

**Output**:
- Cluster assignments (label `-1` = outlier/noise)
- Cluster membership probabilities
- `n_clusters_intent` (excluding noise label)

**Note**: in early batches (small n), HDBSCAN may assign many trajectories as noise. This is expected and does not indicate a problem — the ARI-based convergence criterion handles it gracefully.

**Labeling**: performed after convergence only. For each cluster, the trajectory with highest mean membership probability (most representative) is passed to an LLM to generate a short natural-language label.

---

## 3.2 Execution-Level Clustering (Agglomerative + Levenshtein)

**Deduplication**: extract the set of unique symbolic action sequences before clustering. Maintain a multiplicity count per unique sequence. Clustering is performed on unique sequences only; assignments are mapped back to all trajectories via sequence identity.

**Rationale**: repeated identical sequences would artificially anchor the dendrogram and distort cut-height selection.

**Distance**: normalized Levenshtein over unique sequences:
```
d(A, B) = Lev(A, B) / max(|A|, |B|)
```
This metric is order-sensitive by construction.

**Algorithm**: agglomerative hierarchical clustering (`linkage=AGGLOM_LINKAGE`) on the pairwise distance matrix of unique sequences.

**Cut selection**: for each height in `DENDROGRAM_CUT_GRID`, compute silhouette score over unique sequences. Select height maximizing silhouette score.

**Edge case**: if all unique sequences collapse to 1 cluster at all cut heights, assign label 0 to all trajectories and skip silhouette computation. Single-cluster outcome is valid.

**Output**:
- Cluster assignments (all trajectories, via multiplicity mapping)
- `n_clusters_execution`

---

# 4. ARI-Based Stability

ARI is permutation-invariant: no label-matching step is required.

Convention: ARI = 1 means identical clusterings; ARI ≈ 0 means agreement no better than chance.

## 4.1 Intra-Batch Stability (bootstrap noise floor)

Computed once per batch, per level, after reclustering.

**Procedure**:
1. Take the full current dataset (N trajectories).
2. Draw `BOOTSTRAP_SAMPLES` bootstrap resamples (with replacement, size N).
3. Recluster each resample using the same algorithm and parameters.
4. Compute ARI between each bootstrap clustering and the reference clustering (full dataset).
5. `ari_boot_p05 = percentile(ARI_values, INTRA_BATCH_ARI_PERCENTILE × 100)`

`ari_boot_p05` represents the noise floor: the minimum ARI expected under sampling variation alone. It is used directly as the convergence threshold.

---

## 4.2 Inter-Batch ARI (convergence signal)

Computed after each batch (starting from batch 2), per level.

**Procedure**:
1. `C_prev` = cluster assignments at end of previous batch, restricted to trajectories present in both batches.
2. `C_curr` = cluster assignments at end of current batch, same trajectories.
3. `ARI_inter = ARI(C_prev, C_curr)`

**Convergence condition (per level)**:
```
ARI_inter >= ari_boot_p05
```
Interpreted as: the inter-batch change is within the noise expected from sampling variation.

---

## 4.3 Overall Convergence

Convergence is declared for the scenario type when **both** levels satisfy their convergence condition for `CONVERGENCE_WINDOW` consecutive batches.

---

# 5. Per-Batch Recorded State

At the end of each batch, record:

| Field | Description |
|---|---|
| `batch_index` | Batch number (1-indexed) |
| `n_trajectories` | Total trajectories so far |
| `n_clusters_intent` | Number of HDBSCAN clusters (excl. noise) |
| `n_clusters_execution` | Number of execution-level clusters |
| `ari_inter_intent` | Inter-batch ARI, intent level (null for batch 1) |
| `ari_inter_execution` | Inter-batch ARI, execution level (null for batch 1) |
| `ari_boot_p05_intent` | Noise floor, intent level |
| `ari_boot_p05_execution` | Noise floor, execution level |
| `converged_intent` | Bool (false for batch 1) |
| `converged_execution` | Bool (false for batch 1) |
| `streak_intent` | Consecutive passing batches, intent level |
| `streak_execution` | Consecutive passing batches, execution level |

---

# 6. Iterative Execution Loop

```
for each scenario_type:

  dataset = []
  C_prev_intent, C_prev_execution = None, None
  streak_intent, streak_execution = 0, 0
  converged = False

  # --- Initial phase ---
  collect INITIAL_BATCHES × BATCH_SIZE trajectories → dataset
  recluster intent + execution on full dataset
  compute intra-batch bootstrap → ari_boot_p05 (per level)
  record batch state (ari_inter = null, converged = false)
  C_prev_intent, C_prev_execution = current clusterings

  # --- Iterative phase ---
  while not converged:
    if MAX_BATCHES_PER_SCENARIO and batch_count >= MAX_BATCHES_PER_SCENARIO:
      break

    collect BATCH_SIZE new trajectories → dataset

    recluster intent + execution on full dataset

    compute intra-batch bootstrap → ari_boot_p05 (per level)

    compute ARI_inter (per level) against C_prev

    for each level:
      if ARI_inter >= ari_boot_p05:
        streak += 1
      else:
        streak = 0

    if streak_intent >= CONVERGENCE_WINDOW
       and streak_execution >= CONVERGENCE_WINDOW:
      converged = True

    C_prev_intent, C_prev_execution = current clusterings

    emit iteration report (Section 9.1)

  # --- Post-convergence ---
  compute numerical metrics on full dataset (Section 7)
  run qualitative analysis (Section 8)
  emit final report (Section 9.2)
```

---

# 7. Numerical Metrics (post-convergence)

Computed **once** on the full converged dataset. Not computed per-batch.

## 7.1 Success Rate
```
n = total runs for this scenario type
k = successful runs
CI = Wilson(k, n, alpha=0.05)
```

## 7.2 Continuous Metrics
Applied to: `total_cost`, `n_actions`, `n_tests`, `n_repairs`, `diagnostic_correctness`.
```
for each metric m:
  boot_means = [mean(resample(values_m)) for _ in range(BOOTSTRAP_SAMPLES)]
  point_estimate = mean(values_m)
  CI = (percentile(boot_means, 2.5), percentile(boot_means, 97.5))
  rel_width = (CI[1] - CI[0]) / point_estimate
  if rel_width > CI_REL_WIDTH_THRESHOLD:
    flag as UNDERPOWERED  # informational only, does not block convergence
```

---

# 8. Qualitative Analysis Phase

Performed after convergence (or user-triggered stop).

## 8.1 Rubric Evaluation
For each intent-level cluster, select the most representative trajectory (highest mean HDBSCAN membership probability). Apply predefined rubric:

- Actionability
- Diagnostic coherence
- Efficiency
- Consistency
- Evidence usage

## 8.2 Gold Standard Comparison
Categorical comparison of representative trajectories against ideal diagnosis examples from scenario definitions. Fully qualitative; categorical output only.

## 8.3 Emergent Findings
LLM-assisted open-ended analysis across representative trajectories. Targets:
- Recurring failure modes
- Novel strategies not anticipated by the rubric
- Inefficiencies
- Candidate metrics for future rubric extension

---

# 9. Reports

## 9.1 Per-Iteration Report
All fields from Section 5, plus:
- Pass/fail per convergence criterion
- Recommendation: `CONTINUE` / `CONVERGED` / `HARD_CAPPED`

## 9.2 Final Report (per scenario type)
- Total batches run, total trajectories
- All numerical metrics: point estimate, 95% CI, relative width, underpowered flag
- `n_clusters_intent`, `n_clusters_execution` at convergence
- Intent cluster labels (LLM-generated) and sizes
- Execution cluster sizes
- `ARI_inter` and `ari_boot_p05` at convergence (both levels)
- Qualitative rubric scores per cluster
- Emergent findings