"""
AdaptiveEvaluationProtocol — implements the Specification v4 protocol.

Execution flow per scenario:
  1. Load or initialise checkpoint.
  2. Collect BATCH_SIZE trajectories via subprocess.
  3. Cluster at intent and execution levels.
  4. Bootstrap noise floor; compute inter-batch ARI.
  5. Update convergence streaks.
  6. Compute numerical metrics (informational, not used for stopping).
  7. Save checkpoint; emit iteration report.
  8. Repeat until converged, hard-capped, or user stops.
  9. Post-convergence: label clusters; emit final report.

Scenarios are processed in parallel (one thread per scenario). A shared
semaphore limits total concurrent subprocesses across all scenario threads.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time

import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from Evaluation.checkpoint import load_checkpoint, save_checkpoint
from Evaluation.clustering import (
    cluster_execution, cluster_intent, embed_intent, _hdbscan_on_dist,
    _normalized_levenshtein, execution_sequence, intent_sequence,
    label_clusters_llm, save_dendrogram, save_intent_scatter,
)
from Evaluation.metrics import (
    ari_inter as compute_ari_inter,
    bootstrap_ari_noise_floor,
    compute_numerical_metrics,
)
from Evaluation.report import format_final_report, format_iteration_report
from Evaluation.qualitative import QualitativeReport, run_qualitative_analysis, summarise_qualitative_run
from Implementations.fault_injections import SCENARIOS


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ProtocolConfig:
    # Spec constants
    batch_size: int = 10
    initial_batches: int = 1
    bootstrap_samples: int = 100
    intra_batch_ari_percentile: float = 0.50
    ari_min_threshold: float = 0.30
    convergence_window: int = 1
    ci_rel_width_threshold: float = 0.10
    hdbscan_min_cluster_size: int = 3
    hdbscan_min_samples: int = 1
    agglom_linkage: str = "average"
    dendrogram_cut_grid: list[float] = field(
        default_factory=lambda: [i / 20 for i in range(1, 20)]
    )
    max_batches_per_scenario: Optional[int] = None
    allow_constant_batches: bool = False

    # Execution control
    interactive: bool = True
    checkpoint_dir: Path = field(default_factory=lambda: Path("Logs/Evaluation"))
    protocol_run_id: str = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%dT%H%M%S")
    )

    # Runner config (passed to run_diagnostic_scenario subprocesses)
    assistant_type: str = "FixedRandomTrajectories"
    assistant_config: dict = field(default_factory=dict)
    service_config:   dict = field(default_factory=dict)
    saboteur_config:  dict = field(default_factory=dict)
    service_type: str = "SpiceSimMockNL"
    rounds: int = 10
    ignore_max_rounds_allowance: bool = False
    base_dir: Path = field(default_factory=lambda: Path("."))
    trajectory_base_dir: Path = field(
        default_factory=lambda: Path("Logs/Trajectories")
    )

    # Mocking
    mock_llm_labels: bool = False
    mock_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"
    labeling_model: str = "gpt-4.1"

    # Qualitative analysis (Section 8)
    run_qualitative: bool = True
    qualitative_model: str = "gpt-4.1"

    # Parallelism — defaults to CPU count; lower for RAM-constrained machines,
    # set to ~10 for real LLM runs to stay within OpenAI rate limits.
    max_concurrent_subprocesses: Optional[int] = field(
        default_factory=lambda: __import__("os").cpu_count() or 4
    )


# ---------------------------------------------------------------------------
# Per-scenario state
# ---------------------------------------------------------------------------

@dataclass
class ScenarioState:
    scenario_number: int
    batch_index: int = 0
    trajectory_paths: list[str] = field(default_factory=list)
    cluster_assignments_intent: list[int] = field(default_factory=list)
    cluster_assignments_execution: list[int] = field(default_factory=list)
    streak_intent: int = 0
    streak_execution: int = 0
    converged: bool = False
    stopped: bool = False
    error: bool = False
    batch_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "scenario_number": self.scenario_number,
            "batch_index": self.batch_index,
            "n_trajectories": len(self.trajectory_paths),
            "trajectory_paths": self.trajectory_paths,
            "cluster_assignments_intent": self.cluster_assignments_intent,
            "cluster_assignments_execution": self.cluster_assignments_execution,
            "streak_intent": self.streak_intent,
            "streak_execution": self.streak_execution,
            "converged": self.converged,
            "stopped": self.stopped,
            "error": self.error,
            "batch_history": self.batch_history,
        }

    @staticmethod
    def from_dict(d: dict) -> "ScenarioState":
        s = ScenarioState(scenario_number=d["scenario_number"])
        s.batch_index = d.get("batch_index", 0)
        s.trajectory_paths = d.get("trajectory_paths", [])
        s.cluster_assignments_intent = d.get("cluster_assignments_intent", [])
        s.cluster_assignments_execution = d.get("cluster_assignments_execution", [])
        s.streak_intent = d.get("streak_intent", 0)
        s.streak_execution = d.get("streak_execution", 0)
        s.converged = d.get("converged", False)
        s.stopped = d.get("stopped", False)
        s.error = d.get("error", False)
        s.batch_history = d.get("batch_history", [])
        return s


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class AdaptiveEvaluationProtocol:

    def __init__(self, config: ProtocolConfig) -> None:
        self.config = config
        self._print_lock = threading.Lock()
        self._logger: Optional[logging.Logger] = None

    # ------------------------------------------------------------------ #
    # Logging setup
    # ------------------------------------------------------------------ #

    def _setup_logging(self) -> None:
        run_dir = (
            self.config.checkpoint_dir
            / self.config.assistant_type
            / self.config.protocol_run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)

        log_path = run_dir / "protocol.log"
        logger = logging.getLogger(f"protocol.{self.config.protocol_run_id}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)

        logger.addHandler(fh)
        logger.addHandler(sh)
        self._logger = logger

    def _log(self, msg: str) -> None:
        if self._logger:
            self._logger.info(msg)
        else:
            with self._print_lock:
                print(msg)

    def _log_system_specs(self) -> None:
        import platform
        import sys
        import importlib.metadata as meta

        cpu = platform.processor() or platform.machine()
        try:
            import psutil
            cores_physical = psutil.cpu_count(logical=False)
            cores_logical  = psutil.cpu_count(logical=True)
            ram_gb = round(psutil.virtual_memory().total / 1e9, 1)
            mem_line = f"RAM: {ram_gb} GB"
            cpu_line = f"CPU: {cpu} ({cores_physical} physical / {cores_logical} logical cores)"
        except ImportError:
            import os
            cpu_line = f"CPU: {cpu} ({os.cpu_count()} logical cores)"
            mem_line = "RAM: unknown (psutil not installed)"

        packages = [
            "openai", "rdflib", "owlready2", "hdbscan",
            "scikit-learn", "scipy", "numpy", "sentence-transformers",
        ]
        pkg_versions = []
        for p in packages:
            try:
                pkg_versions.append(f"{p}=={meta.version(p)}")
            except meta.PackageNotFoundError:
                pkg_versions.append(f"{p}==(not found)")

        self._log(
            f"System specs | OS: {platform.system()} {platform.release()} "
            f"({platform.version()}) | {cpu_line} | {mem_line} | "
            f"Python: {sys.version.split()[0]} | "
            + " | ".join(pkg_versions)
        )

    def _save_config(self) -> None:
        run_dir = (
            self.config.checkpoint_dir
            / self.config.assistant_type
            / self.config.protocol_run_id
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config.json"

        def _serialise(v):
            if isinstance(v, Path):
                return str(v)
            return v

        cfg_dict = {k: _serialise(v) for k, v in asdict(self.config).items()}
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def run(self, scenario_numbers: list[int]) -> None:
        """Run the protocol for all scenario numbers in parallel."""
        self._setup_logging()
        self._save_config()

        self._log(f"\nAdaptive Evaluation Protocol — run_id={self.config.protocol_run_id}")
        self._log(f"Scenarios: {scenario_numbers}")
        self._log(f"Batch size: {self.config.batch_size} | Window: {self.config.convergence_window}")
        self._log_system_specs()

        # Shared semaphore across all scenario threads
        sem = (
            threading.Semaphore(self.config.max_concurrent_subprocesses)
            if self.config.max_concurrent_subprocesses
            else None
        )
        self._subprocess_semaphore = sem

        t0 = time.perf_counter()
        threads = [
            threading.Thread(target=self._run_scenario, args=(num,), daemon=True)
            for num in scenario_numbers
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        elapsed = time.perf_counter() - t0
        self._log(f"\nProtocol complete. Total wall-clock time: {_fmt_duration(elapsed)}")

        # Run-level qualitative prose summary across all scenarios
        if self.config.run_qualitative and not self.config.mock_llm_labels:
            self._run_qualitative_summary()

    # ------------------------------------------------------------------ #
    # Per-scenario loop
    # ------------------------------------------------------------------ #

    def _run_scenario(self, scenario_number: int) -> None:
        scenario = next((s for s in SCENARIOS if s.number == scenario_number), None)
        if scenario is None:
            self._log(f"[Scenario {scenario_number}] Not found in SCENARIOS. Skipping.")
            return
        if scenario.fault_fns is None:
            self._log(f"[Scenario {scenario_number}] No simulation support (fault_fns=None). Skipping.")
            return

        ckpt_path = (
            self.config.checkpoint_dir
            / self.config.assistant_type
            / self.config.protocol_run_id
            / str(scenario_number)
            / "checkpoint.json"
        )
        state = self._load_or_init(ckpt_path, scenario_number)

        if state.converged or state.stopped or state.error:
            status = "converged" if state.converged else ("error" if state.error else "stopped")
            self._log(f"[Scenario {scenario_number}] Already {status}. Running post-convergence.")
            self._post_convergence(state, scenario_number, ckpt_path)
            return

        self._log(f"\n{'─'*60}")
        self._log(f"[Scenario {scenario_number}] {scenario.scenario_id} | {scenario.system_name}")
        _scenario_t0 = time.perf_counter()

        # ── Initial phase ──────────────────────────────────────────────
        if state.batch_index == 0:
            for _ in range(self.config.initial_batches):
                self._collect_and_process(state, scenario_number, ckpt_path)
                if state.stopped or state.error:
                    return

        # ── Iterative phase ────────────────────────────────────────────
        while not state.converged:
            cap = self.config.max_batches_per_scenario
            if cap is not None and state.batch_index >= cap:
                self._log(f"[Scenario {scenario_number}] Hard cap reached ({cap} batches).")
                state.stopped = True
                save_checkpoint(ckpt_path, state.to_dict())
                break

            if self.config.interactive:
                action = self._prompt_approval(state, scenario_number)
                if action == "n":
                    self._log("Protocol stopped by user.")
                    state.stopped = True
                    save_checkpoint(ckpt_path, state.to_dict())
                    return
                elif action == "stop":
                    state.stopped = True
                    save_checkpoint(ckpt_path, state.to_dict())
                    break
                elif action == "skip":
                    state.converged = True
                    save_checkpoint(ckpt_path, state.to_dict())
                    break

            self._collect_and_process(state, scenario_number, ckpt_path)
            if state.error:
                return

        self._post_convergence(state, scenario_number, ckpt_path)
        self._log(
            f"[Scenario {scenario_number}] Done in "
            f"{_fmt_duration(time.perf_counter() - _scenario_t0)}"
        )

    # ------------------------------------------------------------------ #
    # Collect + process one batch
    # ------------------------------------------------------------------ #

    def _collect_and_process(
        self, state: ScenarioState, scenario_number: int, ckpt_path: Path
    ) -> None:
        state.batch_index += 1
        _batch_t0 = time.perf_counter()
        self._log(f"[Scenario {scenario_number}] Collecting batch {state.batch_index}…")

        new_paths = self._collect_batch(scenario_number)

        # Error detection: if no trajectories collected, mark scenario as errored
        if not new_paths:
            self._log(
                f"[Scenario {scenario_number}] ERROR — batch {state.batch_index} produced 0 "
                f"trajectory files. Subprocess(es) likely crashed. Stopping this scenario."
            )
            state.error = True
            state.batch_history.append({
                "batch_index": state.batch_index,
                "n_trajectories": 0,
                "recommendation": "ERROR",
                "error": "No trajectory files produced — check subprocess logs.",
            })
            save_checkpoint(ckpt_path, state.to_dict())
            return

        state.trajectory_paths.extend([str(p) for p in new_paths])

        trajectories = self._load_trajectories(state.trajectory_paths)
        n = len(trajectories)

        # Build sequences
        intent_seqs = [intent_sequence(t) for t in trajectories]
        exec_seqs   = [execution_sequence(t) for t in trajectories]

        # Cluster
        prev_intent = list(state.cluster_assignments_intent)
        prev_exec   = list(state.cluster_assignments_execution)

        # Embed once; reused for main clustering and bootstrap below
        intent_dist_matrix, _ = embed_intent(
            intent_seqs,
            embedding_model=self.config.embedding_model,
            mock_embeddings=self.config.mock_embeddings,
        )
        new_intent_labels, intent_probs = _hdbscan_on_dist(
            intent_dist_matrix,
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
        )
        new_exec_labels, _, _, _ = cluster_execution(
            exec_seqs,
            linkage=self.config.agglom_linkage,
            cut_grid=self.config.dendrogram_cut_grid,
        )

        state.cluster_assignments_intent   = new_intent_labels
        state.cluster_assignments_execution = new_exec_labels

        n_clusters_intent    = len(set(l for l in new_intent_labels if l >= 0))
        n_clusters_execution = len(set(new_exec_labels))

        # Constant-behavior early stop: if all trajectories are identical on both
        # levels, force convergence immediately without running more batches.
        # Use raw sequence equality — cluster counts can be misleading (e.g.
        # HDBSCAN returns 0 clusters on random mock embeddings even when all
        # intent strings are identical).
        if not self.config.allow_constant_batches:
            all_exec_identical   = len(set(exec_seqs)) == 1
            all_intent_identical = len(set(intent_seqs)) == 1
            if all_exec_identical and all_intent_identical:
                self._log(
                    f"[Scenario {scenario_number}] Constant behavior detected across all "
                    f"trajectories — converging immediately."
                )
                state.streak_intent    = self.config.convergence_window
                state.streak_execution = self.config.convergence_window

        # Build execution distance matrix once — reused by all bootstrap resamplings
        exec_dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = _normalized_levenshtein(exec_seqs[i], exec_seqs[j])
                exec_dist_matrix[i, j] = d
                exec_dist_matrix[j, i] = d

        # Bootstrap noise floor (full bootstrap resampling with replacement)
        # dist_matrix is passed explicitly so the data dependency is visible at the call site.
        def _intent_recluster(indices, dist_matrix):
            sub = dist_matrix[np.ix_(indices, indices)]
            labels_r, _ = _hdbscan_on_dist(
                sub,
                min_cluster_size=self.config.hdbscan_min_cluster_size,
                min_samples=self.config.hdbscan_min_samples,
            )
            return list(labels_r)

        def _exec_recluster(indices, dist_matrix):
            from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster
            from scipy.spatial.distance import squareform
            from sklearn.metrics import silhouette_score
            sub = dist_matrix[np.ix_(indices, indices)]
            condensed = squareform(sub)
            if condensed.shape[0] == 0:
                return [0] * len(indices)
            Z_r = scipy_linkage(condensed, method=self.config.agglom_linkage)
            cut_grid = self.config.dendrogram_cut_grid or [i / 20 for i in range(1, 20)]
            best_labels = np.zeros(len(indices), dtype=int)
            best_score = -2.0
            for height in cut_grid:
                lbls = fcluster(Z_r, t=height, criterion="distance") - 1
                if len(set(lbls)) < 2:
                    continue
                counts = np.bincount(lbls)
                score = 0.0 if all(c == 1 for c in counts) else silhouette_score(sub, lbls, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_labels = lbls
            return best_labels.tolist()

        from functools import partial
        ari_boot_noise_floor_intent = bootstrap_ari_noise_floor(
            new_intent_labels, self.config.bootstrap_samples,
            self.config.intra_batch_ari_percentile,
            partial(_intent_recluster, dist_matrix=intent_dist_matrix),
        )
        ari_boot_noise_floor_exec = bootstrap_ari_noise_floor(
            new_exec_labels, self.config.bootstrap_samples,
            self.config.intra_batch_ari_percentile,
            partial(_exec_recluster, dist_matrix=exec_dist_matrix),
        )

        # Inter-batch ARI
        ari_inter_intent:    Optional[float] = None
        ari_inter_execution: Optional[float] = None

        if prev_intent and state.batch_index > 1:
            shared = list(range(min(len(prev_intent), len(new_intent_labels))))
            ari_inter_intent = compute_ari_inter(prev_intent, new_intent_labels, shared)

        if prev_exec and state.batch_index > 1:
            shared = list(range(min(len(prev_exec), len(new_exec_labels))))
            ari_inter_execution = compute_ari_inter(prev_exec, new_exec_labels, shared)

        # Convergence check — guard: n_clusters_intent must be >= 1
        # Dual criterion: (1) beat bootstrap noise floor, (2) beat absolute minimum ARI
        converged_intent = (
            n_clusters_intent >= 1
            and ari_inter_intent is not None
            and ari_inter_intent >= ari_boot_noise_floor_intent
            and ari_inter_intent >= self.config.ari_min_threshold
        )
        converged_execution = (
            ari_inter_execution is not None
            and ari_inter_execution >= ari_boot_noise_floor_exec
            and ari_inter_execution >= self.config.ari_min_threshold
        )

        if converged_intent:
            state.streak_intent += 1
        else:
            state.streak_intent = 0

        if converged_execution:
            state.streak_execution += 1
        else:
            state.streak_execution = 0

        if (state.streak_intent  >= self.config.convergence_window and
                state.streak_execution >= self.config.convergence_window):
            state.converged = True

        # Numerical metrics (informational)
        numerical = compute_numerical_metrics(trajectories, self.config.bootstrap_samples)

        # Determine recommendation
        if state.converged:
            recommendation = "CONVERGED"
        elif (self.config.max_batches_per_scenario is not None and
              state.batch_index >= self.config.max_batches_per_scenario):
            recommendation = "HARD_CAPPED"
        else:
            recommendation = "CONTINUE"
        if n_clusters_intent == 0:
            recommendation += " (intent: all noise — embeddings may lack signal)"

        # Record batch history
        batch_record: dict = {
            "batch_index": state.batch_index,
            "batch_elapsed_s": round(time.perf_counter() - _batch_t0, 1),
            "n_trajectories": n,
            "n_clusters_intent": n_clusters_intent,
            "n_clusters_execution": n_clusters_execution,
            "ari_inter_intent": ari_inter_intent,
            "ari_inter_execution": ari_inter_execution,
            "ari_boot_noise_floor_intent": ari_boot_noise_floor_intent,
            "ari_boot_noise_floor_execution": ari_boot_noise_floor_exec,
            "ari_min_threshold": self.config.ari_min_threshold,
            "converged_intent": converged_intent,
            "converged_execution": converged_execution,
            "streak_intent": state.streak_intent,
            "streak_execution": state.streak_execution,
            "numerical_metrics": numerical,
            "recommendation": recommendation,
        }
        state.batch_history.append(batch_record)
        save_checkpoint(ckpt_path, state.to_dict())

        # Emit iteration report
        report = format_iteration_report(
            scenario_number=scenario_number,
            batch_index=state.batch_index,
            n_trajectories=n,
            n_clusters_intent=n_clusters_intent,
            n_clusters_execution=n_clusters_execution,
            ari_inter_intent=ari_inter_intent,
            ari_inter_execution=ari_inter_execution,
            ari_boot_noise_floor_intent=ari_boot_noise_floor_intent,
            ari_boot_noise_floor_execution=ari_boot_noise_floor_exec,
            converged_intent=converged_intent,
            converged_execution=converged_execution,
            streak_intent=state.streak_intent,
            streak_execution=state.streak_execution,
            convergence_window=self.config.convergence_window,
            ari_min_threshold=self.config.ari_min_threshold,
            numerical_metrics=numerical,
            recommendation=recommendation,
        )
        self._log(report)

    # ------------------------------------------------------------------ #
    # Post-convergence (cluster labeling + final report)
    # ------------------------------------------------------------------ #

    def _run_qualitative_summary(self) -> None:
        """Load all per-scenario qualitative.json files and write a prose summary."""
        run_dir = (
            self.config.checkpoint_dir
            / self.config.assistant_type
            / self.config.protocol_run_id
        )
        qual_data: dict[int, dict] = {}
        for p in sorted(run_dir.rglob("qualitative.json")):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                num = d.get("scenario_number")
                if num is not None:
                    qual_data[int(num)] = d
            except Exception:
                continue

        if not qual_data:
            self._log("\n[Qualitative summary] No qualitative.json files found — skipping.")
            return

        self._log(f"\n[Qualitative summary] Summarising {len(qual_data)} scenarios with {self.config.qualitative_model}…")

        # Build a lightweight scenario_meta lookup from SCENARIOS
        _meta_cache: dict[int, dict] = {}
        for s in SCENARIOS:
            tags_raw = getattr(s, "category", "") or ""
            tags = [t.strip() for t in tags_raw.split(",") if t.strip()] if tags_raw else ["Simple"]
            _meta_cache[s.number] = {
                "system":        s.system_name,
                "fault_tags":    tags,
                "injected_fault": s.root_cause.root_cause_description_proper if s.root_cause else "",
            }

        def _meta(num: int) -> dict:
            return _meta_cache.get(num, {"system": "?", "fault_tags": [], "injected_fault": "?"})

        try:
            summary = summarise_qualitative_run(
                qual_data=qual_data,
                model=self.config.qualitative_model,
                scenario_meta_fn=_meta,
            )
        except Exception as exc:
            summary = f"(qualitative summary error: {exc})"

        self._log(f"\n[Qualitative summary]\n{summary}")
        out_path = run_dir / "qualitative_summary.txt"
        out_path.write_text(summary + "\n", encoding="utf-8")
        self._log(f"[Qualitative summary] Written to {out_path}")

    def _post_convergence(
        self, state: ScenarioState, scenario_number: int, ckpt_path: Path
    ) -> None:
        if not state.batch_history:
            return

        # Error state: brief report only
        if state.error:
            self._log(
                f"[Scenario {scenario_number}] ERROR — no valid trajectories collected. "
                f"Check subprocess logs."
            )
            return

        trajectories = self._load_trajectories(state.trajectory_paths)
        intent_seqs  = [intent_sequence(t) for t in trajectories]

        cluster_labels: Optional[dict[int, str]] = None
        if state.cluster_assignments_intent:
            cluster_labels = label_clusters_llm(
                intent_seqs,
                state.cluster_assignments_intent,
                probabilities=None,
                mock=self.config.mock_llm_labels,
                labeling_model=self.config.labeling_model,
            )

        last = state.batch_history[-1]
        numerical = last.get("numerical_metrics", {})

        # Section 8: qualitative analysis
        qual_report: Optional[QualitativeReport] = None
        if self.config.run_qualitative and not self.config.mock_llm_labels:
            try:
                qual_report = run_qualitative_analysis(
                    scenario_number=scenario_number,
                    trajectories=trajectories,
                    cluster_assignments_intent=state.cluster_assignments_intent or [],
                    cluster_probabilities=None,
                    cluster_labels=cluster_labels or {},
                    model=self.config.qualitative_model,
                    output_dir=ckpt_path.parent,
                )
            except Exception as exc:
                self._log(f"[Scenario {scenario_number}] Warning: qualitative analysis failed: {exc}")

        reason = (
            "converged" if state.converged
            else ("error" if state.error else ("stopped" if state.stopped else "unknown"))
        )
        report = format_final_report(
            scenario_number=scenario_number,
            total_batches=state.batch_index,
            total_trajectories=len(state.trajectory_paths),
            n_clusters_intent=last.get("n_clusters_intent", 0),
            n_clusters_execution=last.get("n_clusters_execution", 0),
            ari_inter_intent=last.get("ari_inter_intent"),
            ari_inter_execution=last.get("ari_inter_execution"),
            ari_boot_noise_floor_intent=last.get("ari_boot_noise_floor_intent") or last.get("ari_boot_p05_intent", 0.0),
            ari_boot_noise_floor_execution=last.get("ari_boot_noise_floor_execution") or last.get("ari_boot_p05_execution", 0.0),
            cluster_labels_intent=cluster_labels,
            numerical_metrics=numerical,
            batch_history=state.batch_history,
            reason=reason,
            qualitative_report=qual_report,
        )
        self._log(report)

        # Write per-scenario final report to file
        report_path = ckpt_path.parent / "final_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report + "\n")

        # Save execution-level dendrogram over all collected trajectories
        try:
            from scipy.cluster.hierarchy import linkage as scipy_linkage
            from scipy.spatial.distance import squareform
            exec_seqs = [execution_sequence(t) for t in trajectories]
            _, Z_exec, cut_exec, clustered_exec = cluster_execution(
                exec_seqs,
                linkage=self.config.agglom_linkage,
                cut_grid=self.config.dendrogram_cut_grid,
            )
            if Z_exec is None and clustered_exec:
                # All sequences identical — build a trivial zero-height linkage
                m = len(clustered_exec)
                zero_condensed = squareform(np.zeros((m, m)))
                Z_exec = scipy_linkage(zero_condensed, method=self.config.agglom_linkage)
                cut_exec = 0.0
            if cut_exec is None:
                cut_exec = 0.0
            if Z_exec is not None:
                n_clusters_exec = last.get("n_clusters_execution", "?")
                save_dendrogram(
                    Z=Z_exec,
                    clustered_seqs=clustered_exec,
                    best_cut_height=cut_exec,
                    output_path=ckpt_path.parent / "dendrogram.png",
                    title=(
                        f"Scenario {scenario_number} | Execution clustering "
                        f"({len(trajectories)} trajectories, {n_clusters_exec} clusters)"
                    ),
                )
        except Exception as exc:
            self._log(f"[Scenario {scenario_number}] Warning: dendrogram save failed: {exc}")

        # Save intent cluster scatter plot (PCA 2D)
        try:
            from sklearn.decomposition import PCA
            _, embeddings_intent = embed_intent(
                intent_seqs,
                embedding_model=self.config.embedding_model,
                mock_embeddings=self.config.mock_embeddings,
            )
            if np.allclose(embeddings_intent, embeddings_intent[0]):
                coords = np.zeros((len(embeddings_intent), 2))
            else:
                coords = PCA(n_components=2).fit_transform(embeddings_intent)
            save_intent_scatter(
                embeddings_2d=coords,
                labels=state.cluster_assignments_intent,
                intent_seqs=intent_seqs,
                cluster_label_names=cluster_labels,
                output_path=ckpt_path.parent / "intent_clusters.png",
                title=(
                    f"Scenario {scenario_number} | Intent clustering "
                    f"({len(trajectories)} trajectories, "
                    f"{last.get('n_clusters_intent', '?')} clusters)"
                ),
            )
        except Exception as exc:
            self._log(f"[Scenario {scenario_number}] Warning: intent scatter save failed: {exc}")

    # ------------------------------------------------------------------ #
    # Batch collection (subprocess)
    # ------------------------------------------------------------------ #

    def _collect_batch(self, scenario_number: int) -> list[Path]:
        """Run BATCH_SIZE scenario subprocesses and return paths to produced trajectory JSONs."""
        from run_many_scenarios import run_scenario_multiple_times

        scenario = next(s for s in SCENARIOS if s.number == scenario_number)
        system_name = scenario.system_name

        # Taxonomy: Logs/<type>/<assistant_type>/<scenario_number>/
        traj_dir = (
            self.config.trajectory_base_dir
            / self.config.assistant_type
            / str(scenario_number)
        )
        log_dir = (
            self.config.base_dir / "Logs" / "DebuggingLogs"
            / self.config.assistant_type
            / str(scenario_number)
        )
        chat_dir = (
            self.config.base_dir / "Logs" / "Chats"
            / self.config.assistant_type
            / str(scenario_number)
        )
        for d in (traj_dir, log_dir, chat_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Snapshot existing JSONs before running
        existing = set(traj_dir.glob("*_TRAJECTORY.json"))

        # Compute effective rounds: minimum of config rounds and CSV allowance
        effective_rounds = self.config.rounds
        if not self.config.ignore_max_rounds_allowance:
            scenario = next((s for s in SCENARIOS if s.number == scenario_number), None)
            if scenario and hasattr(scenario, 'min_round_number_allowance'):
                csv_allowance = scenario.min_round_number_allowance
                if csv_allowance is not None:
                    effective_rounds = min(self.config.rounds, csv_allowance)

        run_scenario_multiple_times(
            n_runs=self.config.batch_size,
            base_dir=self.config.base_dir,
            forced_scenario=scenario_number,
            assistant=self.config.assistant_type,
            rounds=effective_rounds,
            system=system_name,
            service=self.config.service_type,
            saboteur="SpiceSim",
            assistant_config=self.config.assistant_config,
            service_config=self.config.service_config,
            saboteur_config=self.config.saboteur_config,
            log_path=str(log_dir),
            chat_path=str(chat_dir),
            trajectory_path=str(traj_dir),
            batch_size_of_same_scenario_runs=self.config.batch_size,
            semaphore=self._subprocess_semaphore,
        )

        all_now = set(traj_dir.glob("*_TRAJECTORY.json"))
        new_files = sorted(all_now - existing, key=lambda p: p.name)
        return new_files

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _load_or_init(self, ckpt_path: Path, scenario_number: int) -> ScenarioState:
        data = load_checkpoint(ckpt_path)
        if data is not None:
            self._log(
                f"[Scenario {scenario_number}] Resuming from checkpoint "
                f"(batch {data.get('batch_index', 0)})."
            )
            return ScenarioState.from_dict(data)
        return ScenarioState(scenario_number=scenario_number)

    @staticmethod
    def _load_trajectories(paths: list[str]) -> list[dict]:
        trajectories = []
        for p in paths:
            try:
                with open(p, encoding="utf-8") as f:
                    trajectories.append(json.load(f))
            except Exception:
                pass
        return trajectories

    def _prompt_approval(self, state: ScenarioState, scenario_number: int) -> str:
        last = state.batch_history[-1] if state.batch_history else {}
        ari_i = last.get("ari_inter_intent")
        ari_e = last.get("ari_inter_execution")
        floor_i = last.get("ari_boot_noise_floor_intent") or last.get("ari_boot_p05_intent", 0.0)
        floor_e = last.get("ari_boot_noise_floor_execution") or last.get("ari_boot_p05_execution", 0.0)
        with self._print_lock:
            print(
                f"\n[Scenario {scenario_number} | Batch {state.batch_index} → {state.batch_index+1} | "
                f"Streak intent={state.streak_intent} execution={state.streak_execution}]"
            )
            print(
                f"  ARI: intent={_fmt(ari_i)} (floor={floor_i:.3f})  "
                f"execution={_fmt(ari_e)} (floor={floor_e:.3f})"
            )
            while True:
                ans = input("  Proceed with next batch? [y/n/stop/skip] ").strip().lower()
                if ans in ("y", "n", "stop", "skip"):
                    return ans
                print("  Please enter y, n, stop, or skip.")


def _fmt(v) -> str:
    return f"{v:.3f}" if v is not None else "N/A"


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration: e.g. '2h 03m 07s', '4m 22s', '18s'."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"
