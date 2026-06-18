"""
Entry point for the Adaptive Evaluation Protocol (Specification v4).

Usage examples
--------------
# Test run with mocks (no LLM, no real embeddings, 1 scenario only)
python -m run_evaluation_protocol --scenarios 1 --assistant FixedRandomTrajectories --service SpiceSimMockNL --batch-size 10 --rounds 5 --mock-llm-labels --mock-embeddings --max-batches 3

# Test run with mocks (no LLM, no real embeddings)
python -m run_evaluation_protocol --scenarios 1,2,3 --assistant FixedRandomTrajectories --service SpiceSimMockNL --batch-size 3 --rounds 5 --mock-llm-labels --mock-embeddings --max-batches 3

# Test run with mocks, all scenarios
python -m run_evaluation_protocol --all-scenarios --assistant FixedRandomTrajectories --service SpiceSimMockNL --batch-size 3 --rounds 5 --mock-llm-labels --mock-embeddings --max-batches 3

# Resume a previous run
python -m run_evaluation_protocol --scenarios 1,2,3 --assistant FixedRandomTrajectories --service SpiceSimMockNL --batch-size 3 --rounds 5 --mock-llm-labels --mock-embeddings --max-batches 5 --resume 20260604T123456

# Real run, fully automated (default)
python -m run_evaluation_protocol --all-scenarios --assistant LLM --service SpiceSim --batch-size 10 --rounds 10

# Real run with manual approval gate before each batch
python -m run_evaluation_protocol --all-scenarios --assistant LLM --service SpiceSim --batch-size 10 --rounds 10 --interactive
"""
import argparse
import sys
from pathlib import Path

from Evaluation.protocol import AdaptiveEvaluationProtocol, ProtocolConfig
from Implementations.fault_injections import SCENARIOS


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Adaptive Evaluation Protocol on diagnostic assistant agents."
    )

    # Scenario selection
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--scenarios", type=str,
                     help="Comma-separated scenario numbers, e.g. '1,2,3,4,5'")
    grp.add_argument("--all-scenarios", action="store_true",
                     help="Run all simulatable scenarios (fault_fns is not None)")

    # Agent types
    parser.add_argument("--assistant", type=str, default="FixedRandomTrajectories",
                        help="Assistant type: FixedRandomTrajectories | RandomSearch | Unhelpful | LLM | EvidenceKGOptimal")
    parser.add_argument("--service", type=str, default="SpiceSimMockNL",
                        help="Service agent type: SpiceSimMockNL | SpiceSim")
    parser.add_argument("--assistant-config", type=str, default="{}",
                        help="JSON config for the assistant agent, e.g. '{\"model\":\"gpt-4.1\"}'")
    parser.add_argument("--service-config", type=str, default="{}",
                        help="JSON config for the service agent")
    parser.add_argument("--saboteur-config", type=str, default="{}",
                        help="JSON config for the saboteur agent")

    # Protocol parameters
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--initial-batches", type=int, default=1)
    parser.add_argument("--bootstrap-samples", type=int, default=100)
    parser.add_argument("--ari-percentile", type=float, default=0.50,
                        help="Percentile of the bootstrap ARI noise floor used as convergence "
                             "threshold (default: 0.50 = median).")
    parser.add_argument("--ari-min-threshold", type=float, default=0.30,
                        help="Absolute minimum inter-batch ARI required for convergence, "
                             "in addition to exceeding the bootstrap noise floor (default: 0.30).")
    parser.add_argument("--convergence-window", type=int, default=1)
    parser.add_argument("--max-batches", type=int, default=None,
                        help="Hard cap on batches per scenario (default: unlimited)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Max diagnostic rounds per scenario run")
    parser.add_argument("--ignore-max-rounds-allowance", action="store_true", default=False,
                        help="Ignore the 'minimum round number allowance' column from SCENARIOS_MASTER.csv "
                             "(restores previous behavior where --rounds is used as absolute max)")
    import os
    _default_concurrent = os.cpu_count() or 4
    parser.add_argument("--max-concurrent", type=int, default=_default_concurrent,
                        help=f"Max concurrent subprocesses across all scenario threads "
                             f"(default: number of CPU cores = {_default_concurrent}). "
                             f"Lower this if you run out of RAM; raise it if cores are idle. "
                             f"For real LLM runs set to ~10 to stay within OpenAI rate limits.")

    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default="Logs/Evaluation",
                        help="Root directory for checkpoints and per-run logs")
    parser.add_argument("--trajectory-dir", type=str, default="Logs/Trajectories",
                        help="Root directory where trajectory JSONs are written")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume a previous run by protocol_run_id (e.g. 20260604T123456)")
    parser.add_argument("--extend", type=str, default=None,
                        help="Like --resume but clears the stopped flag, allowing hard-capped "
                             "scenarios to collect additional batches (e.g. 20260604T123456)")

    # Interaction
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Prompt for manual approval before each batch (default: fully automated)")
    parser.add_argument("--allow-constant-batches", action="store_true", default=False,
                        help="Disable constant-behavior early stop (by default, a batch where "
                             "all trajectories are identical on both intent and execution levels "
                             "triggers immediate convergence without collecting further batches)")

    # Mocking
    parser.add_argument("--mock-llm-labels", action="store_true", default=False,
                        help="Skip LLM cluster labeling (use 'Cluster N' instead)")
    parser.add_argument("--mock-embeddings", action="store_true", default=False,
                        help="Use random unit vectors instead of sentence-transformers "
                             "(faster for smoke-testing; still runs HDBSCAN)")

    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for intent embeddings")
    parser.add_argument("--labeling-model", type=str, default="gpt-4.1-mini",
                        help="LLM model used to label intent clusters (default: gpt-4.1)")
    parser.add_argument("--no-qualitative", action="store_true", default=False,
                        help="Skip qualitative analysis (Section 8) after convergence")
    parser.add_argument("--qualitative-model", type=str, default=None,
                        help="LLM model for qualitative analysis (default: same as --labeling-model)")

    return parser.parse_args()


def main() -> None:
    import os
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    args = _parse_args()

    # Resolve scenario list
    if args.all_scenarios:
        scenario_numbers = [s.number for s in SCENARIOS if s.fault_fns is not None]
    else:
        try:
            scenario_numbers = [int(x.strip()) for x in args.scenarios.split(",")]
        except ValueError:
            print(f"ERROR: --scenarios must be comma-separated integers, got {args.scenarios!r}")
            sys.exit(1)

    if not scenario_numbers:
        print("ERROR: No simulatable scenarios found.")
        sys.exit(1)

    interactive = args.interactive

    import json as _json

    def _parse_agent_config(s: str, flag: str) -> dict:
        try:
            v = _json.loads(s)
            if not isinstance(v, dict):
                raise ValueError("must be a JSON object")
            return v
        except (_json.JSONDecodeError, ValueError) as e:
            print(f"ERROR: {flag}: invalid JSON — {e}")
            sys.exit(1)

    assistant_config = _parse_agent_config(args.assistant_config, "--assistant-config")
    service_config   = _parse_agent_config(args.service_config,   "--service-config")
    saboteur_config  = _parse_agent_config(args.saboteur_config,  "--saboteur-config")

    from datetime import datetime
    run_id = (args.resume or args.extend) if (args.resume or args.extend) else datetime.now().strftime("%Y%m%dT%H%M%S")

    config = ProtocolConfig(
        batch_size=args.batch_size,
        initial_batches=args.initial_batches,
        bootstrap_samples=args.bootstrap_samples,
        intra_batch_ari_percentile=args.ari_percentile,
        ari_min_threshold=args.ari_min_threshold,
        convergence_window=args.convergence_window,
        max_batches_per_scenario=args.max_batches,
        interactive=interactive,
        checkpoint_dir=Path(args.checkpoint_dir),
        trajectory_base_dir=Path(args.trajectory_dir),
        protocol_run_id=run_id,
        assistant_type=args.assistant,
        assistant_config=assistant_config,
        service_config=service_config,
        saboteur_config=saboteur_config,
        service_type=args.service,
        rounds=args.rounds,
        ignore_max_rounds_allowance=args.ignore_max_rounds_allowance,
        base_dir=Path(__file__).resolve().parent,
        mock_llm_labels=args.mock_llm_labels,
        mock_embeddings=args.mock_embeddings,
        embedding_model=args.embedding_model,
        labeling_model=args.labeling_model,
        max_concurrent_subprocesses=args.max_concurrent,
        allow_constant_batches=args.allow_constant_batches,
        extend=args.extend is not None,
        run_qualitative=not args.no_qualitative,
        qualitative_model=args.qualitative_model or args.labeling_model,
    )

    protocol = AdaptiveEvaluationProtocol(config)
    protocol.run(scenario_numbers)


if __name__ == "__main__":
    main()
