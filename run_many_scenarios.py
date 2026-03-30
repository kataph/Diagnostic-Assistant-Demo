""" Executes multiple times the given scenarios with the given input parameters, and records output.
python -m run_many_scenarios
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import re
import statistics

from Implementations.scenarios import SCENARIOS
from Utilities.formatting import to_PascalCase


def run_scenario_multiple_times(
    n_runs: int,
    base_dir: Path,
    forced_scenario: int,
    assistant: str,
    rounds: int,
    system: str,
    service: str,
    saboteur: str = "FixedScenario",
    assistant_model: str = "gpt-4.1",
) -> None:
    """
    Run the diagnostic scenario n_runs times.
    """
    python_executable = sys.executable

    cmd = [
        python_executable,
        "-m",
        "run_diagnostic_scenario",
        "--text-input-file",
        f"Knowledge_sources/Unstructured_knowledge_sources/{system}/{system}_description.txt",
        "--diagram",
        f"Knowledge_sources/Unstructured_knowledge_sources/{system}/{system}_schematics.png",
        "--LLM-assistant-model",
        assistant_model,
        "--NS-assistant-model",
        assistant_model,
        "--forced-scenario",
        str(forced_scenario),
        "--log-level",
        "10",
        "--rounds",
        str(rounds),
        "--kg",
        f"Knowledge_sources/Structured_knowledge_sources/{system}/zorro-ontology-{system.replace('_', '-')}-abox.ttl",
        "--system",
        f"{to_PascalCase(system)}System",
        "--ontology",
        "Knowledge_sources/Structured_knowledge_sources/zorro-ontology-tbox.ttl",
        "--retrieval-folder",
        f"Knowledge_sources/Unstructured_knowledge_sources/{system}",
        "--saboteur",
        saboteur,
        "--service",
        service,
        "--assistant",
        assistant,
        "--interface",
        "cli",
    ]

    for i in range(n_runs):
        print(f"Run {i + 1}/{n_runs} (forced-scenario={forced_scenario})...")
        result = subprocess.run(cmd, cwd=base_dir)
        if result.returncode != 0:
            print(f"Run {i + 1} failed with return code {result.returncode}.")


def get_last_n_log_files(log_dir: Path, n: int):
    """
    Return the last n files in log_dir, ordered by name (newest first).
    """
    if not log_dir.exists() or not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    files = [f for f in log_dir.iterdir() if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No log files found in directory: {log_dir}")

    files_sorted = sorted(files, key=lambda p: p.name, reverse=True)
    return files_sorted[:n]


def parse_cost_and_length_from_log(log_path: Path) -> tuple[float, int]:
    """
    Parse a cost vector line such as  "Cost vector: [3, 3, 5]"
    and return (sum, length).
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    match = re.search(r"Cost vector:\s*\[([^\]]+)\]", text)
    if not match:
        raise ValueError(f"No cost vector found in log file: {log_path}")

    vector_str = match.group(1)
    try:
        values = [float(x.strip()) for x in vector_str.split(",")]
    except ValueError as e:
        raise ValueError(f"Failed to parse cost vector in {log_path}: {e}")

    return sum(values), len(values)


def parse_time_from_log(log_path: Path) -> float:
    """
    Parse a time vector line and return the average.
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    match = re.search(r"Time vector:\s*\[([^\]]+)\]", text)
    if not match:
        raise ValueError(f"No time vector found in log file: {log_path}")

    vector_str = match.group(1)
    try:
        values = [float(x.strip()) for x in vector_str.split(",")]
    except ValueError as e:
        raise ValueError(f"Failed to parse time vector in {log_path}: {e}")

    return sum(values) / len(values) if values else 0.0


def compute_stats_from_logs(log_files):
    costs = []
    time_averages = []
    lengths = []
    for log in log_files:
        total_cost, length = parse_cost_and_length_from_log(log)
        average_time = parse_time_from_log(log)
        costs.append(total_cost)
        lengths.append(length)
        time_averages.append(average_time)

    if not costs:
        raise ValueError("No costs extracted from log files.")

    mean_cost = statistics.mean(costs)
    mean_action_number = statistics.mean(lengths)
    mean_time = statistics.mean(time_averages)
    std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0
    std_time = statistics.stdev(time_averages) if len(time_averages) > 1 else 0.0
    std_action_number = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

    return mean_cost, std_cost, mean_time, std_time, mean_action_number, std_action_number


def append_results(
    out_path: Path,
    mean_cost: float,
    std_cost: float,
    mean_time: float,
    std_time: float,
    n_runs: int,
    forced_scenario: int,
    system: str,
    log_files_interval: str,
    mean_action_number: float,
    std_action_number: float,
) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{n_runs}\t{forced_scenario}\t{system}\t"
            f"{mean_cost:.6f}\t{std_cost:.6f}\t"
            f"{mean_time:.6f}\t{std_time:.6f}\t"
            f"{mean_action_number:.6f}\t{std_action_number:.6f}\t"
            f"{log_files_interval}\n"
        )


def write_separator(out_path: Path, label: str) -> None:
    """Write a labelled separator line to out.txt."""
    with out_path.open("a", encoding="utf-8") as f:
        f.write(f"\n# {'=' * 30} {label} {'=' * 30}\n")


def main_args(
    num_runs: int,
    base_dir: Path,
    forced_scenario: int,
    assistant: str,
    rounds: int,
    skip_runs: bool,
    log_dir: Path,
    out_file: Path,
    service: str,
    saboteur: str = "FixedScenario",
    assistant_model: str = "gpt-4.1",
) -> None:

    target_scenarios = [s for s in SCENARIOS if s.id == forced_scenario]
    if len(target_scenarios) != 1:
        raise ValueError(
            f"Expected exactly 1 scenario with id={forced_scenario}, "
            f"found {len(target_scenarios)}."
        )
    system = target_scenarios[0].system_name

    if not skip_runs:
        run_scenario_multiple_times(
            num_runs, base_dir, forced_scenario, assistant, rounds,
            system, service, saboteur, assistant_model
        )

    log_files = get_last_n_log_files(log_dir, num_runs)
    log_files_interval = log_files[0].name + " --> " + log_files[-1].name
    mean_cost, std_cost, mean_time, std_time, mean_action_number, std_action_number = (
        compute_stats_from_logs(log_files)
    )
    append_results(
        out_file, mean_cost, std_cost, mean_time, std_time, num_runs,
        forced_scenario, system, log_files_interval,
        mean_action_number, std_action_number,
    )

    print(f"Processed {len(log_files)} log files.")
    print(f"Forced scenario: {forced_scenario} | System: {system}")
    print(f"Mean total cost: {mean_cost:.6f}  (std {std_cost:.6f})")
    print(f"Mean action number: {mean_action_number:.6f}  (std {std_action_number:.6f})")
    print(f"Mean suggestion time: {mean_time:.6f}  (std {std_time:.6f})")
    print(f"Results appended to: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run diagnostic scenario multiple times and compute cost statistics."
    )
    parser.add_argument("-n", "--num-runs", type=int, required=True,
                        help="Number of times to run the scenario.")
    parser.add_argument("--forced-scenario", type=int, required=True,
                        help="ID of the scenario to run (must be >= 0).")
    parser.add_argument("--rounds", type=int, required=True,
                        help="Maximum number of diagnostic rounds per run.")
    parser.add_argument("--assistant", type=str, required=True,
                        help="Diagnostic assistant type (e.g. 'LLM', 'KGO').")
    parser.add_argument("--assistant_model", type=str, default="gpt-4.1",
                        help="Model exploited by the assistant (default: 'gpt-4.1').")
    parser.add_argument("--service", type=str, required=True,
                        help="Service agent type (e.g. 'SpiceSim', 'Human', 'Mock').")
    parser.add_argument("--saboteur", type=str, default="FixedScenario",
                        help="Saboteur type used to inject faults (default: 'FixedScenario').")
    parser.add_argument("--skip-runs", action="store_true",
                        help="Skip running the scenario; only parse existing log files.")
    args = parser.parse_args()

    if args.forced_scenario < 0:
        raise ValueError("--forced-scenario must be >= 0")

    base_dir = Path(__file__).resolve().parent
    log_dir = base_dir / "Logs"
    out_file = base_dir / "out.txt"

    main_args(
        args.num_runs, base_dir, args.forced_scenario, args.assistant,
        args.rounds, args.skip_runs, log_dir, out_file,
        args.service, args.saboteur, args.assistant_model,
    )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    log_dir = base_dir / "Logs"
    out_file = base_dir / "out.txt"

    # ── SpiceSim service + LLM assistant ────────────────────────────────────
    
    
    # All scenarios that have simulation support (fault_fns defined).
    # simulatable_ids = [s.id for s in SCENARIOS if s.fault_fns is not None]

    # for scenario_id in simulatable_ids:
    #     print(f"\n=== Scenario {scenario_id} | SpiceSim + LLM ===")
    #     main_args(
    #         10, base_dir, scenario_id, "LLM", 10, False,
    #         log_dir, out_file, "SpiceSim", "SpiceSim",
    #     )
    kwargs = {'num_runs':1, 'base_dir':base_dir, 'forced_scenario':1, 'assistant':"LLM", 'rounds':10, 'skip_runs':False, 'log_dir':log_dir, 'out_file':out_file, 'service':"SpiceSim", 'saboteur':"SpiceSim", 'assistant_model':'gpt-4.1'}#"nf-gpt-4o-2024-08-06"}
    write_separator(
        out_file,
        f"kwargs={kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )
    # main_args(num_runs=2, base_dir=base_dir, forced_scenario=1, assistant="LLM", rounds=10, skip_runs=False, log_dir=log_dir, out_file=out_file, service="SpiceSim", saboteur="SpiceSim", assistant_model="gpt-4.1")
    main_args(**kwargs)
