import argparse
import subprocess
import sys
from pathlib import Path
import re
import statistics

from Implementations.saboteurFixedScenario import SCENARIOS
from Utilities.formatting import to_PascalCase


def run_scenario_multiple_times(n_runs: int, base_dir: Path, forced_scenario: int, assistant: str, rounds: int, system: str, service: str) -> None:
    """
    Run the diagnostic scenario n_runs times.
    """
    # Use the same Python interpreter that runs this script
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
        "gpt-5.2",
        "--NS-assistant-model",
        "gpt-5.2",
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
        "FixedScenario",
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
            # You can choose to break here if failures should stop the loop
            # break


def get_last_n_log_files(log_dir: Path, n: int):
    """
    Return the last n files in log_dir, ordered by modification time (newest first).
    """
    if not log_dir.exists() or not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    files = [f for f in log_dir.iterdir() if f.is_file()]
    if not files:
        raise FileNotFoundError(f"No log files found in directory: {log_dir}")

    # files_sorted = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
    files_sorted = sorted(files, key=lambda p: p.name, reverse=True)
    return files_sorted[:n]


def parse_cost_and_length_from_log(log_path: Path) -> tuple[int, int]:
    """
    Parse a cost vector of the form:
    Cost vector: [3, 3, 5]
    and return the sum of its entries and the length of the vector.
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Regex to find the cost vector line
    match = re.search(r"Cost vector:\s*\[([^\]]+)\]", text)
    if not match:
        raise ValueError(f"No cost vector found in log file: {log_path}")

    vector_str = match.group(1)
    # Split on commas and convert to integers
    try:
        values = [int(x.strip()) for x in vector_str.split(",")]
    except ValueError as e:
        raise ValueError(f"Failed to parse cost vector in {log_path}: {e}")

    return sum(values), len(values)


def parse_time_from_log(log_path: Path):
    """
    Parse a time vector of the form:
    Time vector: [1.2, 2.3]
    and return the average value of its entries.
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")

    # Regex to find the cost vector line
    match = re.search(r"Time vector:\s*\[([^\]]+)\]", text)
    if not match:
        raise ValueError(f"No time vector found in log file: {log_path}")

    vector_str = match.group(1)
    # Split on commas and convert to integers
    try:
        values = [float(x.strip()) for x in vector_str.split(",")]
    except ValueError as e:
        raise ValueError(f"Failed to parse time vector in {log_path}: {e}")

    return sum(values)/len(values) if len(values) > 0 else 0


def compute_stats_from_logs(log_files):
    """
    For a list of log file paths, compute mean and standard deviation
    of the total costs (sum of cost vector entries in each file).
    """
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
    std_time = statistics.stdev(time_averages) if len(
        time_averages) > 1 else 0.0
    std_action_number = statistics.stdev(lengths) if len(lengths) > 1 else 0.0

    return mean_cost, std_cost, mean_time, std_time, mean_action_number, std_action_number


def append_results(out_path: Path, mean_cost: float, std_cost: float, mean_time: float, std_time: float, n_runs: int, forced_scenario: int, system: str, log_files_interval: str, mean_action_number: float, std_action_number: float):
    """
    Append mean and standard deviation to out.txt.
    Format: n_runs forced_scenario mean std
    """
    with out_path.open("a", encoding="utf-8") as f:
        # number_of_runs, scenario_id, system_id, mean_action_cost, std_action_cost, mean_suggestion_time, std_suggestion_time, mean_action_number, std_action_number, log_files_interval
        f.write(f"{n_runs}\t{forced_scenario}\t{system}\t{mean_cost:.6f}\t{std_cost:.6f}\t{mean_time:.6f}\t{std_time:.6f}\t{mean_action_number:.6f}\t{std_action_number:.6f}\t{log_files_interval}\n")


def main_args(num_runs, base_dir, forced_scenario, assistant, rounds, skip_runs: bool, log_dir, out_file, service="LLM"):


    target_scenarios = [scenario for scenario in SCENARIOS if scenario[0] == forced_scenario]
    if len(target_scenarios) != 1:
        raise ValueError(f"The scenarios corresponding to id {forced_scenario} are {len(target_scenarios)}. It should be that it is exactly 1!")
    _, system, _ = target_scenarios[0]

    if not skip_runs:
        run_scenario_multiple_times(
            num_runs, base_dir, forced_scenario, assistant, rounds, system, service)

    # Get last n log files and compute statistics
    log_files = get_last_n_log_files(log_dir, num_runs)
    log_files_interval = log_files[0].name + " --> " + log_files[-1].name
    mean_cost, std_cost, mean_time, std_time, mean_action_number, std_action_number = compute_stats_from_logs(
        log_files)

    append_results(out_file, mean_cost, std_cost, mean_time, std_time, num_runs,
                   forced_scenario, system, log_files_interval, mean_action_number, std_action_number)

    print(f"Processed {len(log_files)} log files.")
    print(f"Forced scenario: {forced_scenario}")
    print(f"System: {system}")
    print(f"Mean total cost: {mean_cost:.6f}")
    print(f"Std dev of total cost: {std_cost:.6f}")
    print(f"Mean suggestion time: {mean_time:.6f}")
    print(f"Std dev of suggestion time: {std_time:.6f}")
    print(f"Mean action number: {mean_action_number:.6f}")
    print(f"Std dev of action number: {std_action_number:.6f}")
    print(f"Results appended to: {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run diagnostic scenario multiple times and compute cost statistics."
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        required=True,
        help="Number of times to run the diagnostic scenario.",
    )
    parser.add_argument(
        "--forced-scenario",
        type=int,
        required=True,
        help="Scenario identifier (int >= 0, see the Fixed scenario assistant class for the scenarios array).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        required=True,
        help="Max number of rounds",
    )
    parser.add_argument(
        "--assistant",
        type=str,
        required=True,
        help="Assistant type for the scenario",
    )
    parser.add_argument(
        "--service",
        type=str,
        default="LLM",
        help="Service agent type for the scenario",
    )
    parser.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip running the scenario and only analyze the last n log files.",
    )
    args = parser.parse_args()

    if args.forced_scenario < 0:
        raise ValueError("--forced-scenario must be an integer >= 0")

    # Base directory = directory of this script
    base_dir = Path(__file__).resolve().parent
    log_dir = base_dir / "Logs"
    out_file = base_dir / "out.txt"

    main_args(args.num_runs, base_dir, args.forced_scenario, args.assistant,
              args.rounds, args.skip_runs, log_dir, out_file, args.service)


if __name__ == "__main__":
    # main()
    base_dir = Path(__file__).resolve().parent
    log_dir = base_dir / "Logs"
    out_file = base_dir / "out.txt"
    # main_args(2, base_dir, 1, "LLM", 10, False, log_dir, out_file) # rapid test
    # main_args(1, base_dir, 1, "EvidenceKGOptimal", 10, False, log_dir, out_file, "LLM") # rapid test

    main_args(10, base_dir, 0, "LLM", 10, False, log_dir, out_file)
    main_args(10, base_dir, 0, "EvidenceKGOptimal",
              10, False, log_dir, out_file)

    # main_args(10, base_dir, 1, "LLM", 10, True, log_dir, out_file)
    # main_args(10, base_dir, 2, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 3, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 4, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 5, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 6, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 7, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 8, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 9, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 10, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 11, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 12, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 12, "LLM", 8, True, log_dir, out_file)
    # main_args(10, base_dir, 1, "EvidenceKGOptimal", 10, False, log_dir, out_file)

    # main_args(10, base_dir, 2, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 3, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 4, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 5, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    main_args(10, base_dir, 6, "EvidenceKGOptimal",
              10, False, log_dir, out_file)
    # main_args(10, base_dir, 7, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 8, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 9, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 10, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 11, "EvidenceKGOptimal", 10, False, log_dir, out_file)
    main_args(10, base_dir, 12, "EvidenceKGOptimal",
              10, False, log_dir, out_file)
    # main_args(10, base_dir, 13, "LLM", 10, False, log_dir, out_file)
    # main_args(10, base_dir, 14, "LLM", 10, False, log_dir, out_file)
