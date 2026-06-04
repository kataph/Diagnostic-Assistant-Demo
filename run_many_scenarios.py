""" Executes multiple times the given scenarios with the given input parameters, and records output.
python -m run_many_scenarios
"""

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
import re
import statistics

from Implementations.fault_injections import SCENARIOS
from Utilities.formatting import to_PascalCase

_out_file_lock = threading.Lock()


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
    log_path: str | None = None,
    chat_path: str | None = None,
    trajectory_path: str | None = None,
    batch_size_of_same_scenario_runs: int = 1,
    semaphore: threading.Semaphore | None = None,
) -> None:
    """
    Run the diagnostic scenario n_runs times, in parallel batches of
    batch_size_of_same_scenario_runs.  Runs within a batch are started 1 second
    apart to guarantee unique timestamp-based filenames.
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
    if log_path is not None:
        cmd += ["--log-path", log_path]
    if chat_path is not None:
        cmd += ["--chat-path", chat_path]
    if trajectory_path is not None:
        cmd += ["--trajectory-path", trajectory_path]

    def _single_run(run_idx: int) -> None:
        if semaphore is not None:
            semaphore.acquire()
        try:
            result = subprocess.run(cmd, cwd=base_dir)
        finally:
            if semaphore is not None:
                semaphore.release()
        if result.returncode != 0:
            print(f"Run {run_idx} failed with return code {result.returncode}.")

    for batch_start in range(0, n_runs, batch_size_of_same_scenario_runs):
        batch = range(batch_start, min(batch_start + batch_size_of_same_scenario_runs, n_runs))
        threads = []
        for idx, i in enumerate(batch):
            print(f"Run {i + 1}/{n_runs} (forced-scenario={forced_scenario})...")
            t = threading.Thread(target=_single_run, args=(i + 1,))
            t.start()
            threads.append(t)
            if idx < len(batch) - 1:
                time.sleep(1)
        for t in threads:
            t.join()


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
        # raise ValueError(f"No cost vector found in log file: {log_path}")
        print(f"No cost vector found in log file: {log_path}, returning zero values")
        return 0, 0

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


def parse_success_from_log(log_path: Path) -> bool:
    """
    Return True if the run ended successfully, i.e. either:
      - A hypothesis verification returned outcome='correct', OR
      - The system was restored via a direct diagnostic action
        (replace_component fixed the only fault without a formal hypothesis).
    Returns False if the session was exhausted by the patience cap.
    """
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    return bool(
        re.search(r"outcome='correct'", text)
        or re.search(r"system_restored_via_action", text)
    )


def compute_stats_from_logs(log_files):
    costs = []
    time_averages = []
    lengths = []
    successes = []
    for log in log_files:
        total_cost, length = parse_cost_and_length_from_log(log)
        average_time = parse_time_from_log(log)
        costs.append(total_cost)
        lengths.append(length)
        time_averages.append(average_time)
        successes.append(parse_success_from_log(log))

    if not costs:
        raise ValueError("No costs extracted from log files.")

    mean_cost = statistics.mean(costs)
    mean_action_number = statistics.mean(lengths)
    mean_time = statistics.mean(time_averages)
    std_cost = statistics.stdev(costs) if len(costs) > 1 else 0.0
    std_time = statistics.stdev(time_averages) if len(time_averages) > 1 else 0.0
    std_action_number = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    success_rate = sum(successes) / len(successes)

    return mean_cost, std_cost, mean_time, std_time, mean_action_number, std_action_number, success_rate


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
    success_rate: float,
) -> None:
    with out_path.open("a", encoding="utf-8") as f:
        f.write(
            f"{n_runs}\t{forced_scenario}\t{system}\t"
            f"{mean_cost:.6f}\t{std_cost:.6f}\t"
            f"{mean_time:.6f}\t{std_time:.6f}\t"
            f"{mean_action_number:.6f}\t{std_action_number:.6f}\t"
            f"{success_rate:.4f}\t"
            f"{log_files_interval}\n"
        )


def write_separator(out_path: Path, label: str) -> None:
    """Write a labelled separator line to batch_output_summary.txt."""
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
    batch_size_of_same_scenario_runs: int = 1,
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
            system, service, saboteur, assistant_model,
            log_path=str(log_dir),
            batch_size_of_same_scenario_runs=batch_size_of_same_scenario_runs,
        )

    log_files = get_last_n_log_files(log_dir, num_runs)
    log_files_interval = log_files[0].name + " --> " + log_files[-1].name
    mean_cost, std_cost, mean_time, std_time, mean_action_number, std_action_number, success_rate = (
        compute_stats_from_logs(log_files)
    )
    with _out_file_lock:
        append_results(
            out_file, mean_cost, std_cost, mean_time, std_time, num_runs,
            forced_scenario, system, log_files_interval,
            mean_action_number, std_action_number, success_rate,
        )

    print(f"Processed {len(log_files)} log files.")
    print(f"Forced scenario: {forced_scenario} | System: {system}")
    print(f"Mean total cost: {mean_cost:.6f}  (std {std_cost:.6f})")
    print(f"Mean action number: {mean_action_number:.6f}  (std {std_action_number:.6f})")
    print(f"Mean suggestion time: {mean_time:.6f}  (std {std_time:.6f})")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Results appended to: {out_file}")


def main_args_parallel(
    num_runs: int,
    base_dir: Path,
    forced_scenarios_list: list[int],
    assistant: str,
    rounds: int,
    skip_runs: bool,
    log_dir: Path,
    out_file: Path,
    service: str,
    saboteur: str = "FixedScenario",
    assistant_model: str = "gpt-4.1",
    chat_dir: Path | None = None,
    trajectory_dir: Path | None = None,
    batch_size_of_same_scenario_runs: int = 1,
    max_concurrent_subprocesses: int | None = None,
) -> None:
    """
    Run each scenario ID in *forced_scenarios_list* with *num_runs* sequential
    runs, executing all scenario groups in parallel.

    Logs for scenario ID ``sid`` go into ``log_dir / str(sid)`` and chat logs
    go into ``chat_dir / str(sid)`` (default: ``log_dir.parent / "Logs/Chats"``).
    Statistics for each scenario are written to *out_file* as they finish
    (protected by a lock).
    """
    if chat_dir is None:
        chat_dir = log_dir.parent / "Logs/Chats"
    if trajectory_dir is None:
        trajectory_dir = log_dir.parent / "Logs/Trajectories"

    semaphore = (
        threading.Semaphore(max_concurrent_subprocesses)
        if max_concurrent_subprocesses is not None
        else None
    )

    def _run_one(scenario_id: int) -> None:
        scenario_log_dir = log_dir / assistant / str(scenario_id)
        scenario_chat_dir = chat_dir / assistant / str(scenario_id)
        scenario_trajectory_dir = trajectory_dir / assistant / str(scenario_id)
        scenario_log_dir.mkdir(parents=True, exist_ok=True)
        scenario_chat_dir.mkdir(parents=True, exist_ok=True)
        scenario_trajectory_dir.mkdir(parents=True, exist_ok=True)

        target_scenarios = [s for s in SCENARIOS if s.id == scenario_id]
        if len(target_scenarios) != 1:
            print(
                f"[scenario {scenario_id}] Expected exactly 1 matching scenario, "
                f"found {len(target_scenarios)}. Skipping."
            )
            return
        system = target_scenarios[0].system_name

        try:
            if not skip_runs:
                run_scenario_multiple_times(
                    num_runs, base_dir, scenario_id, assistant, rounds,
                    system, service, saboteur, assistant_model,
                    log_path=str(scenario_log_dir),
                    chat_path=str(scenario_chat_dir),
                    trajectory_path=str(scenario_trajectory_dir),
                    batch_size_of_same_scenario_runs=batch_size_of_same_scenario_runs,
                    semaphore=semaphore,
                )

            log_files = get_last_n_log_files(scenario_log_dir, num_runs)
            log_files_interval = log_files[0].name + " --> " + log_files[-1].name
            mean_cost, std_cost, mean_time, std_time, mean_action_number, std_action_number, success_rate = (
                compute_stats_from_logs(log_files)
            )
            with _out_file_lock:
                append_results(
                    out_file, mean_cost, std_cost, mean_time, std_time, num_runs,
                    scenario_id, system, log_files_interval,
                    mean_action_number, std_action_number, success_rate,
                )

            print(f"[scenario {scenario_id}] Processed {len(log_files)} log files.")
            print(f"[scenario {scenario_id}] Mean total cost: {mean_cost:.6f}  (std {std_cost:.6f})")
            print(f"[scenario {scenario_id}] Mean action number: {mean_action_number:.6f}  (std {std_action_number:.6f})")
            print(f"[scenario {scenario_id}] Mean suggestion time: {mean_time:.6f}  (std {std_time:.6f})")
            print(f"[scenario {scenario_id}] Success rate: {success_rate:.1%}")
        except Exception:
            import traceback
            print(
                f"[scenario {scenario_id}] ERROR — results not recorded:\n"
                + traceback.format_exc()
            )

    threads = [threading.Thread(target=_run_one, args=(sid,)) for sid in forced_scenarios_list]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


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
    # log_dir = base_dir / "Logs"
    out_file = base_dir / "batch_output_summary.txt"

    main_args(
        args.num_runs, base_dir, args.forced_scenario, args.assistant,
        args.rounds, args.skip_runs, out_file,
        args.service, args.saboteur, args.assistant_model,
    )


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    log_dir = base_dir / "Logs/DebuggingLogs"
    out_file = base_dir / "batch_output_summary.txt"

    # ── SpiceSim service + LLM assistant ────────────────────────────────────
    
    
    # All scenarios that have simulation support (fault_fns defined).
    simulatable_ids = [s.id for s in SCENARIOS if s.fault_fns is not None]
    
    # num_runs = 10
    # assistant = "EvidenceKGOptimal"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': [1,2,3,4,5,6,7],
    #     'assistant': assistant, 'rounds': 10, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    
    # num_runs = 10
    # assistant = "EvidenceKGOptimal"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': [8,9,10,11,12,13,14],
    #     'assistant': assistant, 'rounds': 10, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    
    # num_runs = 10
    # assistant = "LLM"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': [1,2,3,4,5,6,7],
    #     'assistant': assistant, 'rounds': 10, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    
    # num_runs = 10
    # assistant = "LLM"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': [8,9,10,11,12,13,14],
    #     'assistant': assistant, 'rounds': 10, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    
    
    # num_runs = 10
    # assistant = "EvidenceKGOptimal"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': [15],
    #     'assistant': assistant, 'rounds': 10, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    
    # num_runs = 1
    # assistant = "EvidenceKGOptimal"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': [11],
    #     'assistant': assistant, 'rounds': 10, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    
    # num_runs = 10
    # assistant = "EvidenceKGOptimal"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': [11,12],
    #     'assistant': assistant, 'rounds': 10, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    # num_runs = 10
    # scenarios = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # batch_size = 1 # keep batch_size*len(scenarios) <= 10 
    # assistant = "LLM"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': scenarios,
    #     'assistant': assistant, 'rounds': 10, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    #     'batch_size_of_same_scenario_runs': batch_size,
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    # num_runs = 10
    # # scenarios = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # scenarios = [15]
    # batch_size = 10 # keep batch_size*len(scenarios) <= 10 
    # rounds = 25
    # assistant = "EvidenceKGOptimal"
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': scenarios,
    #     'assistant': assistant, 'rounds': rounds, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    #     'batch_size_of_same_scenario_runs': batch_size,
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    # num_runs = 10
    # # scenarios = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # scenarios = [15]
    # batch_size = 10 # keep batch_size*len(scenarios) <= 10 
    # assistant = "LLM"
    # rounds = 25
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': scenarios,
    #     'assistant': assistant, 'rounds': rounds, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': 'gpt-4.1',
    #     'batch_size_of_same_scenario_runs': batch_size,
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    # num_runs = 10
    # scenarios = [15]
    # batch_size = 10 # keep batch_size*len(scenarios) <= 10 or 15 
    # assistant = "EvidenceKGOptimal"
    # rounds = 10
    # model = 'gpt-4.1'
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': scenarios,
    #     'assistant': assistant, 'rounds': rounds, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': model,
    #     'batch_size_of_same_scenario_runs': batch_size,
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
    num_runs = 10
    scenarios = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    batch_size = 2 # keep batch_size*len(scenarios) <= 10 or 15 
    assistant = "LLM"
    rounds = 10
    model = 'gpt-4.1'
    parallel_kwargs = {
        'num_runs': num_runs, 'base_dir': base_dir,
        'forced_scenarios_list': scenarios,
        'assistant': assistant, 'rounds': rounds, 'skip_runs': False,
        'log_dir': log_dir, 'out_file': out_file,
        'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': model,
        'batch_size_of_same_scenario_runs': batch_size,
    }
    write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    main_args_parallel(**parallel_kwargs)
    # num_runs = 10
    # scenarios = [4]
    # batch_size = 10 # keep batch_size*len(scenarios) <= 10 or 15 
    # assistant = "LLM"
    # rounds = 10
    # model = 'gpt-4.1'
    # parallel_kwargs = {
    #     'num_runs': num_runs, 'base_dir': base_dir,
    #     'forced_scenarios_list': scenarios,
    #     'assistant': assistant, 'rounds': rounds, 'skip_runs': False,
    #     'log_dir': log_dir, 'out_file': out_file,
    #     'service': "SpiceSim", 'saboteur': "SpiceSim", 'assistant_model': model,
    #     'batch_size_of_same_scenario_runs': batch_size,
    # }
    # write_separator(out_file, f"parallel kwargs={parallel_kwargs} --- {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    # main_args_parallel(**parallel_kwargs)
