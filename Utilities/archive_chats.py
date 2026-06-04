"""
Archive the last N chat logs (as plain text) per (assistant, scenario) pair.

Usage:
    python -m Utilities.archive_chats \\
        --assistants EvidenceKGOptimal LLM \\
        --n 10 \\
        [--output out_archives]

For each assistant in --assistants and each scenario subfolder found under
Logs/Chats/<assistant>/, the script selects the last N files (by timestamp),
textifies them via textify_chat(), and packs the .txt files into a zip
archive saved to --output.

Archive naming: <assistant>__scenario<scenario>__last<N>.zip
"""
import argparse
import re
import zipfile
from pathlib import Path

from Utilities.select_runs import textify_chat

_TS_RE = re.compile(
    r"DIAGNOSTIC_SCENARIO_RUN_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)"
)

_BASE = Path(__file__).resolve().parent.parent  # ESWC_2026_Demo/


def archive_chats(
    assistants: list[str],
    n: int,
    output: Path | None = None,
    scenario: int | None = None,
) -> list[Path]:
    """
    For each assistant and each scenario subfolder under Logs/Chats/<assistant>/,
    textify the last N chat files and pack them into a zip archive.

    If *scenario* is given, only that scenario subfolder is processed.

    Returns list of written archive paths.
    """
    chat_root = _BASE / "Logs/Chats"
    output_dir = output or (_BASE / "out_archives")
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []

    for assistant in assistants:
        assistant_dir = chat_root / assistant
        if not assistant_dir.exists():
            print(f"Warning: {assistant_dir} does not exist, skipping.")
            continue

        if scenario is not None:
            candidate = assistant_dir / str(scenario)
            if not candidate.is_dir():
                print(f"Warning: {candidate} does not exist, skipping.")
                continue
            scenario_dirs = [candidate]
        else:
            scenario_dirs = sorted(
                [d for d in assistant_dir.iterdir() if d.is_dir()],
                key=lambda d: d.name,
            )

        for scenario_dir in scenario_dirs:
            files_with_ts: list[tuple[str, Path]] = []
            for f in scenario_dir.iterdir():
                m = _TS_RE.search(f.name)
                if m:
                    files_with_ts.append((m.group(1), f))

            if not files_with_ts:
                continue

            # Sort ascending by timestamp, take the last N
            files_with_ts.sort(key=lambda x: x[0])
            selected = files_with_ts[-n:]

            archive_name = (
                f"{assistant}__scenario{scenario_dir.name}__last{n}.zip"
            )
            archive_path = output_dir / archive_name

            with zipfile.ZipFile(
                archive_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for _, src in selected:
                    txt_name = src.stem.removesuffix("_CHAT") + "_CHAT.txt"
                    content = textify_chat(
                        src.read_text(encoding="utf-8", errors="replace")
                    )
                    zf.writestr(txt_name, content.encode("utf-8"))

            print(f"  {archive_name}: {len(selected)} file(s)")
            written.append(archive_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Archive the last N chat logs (textified) per (assistant, scenario)."
        )
    )
    parser.add_argument(
        "--assistants",
        nargs="+",
        required=True,
        help="Assistant names (must match folder names under Chats/)",
    )
    parser.add_argument(
        "--n",
        type=int,
        required=True,
        help="Number of most recent logs to include per scenario",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Destination folder (default: <repo>/out_archives)",
    )
    parser.add_argument(
        "--scenario",
        type=int,
        default=None,
        help="If given, only archive this scenario ID (default: all scenarios)",
    )
    args = parser.parse_args()

    output = Path(args.output) if args.output else None
    written = archive_chats(
        assistants=args.assistants,
        n=args.n,
        output=output,
        scenario=args.scenario,
    )

    dest = output or (_BASE / "out_archives")
    print(f"\nWritten {len(written)} archive(s) to {dest}:")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

"""
python -m Utilities.archive_chats --assistants EvidenceKGOptimal --n 10 --scenario 2
"""
