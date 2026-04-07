"""
Utilities for cleaning up diagnostic run artefacts.
"""
from pathlib import Path


def delete_last_run(
    log_dir: Path | str | None = None,
    chat_dir: Path | str | None = None,
) -> None:
    """
    Delete the most recent log file and its corresponding chat HTML file.

    Files are sorted by name (which encodes an ISO-8601 timestamp, so
    lexicographic order = chronological order).  The youngest file in each
    directory is removed.

    Parameters
    ----------
    log_dir:
        Directory containing plain-text log files.
        Defaults to ``<repo_root>/Logs``.
    chat_dir:
        Directory containing HTML chat logs.
        Defaults to ``<repo_root>/Chats``.
    """
    base = Path(__file__).resolve().parent.parent
    log_dir  = Path(log_dir)  if log_dir  else base / "Logs"
    chat_dir = Path(chat_dir) if chat_dir else base / "Chats"

    for label, directory in (("log", log_dir), ("chat", chat_dir)):
        files = sorted(
            (f for f in directory.iterdir() if f.is_file()),
            key=lambda p: p.name,
        )
        if not files:
            print(f"No {label} files found in {directory}.")
            continue
        target = files[-1]
        target.unlink()
        print(f"Deleted {label} file: {target}")
        
if __name__ == "__main__":
    delete_last_run()