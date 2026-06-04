"""
Select and copy diagnostic run files (Chat or Log) within a timestamp range.

Usage:
    python -m Utilities.select_runs \\
        --start 2026-04-09T12:08:52 \\
        --end   2026-04-09T13:00:00 \\
        --scenario 11 \\
        [--type Chat|Log] \\
        [--output path/to/dest] \\
        [--no-textify]

Both --start and --end are inclusive. If either timestamp does not appear
exactly in any filename an error is raised immediately (typo guard).

When --type Chat and textify=True (default), each HTML chat log is converted
to a plain-text .txt file instead of copying the raw HTML.

python -m Utilities.select_runs \\
        --start 2026-04-09T12:08:52 \\
        --end   2026-04-09T13:00:00 \\
        --scenario 10 \\
        --type Chat \\
        --output old_LLM_10
"""
import argparse
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path

_TS_RE = re.compile(r"DIAGNOSTIC_SCENARIO_RUN_(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)")

_BASE = Path(__file__).resolve().parent.parent  # ESWC_2026_Demo/


# ── HTML → plain text ─────────────────────────────────────────────────────────

class _ChatParser(HTMLParser):
    """Extract structured dialogue from a diagnostic chat HTML log."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.messages: list[dict] = []
        self._div_stack: list[set] = []
        self._collect: str | None = None   # 'sender'|'body'|'ts'|'pill'
        self._collect_depth: int = -1
        self._buf: list[str] = []
        self._cur_sender: str | None = None
        self._cur_body:   str | None = None
        self._cur_ts:     str | None = None

    @staticmethod
    def _classes(attrs) -> set:
        for name, val in attrs:
            if name == "class":
                return set(val.split())
        return set()

    def handle_starttag(self, tag, attrs):
        if tag == "br":
            if self._collect is not None:
                self._buf.append("\n")
            return
        if tag != "div":
            # Inline tags (span, b, i, etc.): insert a space separator when
            # collecting so adjacent elements don't run together.
            if self._collect is not None and self._buf:
                last = "".join(self._buf)
                if last and last[-1] not in (" ", "\n"):
                    self._buf.append(" ")
            return

        classes = self._classes(attrs)
        self._div_stack.append(classes)
        depth = len(self._div_stack)

        if self._collect is not None:
            # Nested div inside a collected section: emit newline separator.
            self._buf.append("\n")
            return

        if "row" in classes:
            self._cur_sender = self._cur_body = self._cur_ts = None
        elif "sender" in classes:
            self._collect, self._collect_depth, self._buf = "sender", depth, []
        elif "body" in classes:
            self._collect, self._collect_depth, self._buf = "body",   depth, []
        elif "ts" in classes:
            self._collect, self._collect_depth, self._buf = "ts",     depth, []
        elif "pill" in classes:
            self._collect, self._collect_depth, self._buf = "pill",   depth, []

    def handle_endtag(self, tag):
        if tag != "div" or not self._div_stack:
            return

        depth = len(self._div_stack)

        # Close a collected section?
        if self._collect is not None and depth == self._collect_depth:
            text = "".join(self._buf).strip()
            if   self._collect == "sender": self._cur_sender = text
            elif self._collect == "body":   self._cur_body   = text
            elif self._collect == "ts":     self._cur_ts     = text
            elif self._collect == "pill":
                self.messages.append({"type": "pill", "text": text})
            self._collect, self._collect_depth, self._buf = None, -1, []

        # Close a bubble row?
        if "row" in self._div_stack[-1] and self._cur_sender is not None:
            self.messages.append({
                "type":   "bubble",
                "sender": self._cur_sender,
                "body":   self._cur_body or "",
                "ts":     self._cur_ts,
            })
            self._cur_sender = self._cur_body = self._cur_ts = None

        self._div_stack.pop()

    def handle_data(self, data):
        if self._collect is not None:
            self._buf.append(data)


def textify_chat(html_content: str) -> str:
    """Convert a diagnostic chat HTML log to structured plain text."""
    run_id_m = re.search(r"<p>(DIAGNOSTIC_SCENARIO_RUN_[^<]+)</p>", html_content)
    run_id = run_id_m.group(1) if run_id_m else "Unknown Session"

    parser = _ChatParser()
    parser.feed(html_content)

    lines = [f"=== {run_id} ===", ""]
    for msg in parser.messages:
        if msg["type"] == "pill":
            lines.append(f"--- {msg['text']}")
            lines.append("")
        else:
            ts     = f"[{msg['ts']}] " if msg.get("ts") else ""
            sender = msg["sender"].upper()
            lines.append(f"{ts}{sender}:")
            lines.append(msg["body"])
            lines.append("")

    return "\n".join(lines)


# ── file selection & copy ─────────────────────────────────────────────────────

_DEFAULT_ASSISTANT = "EvidenceKGOptimal"


def _source_dir(file_type: str, scenario: int, assistant: str = _DEFAULT_ASSISTANT) -> Path:
    folder = "Logs/Chats" if file_type == "Chat" else "Logs/DebuggingLogs"
    return _BASE / folder / assistant / str(scenario)


def _default_output() -> Path:
    return _BASE / "Logs/Textified"


def _write_file(src: Path, output_dir: Path, textify: bool) -> Path:
    """Copy or textify *src* into *output_dir* and return the destination path."""
    if textify:
        txt_name = src.stem.removesuffix("_CHAT") + "_CHAT.txt"
        dest = output_dir / txt_name
        dest.write_text(
            textify_chat(src.read_text(encoding="utf-8", errors="replace")),
            encoding="utf-8",
        )
    else:
        dest = output_dir / src.name
        shutil.copy2(src, dest)
    return dest


def select_runs_by_timestamp(
    time_stamp_1: str,
    time_stamp_2: str,
    scenario: int,
    file_type: str = "Chat",
    output: Path | None = None,
    textify: bool = True,
    assistant: str = _DEFAULT_ASSISTANT,
) -> list[Path]:
    """
    Copy (or textify) files whose timestamp falls within [time_stamp_1, time_stamp_2]
    (inclusive, order-independent) and return the list of written paths.

    Parameters
    ----------
    time_stamp_1, time_stamp_2   : inclusive timestamp strings, e.g. "2026-04-09T12:08:52"
    scenario     : scenario folder number
    file_type    : "Chat" or "Log"
    output       : destination folder (default: <repo>/Logs/Textified)
    textify      : if True and file_type=="Chat", convert HTML to plain text
                   (.txt) instead of copying the raw HTML.  Raises ValueError
                   if textify=True and file_type=="Log".
    assistant    : subfolder name under Logs/Chats/ or Logs/DebuggingLogs/ (default: "EvidenceKGOptimal")

    Raises ValueError  if start/end not found in any filename (typo guard).
    Raises FileNotFoundError if the source directory does not exist.
    """
    if textify and file_type != "Chat":
        raise ValueError("textify=True is only valid when file_type='Chat'.")

    source_dir = _source_dir(file_type, scenario, assistant)
    output_dir = output or _default_output()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files_with_ts: list[tuple[str, Path]] = []
    for f in source_dir.iterdir():
        m = _TS_RE.search(f.name)
        if m:
            files_with_ts.append((m.group(1), f))

    all_timestamps = {ts for ts, _ in files_with_ts}

    if time_stamp_1 <= time_stamp_2:
        start, end = time_stamp_1, time_stamp_2
    else:
        start, end = time_stamp_2, time_stamp_1

    if start not in all_timestamps:
        raise ValueError(
            f"Start timestamp '{start}' not found in any filename under {source_dir}.\n"
            f"Available: {sorted(all_timestamps)}"
        )
    if end not in all_timestamps:
        raise ValueError(
            f"End timestamp '{end}' not found in any filename under {source_dir}.\n"
            f"Available: {sorted(all_timestamps)}"
        )

    # ISO timestamps sort lexicographically, so string comparison is correct.
    selected = sorted(
        ((ts, f) for ts, f in files_with_ts if start <= ts <= end),
        key=lambda x: x[0],
    )

    return [_write_file(src, output_dir, textify) for _, src in selected]


def select_last_n_runs(
    n: int,
    scenario: int,
    file_type: str = "Chat",
    output: Path | None = None,
    textify: bool = True,
    assistant: str = _DEFAULT_ASSISTANT,
) -> list[Path]:
    """
    Copy (or textify) the *n* most-recent run files for *scenario* and return
    the list of written paths.

    Parameters
    ----------
    n            : number of most-recent files to select
    scenario     : scenario folder number
    file_type    : "Chat" or "Log"
    output       : destination folder (default: <repo>/out)
    textify      : if True and file_type=="Chat", convert HTML to plain text.
                   Raises ValueError if textify=True and file_type=="Log".
    assistant    : subfolder name under Logs/Chats/ or Logs/DebuggingLogs/ (default: "EvidenceKGOptimal")

    Raises FileNotFoundError if the source directory does not exist.
    """
    if textify and file_type != "Chat":
        raise ValueError("textify=True is only valid when file_type='Chat'.")

    source_dir = _source_dir(file_type, scenario, assistant)
    output_dir = output or _default_output()

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    files_with_ts: list[tuple[str, Path]] = [
        (m.group(1), f)
        for f in source_dir.iterdir()
        if (m := _TS_RE.search(f.name))
    ]
    files_with_ts.sort(key=lambda x: x[0])
    selected = files_with_ts[-n:]

    return [_write_file(src, output_dir, textify) for _, src in selected]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy or textify diagnostic run files. "
            "Use --n for the last N runs, or --t1/--t2 for a timestamp range."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--n", type=int, metavar="N",
                      help="Select the N most-recent runs")
    mode.add_argument("--t1", metavar="TIMESTAMP",
                      help="Inclusive first timestamp, e.g. 2026-04-09T12:08:52")
    parser.add_argument("--t2", metavar="TIMESTAMP",
                        help="Inclusive second timestamp (required with --t1)")
    parser.add_argument("--assistant", default=_DEFAULT_ASSISTANT,
                        help=f"Assistant subfolder under Logs/Chats/ or Logs/DebuggingLogs/ (default: {_DEFAULT_ASSISTANT!r})")
    parser.add_argument("--type", choices=["Chat", "Log"], default="Chat", dest="file_type",
                        help="File type to select (default: Chat)")
    parser.add_argument("--scenario", type=int, required=True,
                        help="Scenario number")
    parser.add_argument("--output", default=None,
                        help="Destination folder (default: <repo>/out)")
    parser.add_argument("--textify", default=True, type=lambda v: str(v).lower() != "false",
                        help="Convert HTML chats to plain text (default: true)")
    args = parser.parse_args()

    output = Path(args.output) if args.output else None

    if args.n is not None:
        written = select_last_n_runs(
            n=args.n,
            scenario=args.scenario,
            file_type=args.file_type,
            output=output,
            textify=args.textify,
            assistant=args.assistant,
        )
    else:
        if not args.t2:
            parser.error("--t2 is required when --t1 is used")
        written = select_runs_by_timestamp(
            time_stamp_1=args.t1,
            time_stamp_2=args.t2,
            scenario=args.scenario,
            file_type=args.file_type,
            output=output,
            textify=args.textify,
            assistant=args.assistant,
        )

    dest = output or _default_output()
    print(f"Written {len(written)} file(s) to {dest}:")
    for p in written:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

"""
python -m Utilities.select_runs \
        --t1 2026-04-09T12:09:18 \
        --t2   2026-04-09T11:58:47 \
        --scenario 10 \
        --type Chat \
        --output out_old_LLM_10
"""
"""
python -m Utilities.select_runs \
        --t1 2026-04-16T04:40:48 \
        --t2   2026-04-16T04:40:39 \
        --scenario 10 \
        --type Chat \
        --output out_new_LLM_10
"""
"""
python -m Utilities.select_runs \
        --t1 2026-04-09T11:47:44 \
        --t2   2026-04-09T11:38:53 \
        --scenario 10 \
        --type Chat \
        --output out_old_EKGO_10
"""
"""
python -m Utilities.select_runs \
        --t1 2026-04-16T05:37:12 \
        --t2   2026-04-16T05:37:03 \
        --scenario 10 \
        --type Chat \
        --output out_new_EKGO_10
"""
"""
python -m Utilities.select_runs \
        --t1 2026-04-16T06:15:03 \
        --t2   2026-04-16T06:14:54 \
        --scenario 10 \
        --type Chat \
        --output out_new_new_EKGO_10
"""
"""
zip -r Archive2.zip out_old_EKGO_10 out_new_EKGO_10
zip -r Archive3.zip out_old_EKGO_10 out_new_new_EKGO_10
"""
"""
python -m Utilities.select_runs \
        --t1 2026-04-16T06:57:32 \
        --t2   2026-04-16T06:44:41 \
        --scenario 14 \
        --type Chat \
        --output out_new_EKGO_14 
zip -r Archive4.zip out_new_EKGO_14
"""
"""
python -m Utilities.select_runs --n 10 --scenario 2 --type Chat --output out_new_EKGO_2
zip -r Archive5.zip out_new_EKGO_2
"""
