"""
ChatLog — writes a Telegram-style HTML chat log alongside the structured log.

Each agent gets its own bubble colour and alignment:
  • Saboteur      left  — red          (fault injector)
  • Assistant     left  — green        (diagnostic AI)
  • ServiceAgent  right — white/blue   (hands-on technician)
  • System        centre — grey pill   (orchestrator events)
"""
from __future__ import annotations

import html
from datetime import datetime


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    background: #0e1117;
    font-family: 'Segoe UI', system-ui, sans-serif;
    color: #e0e0e0;
}
.header {
    background: #1f2933;
    border-bottom: 1px solid #2e3d4f;
    padding: 14px 20px;
    position: sticky; top: 0; z-index: 10;
    display: flex; align-items: center; gap: 12px;
}
.header .icon { font-size: 1.8em; }
.header h1 { font-size: 1em; font-weight: 600; color: #cdd8e3; }
.header p  { font-size: 0.75em; color: #7a8fa6; margin-top: 2px; }
.chat {
    max-width: 780px;
    margin: 0 auto;
    padding: 18px 12px 40px;
    display: flex;
    flex-direction: column;
    gap: 6px;
}

/* ── row wrappers ── */
.row        { display: flex; align-items: flex-end; gap: 8px; }
.row.left   { flex-direction: row; }
.row.right  { flex-direction: row-reverse; }
.row.centre { justify-content: center; }

/* ── avatar ── */
.avatar {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1em; flex-shrink: 0;
}
.avatar.saboteur  { background: #5c1a1a; }
.avatar.assistant { background: #1a4a2e; }
.avatar.service   { background: #1a2e5c; }

/* ── bubble ── */
.bubble {
    max-width: 68%;
    padding: 9px 13px 7px;
    border-radius: 12px;
    font-size: 0.875em;
    line-height: 1.5;
    word-break: break-word;
}
.row.left  .bubble { border-bottom-left-radius:  3px; }
.row.right .bubble { border-bottom-right-radius: 3px; }

.bubble.saboteur  { background: #3d0f0f; border: 1px solid #7a2020; }
.bubble.assistant { background: #0f2d1a; border: 1px solid #1e6640; }
.bubble.service   { background: #0d1f3c; border: 1px solid #1e3a6e; }

/* ── sender label ── */
.sender {
    font-size: 0.72em; font-weight: 700;
    margin-bottom: 4px;
    text-transform: uppercase; letter-spacing: .04em;
}
.sender.saboteur  { color: #e57373; }
.sender.assistant { color: #66bb6a; }
.sender.service   { color: #64b5f6; }

/* ── timestamp ── */
.ts { font-size: 0.68em; color: #5a6a7a; text-align: right; margin-top: 4px; }

/* ── pre-formatted message body ── */
.body { white-space: pre-wrap; }

/* ── tag (action type badge) ── */
.tag {
    display: inline-block;
    background: #ffffff18;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.78em;
    font-weight: 600;
    margin-right: 5px;
    letter-spacing: .02em;
}

/* ── system pill ── */
.pill {
    background: #1c2533;
    border: 1px solid #2e3d4f;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.75em;
    color: #8899aa;
    text-align: center;
    max-width: 90%;
}

/* ── outcome colours ── */
.correct { color: #66bb6a; font-weight: 700; }
.partial { color: #ffa726; font-weight: 700; }
.wrong   { color: #ef5350; font-weight: 700; }
"""

_HTML_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Diagnostic Session Chat</title>
<style>
{css}
</style>
</head>
<body>
<div class="header">
  <div class="icon">🔧</div>
  <div>
    <h1>Diagnostic Session</h1>
    <p>{info}</p>
  </div>
</div>
<div class="chat">
"""

_HTML_FOOT = """\
</div>
</body>
</html>
"""


def _e(text: str) -> str:
    """HTML-escape a string."""
    return html.escape(str(text))


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


class ChatLog:
    """
    Writes an HTML chat log to *path*.

    Call the typed helpers (saboteur / assistant / service / system) to
    append messages, then call close() at the end of the session.
    """

    def __init__(self, path: str, session_info: str = "") -> None:
        self._f = open(path, "w", encoding="utf-8")
        self._f.write(_HTML_HEAD.format(css=_CSS, info=_e(session_info)))
        self._f.flush()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _left(self, role: str, emoji: str, sender_label: str, body_html: str) -> None:
        ts = _ts()
        self._f.write(
            f'<div class="row left">'
            f'<div class="avatar {role}">{emoji}</div>'
            f'<div class="bubble {role}">'
            f'<div class="sender {role}">{_e(sender_label)}</div>'
            f'{body_html}'
            f'<div class="ts">{ts}</div>'
            f'</div></div>\n'
        )
        self._f.flush()

    def _right(self, role: str, emoji: str, sender_label: str, body_html: str) -> None:
        ts = _ts()
        self._f.write(
            f'<div class="row right">'
            f'<div class="avatar {role}">{emoji}</div>'
            f'<div class="bubble {role}">'
            f'<div class="sender {role}">{_e(sender_label)}</div>'
            f'{body_html}'
            f'<div class="ts">{ts}</div>'
            f'</div></div>\n'
        )
        self._f.flush()

    def _centre(self, text: str) -> None:
        self._f.write(
            f'<div class="row centre">'
            f'<div class="pill">{_e(text)}</div>'
            f'</div>\n'
        )
        self._f.flush()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def system(self, text: str) -> None:
        """Centred grey pill — for orchestrator / meta events."""
        self._centre(text)

    def saboteur(self, fault_description: str) -> None:
        """Red left bubble — fault injection event."""
        body = f'<div class="body">💣 <b>Fault injected</b>\n{_e(fault_description)}</div>'
        self._left("saboteur", "💣", "Saboteur", body)

    def assistant_action(self, action_type: str, target: str, description: str) -> None:
        """Green left bubble — assistant suggesting a diagnostic action."""
        tag = f'<span class="tag">{_e(action_type)}</span>'
        body = (
            f'<div class="body">{tag}<b>{_e(target)}</b>'
            + (f'\n{_e(description)}' if description else '')
            + '</div>'
        )
        self._left("assistant", "🤖", "Diagnostic Assistant", body)

    def assistant_hypothesis(self, suspected: "set[str] | list[str]", explanation: str | None) -> None:
        """Green left bubble — assistant declaring a fault hypothesis."""
        comps = ", ".join(sorted(suspected))
        expl_html = f'\n\n{_e(explanation)}' if explanation else ''
        body = (
            f'<div class="body">💡 <b>Hypothesis</b>\n'
            f'Suspected: <b>{_e(comps)}</b>'
            f'{expl_html}</div>'
        )
        self._left("assistant", "🤖", "Diagnostic Assistant", body)

    def service_result(self, action_name: str, outcome: str, cost: "float | None" = None, cost_breakdown: "list[tuple[str, float]] | None" = None) -> None:
        """Blue right bubble — service agent reporting an action result."""
        if cost is not None:
            cost_html = f'<span style="font-size:0.8em;color:#7a9abf;"> · {cost:.0f}s</span>'
            if cost_breakdown and len(cost_breakdown) > 1:
                parts = " + ".join(
                    f"{name.replace('_', ' ')} {t:.0f}s"
                    for name, t in cost_breakdown
                )
                cost_html += (
                    f'<div style="font-size:0.70em;color:#4a5a6a;margin-top:1px;">'
                    f'{_e(parts)}</div>'
                )
        else:
            cost_html = ''
        body = (
            f'<div class="body"><b>{_e(action_name)}</b>{cost_html}\n{_e(outcome)}</div>'
        )
        self._right("service", "🔧", "Service Agent", body)

    def service_verification(self, outcome: str, narrative: str, cost: "float | None" = None) -> None:
        """Blue right bubble — verification result."""
        icons = {"correct": "✅", "partial": "⚠️", "wrong": "❌"}
        icon = icons.get(outcome, "❓")
        css = outcome if outcome in ("correct", "partial", "wrong") else ""
        cost_html = (
            f'<span style="font-size:0.8em;color:#7a9abf;"> · {cost:.0f}s</span>'
            if cost is not None else ''
        )
        body = (
            f'<div class="body">'
            f'{icon} <span class="{css}">{_e(outcome.upper())}</span>{cost_html}\n'
            f'{_e(narrative)}'
            f'</div>'
        )
        self._right("service", "🔧", "Service Agent", body)

    def close(self, summary: str = "") -> None:
        """Append a final summary pill and close the file."""
        if summary:
            self._centre(f"── Session ended ── {summary}")
        self._f.write(_HTML_FOOT)
        self._f.close()