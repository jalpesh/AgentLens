"""Claude Code status line integration — the live, ambient counterpart to the
post-hoc `waste` report.

## Why this exists

Every other surface in AgentLens is retrospective: you finish a session (or a
week of them), then run a command or open the dashboard to find out what it
cost you and why. That's the right shape for "what happened" — but it means
the earliest a bad habit can reach you is after it already happened.

Claude Code's status line hook (https://code.claude.com/docs/en/statusline)
runs a script on (roughly) every turn and shows whatever it prints at the
bottom of the terminal. It hands the script `transcript_path` — the exact
JSONL file `ClaudeCodeAdapter` already knows how to read. That means this
module adds no new detection logic at all: it re-runs the same per-session
detectors used everywhere else, just pointed at a session that is still being
written to instead of a finished one.

## The one thing that's genuinely different about running live

A post-hoc finding knows how the story ended — `LoopDetector`'s convergence
suppression can look ahead and see whether the model eventually got it right,
and stays quiet if so. A live view can't look ahead; it only knows what has
happened *so far*. A loop flagged here might resolve on the very next turn.
That's an inherent property of "live," not a bug — the wording in
`render_line` below says "heads up," never "verdict," on purpose.

## The contract this script must honor

Claude Code renders whatever this prints, on a timer, in the UI. That means:
this must never raise, never hang, never do anything slow enough to be
noticeable, and never touch the network or write to disk. A crashing status
line script doesn't show an error — it just breaks the status bar, which is
a worse failure than a wrong or missing number. Every code path here is
wrapped so the worst case is "print something minimal and exit 0."
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from .adapters.base import SourceFile
from .adapters.claude_code import ClaudeCodeAdapter
from .analytics.detectors import run_session
from .schema import Event

#: Bound how much of a very long session's events we re-analyze on every
#: turn. Detectors only need recent context to catch a live loop/churn/vague
#: prompt forming — the same "don't turn a UI action into reading gigabytes"
#: discipline as the citation feature's read cap (see web/server.py). A loop
#: that started 2,000 events ago and is still ongoing will still show once
#: its most recent occurrences fall inside this window.
_MAX_EVENTS = 2000

#: Detectors worth interrupting the baseline line for, in priority order
#: when more than one fires on the same turn. Loop-family first (the
#: detector that blames the model, not the user, and the one with an
#: escalation ladder) — matches the ranking `run_all` already uses
#: elsewhere (highest `wasted_usd`/`occurrences` wins), just pre-filtered to
#: the handful that make sense to show mid-session rather than a full report.
_LIVE_WORTHY = {"loop", "target_churn", "vague_instruction", "output_flood", "model_mismatch"}


def _money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 else f"${x:.4f}"


def _read_events(transcript_path: str) -> list[Event]:
    p = Path(transcript_path)
    if not p.exists():
        return []
    events = list(ClaudeCodeAdapter().parse(SourceFile.of(p)))
    if len(events) > _MAX_EVENTS:
        events = events[-_MAX_EVENTS:]
    return events


def _pick_live_finding(events: list[Event]):
    if not events:
        return None
    findings = [f for f in run_session(events) if f.detector in _LIVE_WORTHY]
    if not findings:
        return None
    findings.sort(key=lambda f: (-f.wasted_usd, -f.occurrences))
    return findings[0]


def render_line(payload: dict) -> str:
    """Build the status line text from Claude Code's stdin payload.

    Always returns *something* — never raises. Two possible shapes:

    * Baseline only (nothing live-worthy right now): model, session cost,
      context-window fill. The same kind of thing ccusage's statusline
      shows — this part isn't the differentiator, it's the floor that
      keeps the line from looking broken most of the time (this tool is
      deliberately quiet when there's nothing to say — an empty status
      line would just look like it stopped working).
    * Baseline, plus a second line with the live coaching signal, when one
      is currently active. This part is what a raw usage ticker can't do.
    """
    model = (payload.get("model") or {}).get("display_name") or "?"
    cost = (payload.get("cost") or {}).get("total_cost_usd")
    ctx_pct = (payload.get("context_window") or {}).get("used_percentage")

    parts = [f"[{model}]"]
    if isinstance(cost, (int, float)):
        parts.append(_money(cost))
    if isinstance(ctx_pct, (int, float)):
        parts.append(f"{ctx_pct:.0f}% ctx")
    baseline = " · ".join(parts) + "  (agentlens)"

    transcript_path = payload.get("transcript_path")
    if not transcript_path:
        return baseline

    try:
        events = _read_events(transcript_path)
        finding = _pick_live_finding(events)
    except Exception:  # noqa: BLE001 — never let a bad transcript break the status line
        return baseline

    if finding is None:
        return baseline

    # "heads up", not "verdict" — see module docstring. A live finding can
    # still resolve on the next turn; convergence suppression only applies
    # in hindsight, which this view doesn't have.
    return f"{baseline}\n⚠ {finding.title} — {finding.prescription}"


def _print_setup_instructions() -> None:
    import shutil

    exe = shutil.which("agentlens") or "agentlens"
    print(
        "agentlens statusline is a Claude Code status line script — it isn't meant to be "
        "run directly.\n\n"
        "Add this to ~/.claude/settings.json (or your project's .claude/settings.json):\n\n"
        "  {\n"
        '    "statusLine": {\n'
        '      "type": "command",\n'
        f'      "command": "{exe} statusline"\n'
        "    }\n"
        "  }\n\n"
        "Leave refreshInterval unset — this only needs to re-run when the conversation "
        "changes, which Claude Code already triggers on its own. See "
        "https://code.claude.com/docs/en/statusline for the full mechanism."
    )


def main() -> int:
    if sys.stdin.isatty():
        _print_setup_instructions()
        return 0
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        line = render_line(payload if isinstance(payload, dict) else {})
    except Exception:  # noqa: BLE001 — see module docstring: never break the status bar
        line = "(agentlens)"
    print(line)
    return 0
