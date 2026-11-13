"""Claude Code status line integration.

Covers the two things that actually matter for a script something else's UI
renders on a timer: it must always print *something* rather than raise, and
the live signal must be the same detector output used everywhere else in
AgentLens, not a separate implementation that could drift from it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from generate import build  # noqa: E402

from agentlens.statusline import render_line  # noqa: E402

#: Matches the hardcoded session ids in fixtures/generate.py's
#: `session_loop`/`session_clean` builders.
_LOOPING = "66666666-6666-4666-8666-666666666666"
_CLEAN = "55555555-5555-4555-8555-555555555555"


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    target = tmp_path_factory.mktemp("statusline")
    build(target)
    return target


def _path_for(fixture_dir: Path, session_id: str) -> str:
    return str(next(fixture_dir.rglob(f"{session_id}.jsonl")))


def test_baseline_line_with_no_transcript_path():
    line = render_line({"model": {"display_name": "claude-sonnet-4-5"},
                         "cost": {"total_cost_usd": 1.23},
                         "context_window": {"used_percentage": 42}})
    assert "claude-sonnet-4-5" in line
    assert "1.23" in line
    assert "42% ctx" in line
    assert "\n" not in line, "no live finding — must stay a single line"


def test_missing_fields_degrade_gracefully_not_crash():
    assert render_line({}) != ""
    assert render_line({"model": None, "cost": None}) != ""


def test_bad_transcript_path_falls_back_to_baseline():
    line = render_line({
        "model": {"display_name": "claude-sonnet-4-5"},
        "transcript_path": "/definitely/not/a/real/session.jsonl",
    })
    assert "\n" not in line
    assert "claude-sonnet-4-5" in line


def test_live_loop_signal_appears_for_a_looping_session(fixture_dir):
    line = render_line({
        "model": {"display_name": "claude-sonnet-4-5"},
        "cost": {"total_cost_usd": 0.5},
        "transcript_path": _path_for(fixture_dir, _LOOPING),
    })
    assert "\n" in line, "a real in-progress loop must add the second line"
    assert "⚠" in line
    # "heads up" phrasing, not a verdict — a live finding can still resolve
    # on the next turn, unlike the post-hoc report which knows the ending.
    assert "verdict" not in line.lower()


def test_clean_session_stays_single_line(fixture_dir):
    """No crying wolf here either: a session with nothing detector-worthy
    must render exactly like the no-transcript case, not add noise."""
    line = render_line({
        "model": {"display_name": "claude-sonnet-4-5"},
        "transcript_path": _path_for(fixture_dir, _CLEAN),
    })
    assert "\n" not in line
