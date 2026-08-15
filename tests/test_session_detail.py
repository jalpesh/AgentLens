"""Session drill-down tests.

Correctness only — this isn't a waste finding, so there's no suppression
logic to protect, just the usual "the numbers must add up and never lie"
bar every other rollup in this project is held to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.analytics.session_detail import build_session_detail  # noqa: E402
from generate import build  # noqa: E402

LOOPING = "66666666-6666-4666-8666-666666666666"
CONVERGED = "77777777-7777-4777-8777-777777777777"
CLEAN = "55555555-5555-4555-8555-555555555555"


@pytest.fixture(scope="module")
def sessions(tmp_path_factory) -> dict[str, list]:
    paths = build(tmp_path_factory.mktemp("cc") / "projects")
    a = ClaudeCodeAdapter()
    return {p.stem: list(a.parse(SourceFile.of(p))) for p in paths}


def test_empty_session_returns_empty_shape():
    d = build_session_detail([])
    assert d == {
        "cost_curve": [],
        "context_fill_curve": [],
        "models_used": [],
        "prompts": [],
    }


def test_cost_curve_is_monotonic_non_decreasing(sessions):
    d = build_session_detail(sessions[LOOPING])
    values = [row["cumulative_usd"] for row in d["cost_curve"]]
    assert values == sorted(values), "cumulative cost must never go down"


def test_context_fill_curve_is_clamped_at_100(sessions):
    for evs in sessions.values():
        d = build_session_detail(evs)
        for row in d["context_fill_curve"]:
            assert 0.0 <= row["pct"] <= 100.0


def test_models_used_sums_match_session_total(sessions):
    from agentlens.analytics import rollups

    evs = sessions[LOOPING]
    d = build_session_detail(evs)
    total_from_detail = sum(m["cost_usd"] for m in d["models_used"])
    total_session = rollups.summarize_session(evs).total_cost_usd
    assert round(total_from_detail, 4) == round(total_session, 4)


def test_converged_session_still_produces_a_full_curve(sessions):
    """Not a waste finding, so no suppression logic applies here — a
    converged session must still show its full cost/context story."""
    evs = sessions[CONVERGED]
    d = build_session_detail(evs)
    assert len(d["cost_curve"]) > 0
    assert len(d["prompts"]) > 0


def test_prompts_carry_estimated_tokens(sessions):
    d = build_session_detail(sessions[LOOPING])
    assert d["prompts"]
    for p in d["prompts"]:
        assert p["est_tokens"] > 0
        assert isinstance(p["has_lab_evidence"], bool)


def test_lab_evidence_flag_matches_a_real_finding(sessions):
    """The blind-retry fixture's first prompt is quoted by that detector's
    findings — the flag on the session prompt list must agree."""
    d = build_session_detail(sessions["11111111-1111-4111-8111-111111111111"])
    assert any(p["has_lab_evidence"] for p in d["prompts"])


def test_clean_session_detail_has_no_flagged_prompts(sessions):
    d = build_session_detail(sessions[CLEAN])
    assert all(not p["has_lab_evidence"] for p in d["prompts"])
