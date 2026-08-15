"""Detector tests.

Two properties matter equally, and the second is the one that gets neglected:

1. Each detector fires on the pattern it targets.
2. **Detectors stay silent on a well-run session.** A tool that finds something
   wrong with everything gets closed and never reopened. The clean-session test
   is the guard against that, and it is not optional.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.analytics.detectors import run_all, run_session  # noqa: E402
from generate import build  # noqa: E402


@pytest.fixture(scope="module")
def sessions(tmp_path_factory) -> dict[str, list]:
    paths = build(tmp_path_factory.mktemp("cc") / "projects")
    a = ClaudeCodeAdapter()
    out = {}
    for p in paths:
        evs = list(a.parse(SourceFile.of(p)))
        out[p.stem] = evs
    return out


SID = {
    "blind_retry": "11111111-1111-4111-8111-111111111111",
    "output_flood": "22222222-2222-4222-8222-222222222222",
    "vague": "33333333-3333-4333-8333-333333333333",
    "compaction": "44444444-4444-4444-8444-444444444444",
    "clean": "55555555-5555-4555-8555-555555555555",
}


def names(findings) -> set[str]:
    return {f.detector for f in findings}


def test_blind_retry_fires_on_frustration_without_evidence(sessions):
    """The flagship detector.

    A blind retry usually shares almost NO vocabulary with the original ask
    ("still not working" vs "the webhook handler is broken"), so a purely
    lexical-similarity implementation misses the most common — and most
    expensive — form of it. This test exists because an earlier build did.
    """
    f = run_session(sessions[SID["blind_retry"]])
    assert "blind_retry" in names(f)
    finding = next(x for x in f if x.detector == "blind_retry")
    assert finding.wasted_usd > 0
    assert finding.evidence, "must quote the user's own prompt"
    assert any("still not working" in e.quote for e in finding.evidence)


def test_blind_retry_ignores_prompts_that_carry_evidence(sessions):
    """The final turn in that session supplies a command, output and an
    expectation. That is the behaviour we want — never penalise it."""
    finding = next(
        x for x in run_session(sessions[SID["blind_retry"]]) if x.detector == "blind_retry"
    )
    assert not any("AssertionError" in e.quote for e in finding.evidence)


def test_output_flood_fires_and_counts_resends(sessions):
    f = run_session(sessions[SID["output_flood"]])
    assert "output_flood" in names(f)
    finding = next(x for x in f if x.detector == "output_flood")
    assert "npm ls --all" in finding.evidence[0].quote
    assert "re-sent as input on" in finding.evidence[0].detail
    assert finding.wasted_tokens > 10_000


def test_vague_instruction_fires(sessions):
    f = run_session(sessions[SID["vague"]])
    assert "vague_instruction" in names(f)
    finding = next(x for x in f if x.detector == "vague_instruction")
    assert "clean up the auth stuff" in finding.evidence[0].quote


def test_whole_file_reread_fires(sessions):
    assert "whole_file_reread" in names(run_session(sessions[SID["vague"]]))


def test_compaction_thrash_fires(sessions):
    assert "compaction_thrash" in names(run_session(sessions[SID["compaction"]]))


def test_clean_session_produces_no_findings(sessions):
    """The most important test in the suite."""
    f = run_session(sessions[SID["clean"]])
    assert f == [], f"clean session must stay silent, got {names(f)}"


def test_every_finding_has_evidence_cost_and_a_fix(sessions):
    """The three non-negotiables from detectors/base.py, enforced."""
    findings = run_all(list(sessions.values()))
    assert findings
    for f in findings:
        assert f.prescription and len(f.prescription) > 20, f"{f.detector}: no concrete fix"
        assert f.mechanism, f"{f.detector}: no explanation of why it costs"
        assert f.evidence, f"{f.detector}: no evidence"
        assert f.severity in ("low", "medium", "high")
        for e in f.evidence:
            assert e.quote, f"{f.detector}: evidence must quote something"


def test_findings_are_ranked_by_cost(sessions):
    findings = run_all(list(sessions.values()))
    costs = [f.wasted_usd for f in findings]
    assert costs == sorted(costs, reverse=True)


def test_a_broken_detector_cannot_break_the_report(sessions, monkeypatch):
    from agentlens.analytics import detectors

    class Exploding:
        name = "boom"

        def run(self, events):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(detectors, "DETECTORS", [Exploding(), *detectors.DETECTORS])
    assert run_session(sessions[SID["vague"]]), "other detectors must still report"
