"""Loop detector tests.

The load-bearing test in this file is `test_converged_iteration_is_silent`.
A loop and healthy debugging look almost identical right up until the moment
one of them works; a detector that can't tell them apart would fire on every
productive session and destroy the credibility of the whole report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.analytics.detectors import (  # noqa: E402
    CrossSessionLoopDetector,
    run_all,
    run_session,
    signature,
)
from generate import build  # noqa: E402

LOOPING = "66666666-6666-4666-8666-666666666666"
CONVERGED = "77777777-7777-4777-8777-777777777777"
RECURRING = "88888888-8888-4888-8888-888888888888"
CLEAN = "55555555-5555-4555-8555-555555555555"


@pytest.fixture(scope="module")
def sessions(tmp_path_factory) -> dict[str, list]:
    paths = build(tmp_path_factory.mktemp("cc") / "projects")
    a = ClaudeCodeAdapter()
    return {p.stem: list(a.parse(SourceFile.of(p))) for p in paths}


def names(findings) -> set[str]:
    return {f.detector for f in findings}


# --- signature normalisation ------------------------------------------------


def test_signature_ignores_volatile_details():
    """Two occurrences of the same failure differ in line numbers, timings and
    paths. If those defeat the fingerprint, no loop is ever detected."""
    a = (
        'File "/home/dev/app/tests/test_x.py", line 40, in test_v2\n'
        "AssertionError: expected True, got False (ran in 0.13s)"
    )
    b = (
        'File "/Users/other/proj/tests/test_x.py", line 87, in test_v2\n'
        "AssertionError: expected True, got False (ran in 2.90s)"
    )
    assert signature(a) == signature(b)


def test_signature_distinguishes_genuinely_different_failures():
    a = "AssertionError: expected True, got False"
    b = "TypeError: unsupported operand type(s) for +: 'int' and 'str'"
    assert signature(a) != signature(b)


def test_signature_handles_empty_input():
    assert signature(None) is None
    assert signature("") is None
    assert signature("   ") is None


# --- the detector itself ----------------------------------------------------


def test_loop_fires_on_repeated_identical_failure(sessions):
    findings = run_session(sessions[LOOPING])
    assert "loop" in names(findings)
    f = next(x for x in findings if x.detector == "loop")
    assert f.occurrences >= 3
    assert f.evidence
    assert f.wasted_usd > 0


def test_loop_blames_the_model_not_the_user(sessions):
    """Every other detector implicitly blames the user. This one must not —
    it's the thing that makes the rest of the report trustworthy."""
    f = next(x for x in run_session(sessions[LOOPING]) if x.detector == "loop")
    assert f.blames_model is True
    assert "not a badly worded prompt" in f.mechanism


def test_loop_fires_even_though_the_prompts_are_good(sessions):
    """The looping fixture's prompts name the file AND the acceptance check.
    A loop is not a prompting failure, so the well-written-prompt detectors
    should stay quiet while the loop detector fires."""
    found = names(run_session(sessions[LOOPING]))
    assert "loop" in found
    assert "vague_instruction" not in found


def test_converged_iteration_is_silent(sessions):
    """THE load-bearing test.

    Three failures then a pass is successful debugging. If anything fires here,
    the detector is punishing people for fixing bugs.
    """
    found = names(run_session(sessions[CONVERGED]))
    assert found == set(), f"converged iteration must be silent, got {found}"


def test_clean_session_still_silent_with_loop_detectors_added(sessions):
    assert run_session(sessions[CLEAN]) == []


def test_escalation_ladder_changes_with_severity(sessions):
    f = next(x for x in run_session(sessions[LOOPING]) if x.detector == "loop")
    two = f.advice_for(2)
    three = f.advice_for(3)
    six = f.advice_for(6)
    assert two != three != six
    assert "exact error" in two           # 2 → better prompt
    assert "plan" in three.lower()        # 3 → plan mode
    assert "skill" in six.lower()         # 5+ → write it down


def test_advice_falls_back_to_prescription_for_normal_detectors(sessions):
    """Detectors without a ladder must be completely unaffected by it."""
    findings = run_session(sessions["33333333-3333-4333-8333-333333333333"])
    f = next(x for x in findings if x.detector == "vague_instruction")
    assert f.escalation == []
    assert f.advice_for() == f.prescription


# --- cross-session ----------------------------------------------------------


def test_cross_session_detector_needs_two_sessions(sessions):
    """One session hitting an error repeatedly is a loop, not a recurrence.
    Only the same signature in *different* sessions justifies "write a skill"."""
    only_one = [sessions[LOOPING]]
    assert CrossSessionLoopDetector().run_global(only_one) is None


def test_cross_session_detector_fires_across_sessions(sessions):
    both = [sessions[LOOPING], sessions[RECURRING]]
    f = CrossSessionLoopDetector().run_global(both)
    assert f is not None
    assert f.occurrences >= 2
    assert "skill" in f.advice_for().lower()


def test_cross_session_finding_carries_a_dollar_figure(sessions):
    """Every finding needs a cost — not only for persuasiveness, but because
    the report suppresses low-value findings, so a $0 finding is an invisible
    one."""
    f = CrossSessionLoopDetector().run_global([sessions[LOOPING], sessions[RECURRING]])
    assert f.wasted_usd > 0
    assert all(e.cost_usd >= 0 for e in f.evidence)


def test_run_all_includes_global_findings(sessions):
    all_findings = run_all(list(sessions.values()))
    assert "cross_session_loop" in names(all_findings)


def test_run_all_still_ranks_by_cost(sessions):
    costs = [f.wasted_usd for f in run_all(list(sessions.values()))]
    assert costs == sorted(costs, reverse=True)


def test_broken_global_detector_cannot_break_the_report(sessions, monkeypatch):
    from agentlens.analytics import detectors

    class Exploding:
        name = "boom"

        def run_global(self, sessions):
            raise RuntimeError("kaboom")

    monkeypatch.setattr(detectors, "GLOBAL_DETECTORS", [Exploding()])
    assert run_all(list(sessions.values())), "per-session findings must survive"


# --- evidence ---------------------------------------------------------------


def test_loop_evidence_carries_full_error_text(sessions):
    """`quote` is ellipsised for display; `full_text` is what the Prompt Lab
    and the skills generator actually need."""
    f = next(x for x in run_session(sessions[LOOPING]) if x.detector == "loop")
    ev = f.evidence[0]
    assert ev.full_text
    assert len(ev.full_text) >= len(ev.quote.rstrip("…"))


def test_error_text_only_stored_for_failures(sessions):
    """Successful tool output must never be persisted — see ToolInfo.error_text."""
    for evs in sessions.values():
        for e in evs:
            if e.tool and e.tool.error_text:
                assert e.tool.failed, "error_text set on a successful tool result"


# --- attempt detail (Phase 4, section B) ------------------------------------


def test_loop_attempts_are_ordered(sessions):
    f = next(x for x in run_session(sessions[LOOPING]) if x.detector == "loop")
    assert f.attempts
    stamps = [a.when for a in f.attempts]
    assert stamps == sorted(stamps)


def test_loop_synthetic_fixture_with_five_repeats_produces_five_attempts(sessions):
    """The looping fixture fires the same failure 5 times — every occurrence
    must show up as its own attempt record."""
    f = next(x for x in run_session(sessions[LOOPING]) if x.detector == "loop")
    assert len(f.attempts) == 5


def test_loop_first_attempt_costs_less_than_the_total_across_all_attempts(sessions):
    """Sanity check: the first attempt is one window out of several: it must
    cost less than all the attempts summed together."""
    f = next(x for x in run_session(sessions[LOOPING]) if x.detector == "loop")
    total = sum(a.cost_usd for a in f.attempts)
    assert f.attempts[0].cost_usd < total


def test_loop_attempts_carry_the_prompt_that_was_in_force(sessions):
    f = next(x for x in run_session(sessions[LOOPING]) if x.detector == "loop")
    for a in f.attempts:
        assert "verify_signature" in a.prompt_excerpt


def test_churn_detector_produces_attempts_too(sessions):
    """A weaker independent signal from LoopDetector, same attempt shape."""
    findings = run_session(sessions[LOOPING])
    churn = next((x for x in findings if x.detector == "target_churn"), None)
    if churn is None:
        pytest.skip("target_churn did not fire on this fixture")
    assert churn.attempts
    total = sum(a.cost_usd for a in churn.attempts)
    assert churn.attempts[0].cost_usd <= total


def test_cross_session_loop_has_one_attempt_per_session(sessions):
    f = CrossSessionLoopDetector().run_global([sessions[LOOPING], sessions[RECURRING]])
    assert f.attempts
    assert len(f.attempts) == f.occurrences


def test_merge_concatenates_attempts_across_sessions():
    """`merge()` combines per-session findings of the same detector — the
    attempts list must be carried along, not dropped."""
    from agentlens.analytics.detectors.base import Attempt, Finding, merge

    def mk(n):
        return Finding(
            detector="loop", title="t", prescription="p", mechanism="m",
            wasted_usd=1.0, occurrences=3,
            attempts=[Attempt(when=f"2026-01-0{n}", prompt_excerpt="x", tokens=1,
                               cost_usd=0.1, calls=1)],
        )

    merged = merge([mk(1), mk(2)])
    assert len(merged) == 1
    assert len(merged[0].attempts) == 2


def test_evidence_carries_target_path_when_the_underlying_event_named_one(sessions):
    """Metadata AgentLens already has — a tool's own `target_path` — exposed
    on the evidence so the dashboard knows which file a citation drill-down
    should ask about. This is not the disk-read feature itself."""
    findings = run_session(sessions[LOOPING])
    churn = next((x for x in findings if x.detector == "target_churn"), None)
    if churn is None:
        pytest.skip("target_churn did not fire on this fixture")
    assert any(e.target_path for e in churn.evidence)


# --- regression: cross-session waste cannot exceed real spend ---------------
#
# A real user reported the Engineering tab's "Loop Cost" tile showing ~$19k
# against roughly $650 of actual billed spend. Root cause: CrossSessionLoopDetector
# attributed a session's FULL cost once per recurring signature, with no
# bookkeeping across signatures — a handful of generic, weakly-specific
# failure shapes that all happened to touch mostly the same sessions each
# re-added that session pool's cost on top of the others. These tests build
# that exact shape directly (many sessions, several near-generic recurring
# errors) and assert the fixed detector can no longer produce a total larger
# than what was actually spent.


def _mk_event(session_id, seq, ts, cost, error_text=None, target_path=None):
    from agentlens.schema import Event, Provider, Role, ToolInfo, Usage
    from agentlens.schema import ToolKind as TK

    tool = None
    role = Role.ASSISTANT
    if error_text is not None:
        tool = ToolInfo(kind=TK.BASH, exit_code=1, error_text=error_text,
                         target_path=target_path)
        role = Role.TOOL_RESULT
    return Event(
        session_id=session_id, provider=Provider.CLAUDE_CODE, seq=seq, ts=ts,
        role=role, usage=Usage(input=1000, output=200), cost_usd=cost, tool=tool,
    )


def _session_with_generic_failures(session_id, month, n_events=20, cost_per_event=0.5):
    """A session whose sessionwide cost is `n_events * cost_per_event`, with a
    couple of failures whose text is generic enough that, pre-fix, it would
    fingerprint the same across unrelated sessions."""
    from agentlens.schema import Event, Provider, Role

    events = []
    for i in range(n_events):
        events.append(Event(
            session_id=session_id, provider=Provider.CLAUDE_CODE, seq=i,
            ts=f"2026-{month:02d}-01T00:00:{i:02d}", role=Role.ASSISTANT,
            usage=None, cost_usd=cost_per_event,
        ))
    # Two distinct-but-both-generic failure shapes, so >1 signature ends up
    # pointing at the same session pool — this is what compounded the bug.
    events.append(_mk_event(session_id, n_events, f"2026-{month:02d}-01T00:01:00",
                             0.0, error_text="Error: command failed", target_path="run.sh"))
    events.append(_mk_event(session_id, n_events + 1, f"2026-{month:02d}-01T00:01:01",
                             0.0, error_text="exit status 1", target_path="run.sh"))
    return events


def test_cross_session_waste_cannot_exceed_actual_spend():
    """The regression itself: many sessions, each with the same two generic
    failure shapes. Before the fix, wasted_usd for the cross-session finding
    was a multiple of the real total; after the fix it's bounded by it."""
    sessions = [_session_with_generic_failures(f"sess-{i}", month=(i % 12) + 1)
                for i in range(15)]
    total_spend = sum(e.cost_usd for evs in sessions for e in evs)

    f = CrossSessionLoopDetector().run_global(sessions)
    assert f is not None
    assert f.wasted_usd <= total_spend + 1e-9, (
        f"cross-session loop waste (${f.wasted_usd:.2f}) exceeded total spend "
        f"(${total_spend:.2f}) — this is the exact shape of the reported bug"
    )


def test_generic_short_failures_do_not_form_a_signature():
    """Root-cause fix: text that strips down to near-nothing must not become
    a trusted signature at all, or unrelated sessions falsely "match"."""
    assert signature("Error: command failed") is None or len(
        signature("Error: command failed") or ""
    ) >= 8
    assert signature("1") is None
    assert signature("exit 1") is None


def test_engineering_loop_cost_tile_is_capped_at_total_spend():
    """Defense in depth: even if a future detector bug re-inflates a single
    finding, the Engineering tab's tile can never show more than what these
    sessions actually cost, because it's clamped at the rollup layer too."""
    from agentlens.analytics import rollups
    from agentlens.analytics.detectors.base import Evidence, Finding

    sessions = [_session_with_generic_failures(f"sess-{i}", month=(i % 12) + 1)
                for i in range(5)]
    total_spend = sum(e.cost_usd for evs in sessions for e in evs)

    # Simulate a detector bug: a finding claiming far more waste than exists.
    bogus = Finding(
        detector="cross_session_loop", title="t", prescription="p", mechanism="m",
        wasted_usd=total_spend * 50, occurrences=5,
        evidence=[Evidence(session_id="sess-0", ts="2026-01-01", quote="x")],
    )
    metrics = rollups.engineering_metrics(sessions, [bogus])
    assert metrics["loop_cost_usd"] <= round(total_spend, 4) + 1e-6


# --- regression: a single session's finding cannot exceed that session's ----
# --- own spend (found via real usage data, 2026-08-16) ----------------------
#
# The three regressions above fixed CrossSessionLoopDetector and two
# downstream *aggregate* rollups. They did not fix — and real usage data
# immediately exposed — that `LoopDetector` and `SameTargetChurnDetector` each
# sum a separate `cost_between(...)` window per signature / per churned file
# within ONE session, and those windows can overlap. A session with three
# overlapping churned files (or three overlapping failure signatures) had its
# overlapping events counted once per file/signature, so a single finding
# could report several times what that one session actually cost — e.g.
# target_churn reporting $8,666 against a dataset that only ever spent
# $3,298 total. These tests build that exact overlap shape.


def _session_with_overlapping_churn(session_id, n_files=4, edits_per_file=5,
                                     cost_per_event=1.0):
    """One session, several files, each edited `edits_per_file` times with a
    failure between each edit — but the files' edit spans all overlap (they're
    interleaved in the same stretch of the session), which is exactly the
    shape that let each file's window double-count the others' events."""
    from agentlens.schema import Event, Provider, Role, ToolInfo
    from agentlens.schema import ToolKind as TK

    events = []
    seq = 0
    # Interleave edits to N files across the same seq range, so every file's
    # [first_edit, last_edit] window spans nearly the whole session.
    for _round in range(edits_per_file):
        for fi in range(n_files):
            events.append(Event(
                session_id=session_id, provider=Provider.CLAUDE_CODE, seq=seq,
                ts=f"2026-01-01T00:00:{seq:02d}", role=Role.TOOL_CALL,
                usage=None, cost_usd=cost_per_event,
                tool=ToolInfo(kind=TK.EDIT, target_path=f"src/file{fi}.py"),
            ))
            seq += 1
            events.append(Event(
                session_id=session_id, provider=Provider.CLAUDE_CODE, seq=seq,
                ts=f"2026-01-01T00:00:{seq:02d}", role=Role.TOOL_RESULT,
                usage=None, cost_usd=0.0,
                tool=ToolInfo(kind=TK.BASH, exit_code=1,
                               error_text=f"AssertionError: file{fi} still broken"),
            ))
            seq += 1
    return events


def test_target_churn_cannot_exceed_this_sessions_own_spend():
    from agentlens.analytics.detectors import SameTargetChurnDetector

    events = _session_with_overlapping_churn("sess-churn")
    session_cost = sum(e.cost_usd for e in events)

    f = SameTargetChurnDetector().run(events)
    assert f is not None
    assert f.wasted_usd <= session_cost + 1e-9, (
        f"target_churn waste (${f.wasted_usd:.2f}) exceeded this session's own "
        f"spend (${session_cost:.2f}) — the exact shape found in real usage data"
    )


def _session_with_overlapping_loops(session_id, n_signatures=3, repeats=4,
                                     cost_per_event=1.0):
    """One session, several distinct (specific-enough) failure signatures,
    interleaved across the same seq range so each signature's occurrence
    window overlaps the others'."""
    from agentlens.schema import Event, Provider, Role, ToolInfo
    from agentlens.schema import ToolKind as TK

    # Distinguish signatures by a *word*, not a digit — `signature()` strips
    # standalone digits as volatile (line numbers, counts), so an index alone
    # would collapse every "signature" into the same one and the fixture
    # would fail to reproduce the overlap this test exists to catch.
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    events = []
    seq = 0
    for _round in range(repeats):
        for si in range(n_signatures):
            events.append(Event(
                session_id=session_id, provider=Provider.CLAUDE_CODE, seq=seq,
                ts=f"2026-01-01T00:00:{seq:02d}", role=Role.ASSISTANT,
                usage=None, cost_usd=cost_per_event,
            ))
            seq += 1
            events.append(Event(
                session_id=session_id, provider=Provider.CLAUDE_CODE, seq=seq,
                ts=f"2026-01-01T00:00:{seq:02d}", role=Role.TOOL_RESULT,
                usage=None, cost_usd=0.0,
                tool=ToolInfo(kind=TK.BASH, exit_code=1,
                               error_text=f"AssertionError: {words[si % len(words)]} target still broken"),
            ))
            seq += 1
    return events


def test_loop_detector_cannot_exceed_this_sessions_own_spend():
    from agentlens.analytics.detectors import LoopDetector

    events = _session_with_overlapping_loops("sess-loop")
    session_cost = sum(e.cost_usd for e in events)

    f = LoopDetector().run(events)
    assert f is not None
    assert f.wasted_usd <= session_cost + 1e-9, (
        f"loop waste (${f.wasted_usd:.2f}) exceeded this session's own spend "
        f"(${session_cost:.2f}) — the exact shape found in real usage data"
    )


def test_run_all_caps_every_individual_finding_at_total_spend():
    """The central backstop in `run_all()`: even a detector bug we haven't
    found yet cannot produce a finding costing more than every session in the
    report combined — not just the aggregate tiles downstream, the finding
    itself."""
    sessions = [_session_with_overlapping_churn(f"sess-{i}") for i in range(3)]
    total_spend = sum(e.cost_usd for evs in sessions for e in evs)

    findings = run_all(sessions)
    for f in findings:
        assert f.wasted_usd <= total_spend + 1e-9, (
            f"{f.detector} finding (${f.wasted_usd:.2f}) exceeded total analyzed "
            f"spend (${total_spend:.2f})"
        )
