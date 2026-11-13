"""Regression tests for the "Reclaimable 100%" overstatement (2026-11-13).

Each test pins one way a detector used to bill far more than the evidence
supports on real data.
"""

from agentlens.analytics.detectors import run_all_scored
from agentlens.analytics.detectors.context import ModelMismatchDetector
from agentlens.analytics.detectors.loops import LoopDetector
from agentlens.analytics.detectors.output_flood import OutputFloodDetector
from agentlens.schema import Event, Provider, Role, ToolInfo, ToolKind, Usage

MODEL = "claude-opus-4-1"


def _ev(seq, role, sid="s", **kw):
    return Event(session_id=sid, provider=Provider.CLAUDE_CODE, seq=seq,
                 ts=f"2026-03-01T10:{seq // 60:02d}:{seq % 60:02d}", role=role, **kw)


def _turn(seq, ctx=1000, cost=0.01):
    return _ev(seq, Role.ASSISTANT, model=MODEL, cost_usd=cost,
               usage=Usage(input=10, cache_read=ctx, output=10))


def _fail(seq):
    return _ev(seq, Role.TOOL_RESULT, tool=ToolInfo(
        kind=ToolKind.BASH, exit_code=1, error_text="FileNotFoundError: no such file 'x'"))


def test_same_error_far_apart_is_not_a_loop():
    evs, seq = [], 0
    for _ in range(3):
        evs.append(_fail(seq))
        seq += 1
        for _ in range(30):  # 30 assistant turns between repeats
            evs.append(_turn(seq))
            seq += 1
    assert LoopDetector().run(evs) is None


def test_same_error_back_to_back_is_still_a_loop():
    evs = [e for i in range(4) for e in (_fail(2 * i), _turn(2 * i + 1))]
    assert LoopDetector().run(evs) is not None


def test_flood_resends_stop_at_compaction():
    big = _ev(1, Role.TOOL_RESULT, tool=ToolInfo(kind=ToolKind.BASH, output_bytes=200_000))
    evs = [_turn(0, ctx=60_000), big, *(_turn(s, ctx=60_000) for s in range(2, 5)),
           _ev(5, Role.COMPACTION), *(_turn(s, ctx=60_000) for s in range(6, 50))]
    f = OutputFloodDetector().run(evs)
    assert "on 3 later turn(s)" in f.evidence[0].detail


def test_concurrent_floods_cannot_exceed_turn_context():
    floods = [_ev(s, Role.TOOL_RESULT, tool=ToolInfo(kind=ToolKind.BASH, output_bytes=200_000))
              for s in range(1, 11)]
    evs = [*floods, *(_turn(s, ctx=100_000) for s in range(11, 21))]
    f = OutputFloodDetector().run(evs)
    per_turn = f.wasted_tokens / 10
    assert per_turn <= 100_010


def test_reclaimable_counts_overlapping_findings_once_per_session():
    evs = [e for i in range(6) for e in (_fail(2 * i), _turn(2 * i + 1, cost=1.0))]
    findings, reclaimable = run_all_scored([evs])
    spend = sum(e.cost_usd for e in evs)
    assert reclaimable <= spend
    assert reclaimable <= max(f.wasted_usd for f in findings) + 1e-9


def test_mismatch_skips_turns_too_big_for_the_cheaper_model():
    evs = [_ev(0, Role.USER, text="rename foo to bar"), _turn(1, ctx=600_000, cost=5.0)]
    assert ModelMismatchDetector().run(evs) is None
