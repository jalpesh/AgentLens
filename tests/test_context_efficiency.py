import json

from agentlens.analytics.context_efficiency import context_efficiency
from agentlens.schema import Event, Provider, Role, ToolInfo, ToolKind, Usage


def _ev(sid, seq, role, **kw):
    return Event(session_id=sid, provider=Provider.CLAUDE_CODE, seq=seq,
                 ts=f"2026-03-01T10:00:{seq:02d}", role=role, **kw)


def _read(sid, seq, path, nbytes=4000):
    return _ev(sid, seq, Role.TOOL_RESULT,
               tool=ToolInfo(kind=ToolKind.READ, target_path=path, output_bytes=nbytes))


def _opps(out):
    return {o["kind"]: o for o in out["opportunities"]}


def test_empty_is_unavailable():
    out = context_efficiency([])
    assert not out["measures"][0].available and out["opportunities"] == []


def test_reread_of_unchanged_file_counts_but_reread_after_edit_does_not():
    evs = [
        _ev("s", 1, Role.ASSISTANT, usage=Usage(input=1000), cost_usd=1.0),
        _read("s", 2, "src/a.py"),
        _read("s", 3, "src/a.py"),  # unchanged -> excess
        _ev("s", 4, Role.TOOL_CALL, tool=ToolInfo(kind=ToolKind.EDIT, target_path="src/a.py")),
        _read("s", 5, "src/a.py"),  # after edit -> legitimate
    ]
    out = context_efficiency([evs])
    o = _opps(out)["reread"]
    assert o["occurrences"] == 1 and o["tokens"] == 1000
    assert o["cost_usd"] == 1.0  # 1000 tokens x $0.001/token session price
    m = {x.label: x for x in out["measures"]}
    assert m["Context loaded"].value == 3000
    assert m["Context efficiency"].value == round(100 - 100 * 1000 / 3000, 1)


def test_generated_paths_are_reported_by_pattern_not_full_path():
    out = context_efficiency([[_read("s", 1, "web/dist/bundle.min.js", 40_000)]])
    o = _opps(out)["generated"]
    assert o["tokens"] == 10_000
    assert all("bundle" not in ex for ex in o["examples"])
    assert {s["category"] for s in out["sources"]} == {"generated files"}


def test_oversized_output_counts_only_the_part_above_the_cap():
    big = _ev("s", 1, Role.TOOL_RESULT,
              tool=ToolInfo(kind=ToolKind.BASH, output_bytes=4 * 15_000))
    small = _ev("s", 2, Role.TOOL_RESULT, tool=ToolInfo(kind=ToolKind.BASH, output_bytes=400))
    assert _opps(context_efficiency([[big, small]]))["oversized"]["tokens"] == 5_000


def test_same_block_pasted_in_two_sessions_is_flagged_once():
    block = "\n".join(f"2026-03-01 10:00:{i:02d} ERROR thing failed id={i}" for i in 40 * [0])
    block = block * 2
    p = lambda sid: _ev(sid, 1, Role.USER, text=f"fix this\n\n{block}")  # noqa: E731
    o = _opps(context_efficiency([[p("a")], [p("b")]]))["pasted"]
    assert o["occurrences"] == 1 and o["sessions"] == 1
    assert o["examples"][0].startswith("log block")


def test_clean_session_has_no_opportunities_and_serializes():
    evs = [_read("s", 1, "src/a.py"), _read("s", 2, "src/b.py"),
           _ev("s", 3, Role.USER, text="short prompt")]
    out = context_efficiency([evs])
    assert out["opportunities"] == []
    m = {x.label: x for x in out["measures"]}
    assert m["Context efficiency"].value == 100.0
    json.dumps({**out, "measures": [x.to_dict() for x in out["measures"]]})


def test_unpriced_session_costs_zero_not_nan():
    out = context_efficiency([[_read("s", 1, "a.py"), _read("s", 2, "a.py")]])
    assert _opps(out)["reread"]["cost_usd"] == 0.0


def test_compaction_transcript_replay_is_not_a_paste():
    replay = "\n".join(f"[{i}] user: please fix the build step number {i} again" for i in range(80))
    p = lambda sid: _ev(sid, 1, Role.USER, text=replay)  # noqa: E731
    assert "pasted" not in _opps(context_efficiency([[p("a")], [p("b")]]))
