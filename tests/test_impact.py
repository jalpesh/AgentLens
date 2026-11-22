import json

from agentlens.analytics.impact import before_after
from agentlens.schema import Event, Provider, Role, ToolInfo, ToolKind, Usage


def _session(sid, day, cost, fails=0, model="claude-opus-5", corrections=0):
    ts = f"2026-{day}T10:00:00"
    evs = [Event(session_id=sid, provider=Provider.CLAUDE_CODE, seq=0, ts=ts, role=Role.USER,
                 text="Add a login page to the app.")]
    seq = 1
    for i in range(2):
        evs.append(Event(session_id=sid, provider=Provider.CLAUDE_CODE, seq=seq, ts=ts,
                         role=Role.TOOL_RESULT,
                         tool=ToolInfo(kind=ToolKind.BASH, exit_code=1 if i < fails else 0)))
        seq += 1
    for _ in range(corrections):
        evs += [Event(session_id=sid, provider=Provider.CLAUDE_CODE, seq=seq, ts=ts,
                      role=Role.TOOL_CALL, tool=ToolInfo(kind=ToolKind.EDIT)),
                Event(session_id=sid, provider=Provider.CLAUDE_CODE, seq=seq + 1, ts=ts,
                      role=Role.USER, text="No, reuse the existing API.")]
        seq += 2
    evs.append(Event(session_id=sid, provider=Provider.CLAUDE_CODE, seq=seq, ts=ts,
                     role=Role.ASSISTANT, model=model, cost_usd=cost,
                     usage=Usage(input=int(cost * 1000))))
    return evs


def _rows(out):
    return {r["metric"]: r for r in out["rows"]}


def test_cheaper_smoother_sessions_after_the_change_are_better():
    ss = ([_session(f"b{i}", "10-01", 4.0, fails=2, corrections=1) for i in range(6)]
          + [_session(f"a{i}", "11-01", 1.0) for i in range(6)])
    out = before_after(ss, "2026-10-15", "repo-layer rule")
    r = _rows(out)
    assert out["available"] and out["before_sessions"] == 6 and out["after_sessions"] == 6
    assert r["Cost / session (median)"]["verdict"] == "better"
    assert r["Cost / session (median)"]["change_pct"] == -75.0
    assert r["Failed tool calls %"]["verdict"] == "better"
    assert r["Correction rate %"]["verdict"] == "better"
    assert r["Cost / session (median)"]["confidence"] == "low"  # < 20 per side
    json.dumps(out)


def test_too_few_sessions_on_one_side_gives_no_numbers():
    ss = [_session(f"b{i}", "10-01", 1.0) for i in range(10)] + [_session("a", "11-01", 1.0)]
    out = before_after(ss, "2026-10-15")
    assert not out["available"] and out["rows"] == []
    assert out["caveats"][0].startswith("Insufficient data")


def test_model_switch_is_flagged_as_a_confounder():
    ss = ([_session(f"b{i}", "10-01", 4.0) for i in range(5)]
          + [_session(f"a{i}", "11-01", 1.0, model="claude-haiku-5-5") for i in range(5)])
    out = before_after(ss, "2026-10-15")
    assert any("Main model changed" in c for c in out["caveats"])
    assert {r["confidence"] for r in out["rows"]} == {"low"}


def test_median_ignores_one_marathon_session():
    ss = ([_session(f"b{i}", "10-01", 1.0) for i in range(5)]
          + [_session(f"a{i}", "11-01", 1.0) for i in range(4)] + [_session("m", "11-02", 500.0)])
    assert _rows(before_after(ss, "2026-10-15"))["Cost / session (median)"]["verdict"] \
        == "no clear change"


def test_a_long_session_contributes_its_later_work_to_after():
    long = _session("L", "10-01", 1.0) + [
        Event(session_id="L", provider=Provider.CLAUDE_CODE, seq=98, ts="2026-11-05T10:00:00",
              role=Role.USER, text="Now add logout."),
        Event(session_id="L", provider=Provider.CLAUDE_CODE, seq=99, ts="2026-11-05T10:00:00",
              role=Role.ASSISTANT, model="claude-opus-5", cost_usd=1.0)]
    ss = [_session(f"b{i}", "10-01", 1.0) for i in range(5)] + [long] \
        + [_session(f"a{i}", "11-01", 1.0) for i in range(4)]
    out = before_after(ss, "2026-10-15")
    assert out["before_sessions"] == 6 and out["after_sessions"] == 5


def test_segments_without_a_prompt_are_not_counted():
    idle = [Event(session_id="i", provider=Provider.CLAUDE_CODE, seq=0, ts="2026-11-01T10:00:00",
                  role=Role.ASSISTANT, model="claude-opus-5", cost_usd=0.0)]
    ss = [_session(f"b{i}", "10-01", 1.0) for i in range(5)] + [idle] * 3 \
        + [_session(f"a{i}", "11-01", 1.0) for i in range(5)]
    assert before_after(ss, "2026-10-15")["after_sessions"] == 5
