import json

from agentlens.analytics.prompt_patterns import prompt_patterns
from agentlens.schema import Event, Provider, Role, ToolInfo, ToolKind

RULE = "Use the existing repository layer, do not create another service."


def _ev(sid, seq, role, provider=Provider.CLAUDE_CODE, **kw):
    return Event(session_id=sid, provider=provider, seq=seq,
                 ts=f"2026-03-01T10:00:{seq:02d}", role=role, **kw)


def _prompt(sid, seq, text, **kw):
    return _ev(sid, seq, Role.USER, text=text, **kw)


def _edit(sid, seq):
    return _ev(sid, seq, Role.TOOL_CALL, tool=ToolInfo(kind=ToolKind.EDIT, target_path="a.py"))


def _kinds(out):
    return {p["kind"]: p for p in out["patterns"]}


def test_empty_is_unavailable():
    out = prompt_patterns([])
    assert not out["measures"][0].available and out["patterns"] == []


def test_instruction_in_three_sessions_is_a_pattern_with_the_right_files():
    ss = [[_prompt("a", 1, f"Add login. {RULE}")],
          [_prompt("b", 1, f"Add logout. {RULE}")],
          [_prompt("c", 1, f"Add reset. {RULE}", provider=Provider.CODEX)]]
    p = _kinds(prompt_patterns(ss))["repeated_instruction"]
    assert p["sessions"] == 3 and p["occurrences"] == 3
    assert p["targets"] == ["AGENTS.md", "CLAUDE.md"]
    assert p["tokens"] > 0
    assert RULE in p["examples"][0]


def test_two_sessions_is_coincidence_not_a_habit():
    ss = [[_prompt(s, 1, RULE)] for s in "ab"]
    assert "repeated_instruction" not in _kinds(prompt_patterns(ss))


def test_a_pasted_rules_block_is_one_pattern_not_one_per_sentence():
    block = f"{RULE}\nAlways run the tests before saying done.\nNever edit generated files."
    ss = [[_prompt(s, 1, block)] for s in "abc"]
    rep = [p for p in prompt_patterns(ss)["patterns"] if p["kind"] == "repeated_instruction"]
    assert len(rep) == 1


def test_one_off_requests_are_not_standing_rules():
    ss = [[_prompt(s, 1, "Add a dark mode toggle to the settings page.")] for s in "abc"]
    assert prompt_patterns(ss)["patterns"] == []


def test_harness_messages_are_ignored():
    ss = [[_prompt(s, 1, f"<system-reminder>{RULE}</system-reminder>")] for s in "abc"]
    out = prompt_patterns(ss)
    assert out["patterns"] == [] and not out["measures"][0].available


def test_corrections_only_count_after_the_agent_acted():
    evs = [_prompt("s", 0, "no, start with the API")]  # nothing to correct yet
    seq = 1
    for _ in range(3):
        evs += [_edit("s", seq), _prompt("s", seq + 1, "No, reuse the existing API instead.")]
        seq += 2
    p = _kinds(prompt_patterns([evs]))["repeated_correction"]
    assert p["occurrences"] == 3 and p["sessions"] == 1
    m = {x.label: x for x in prompt_patterns([evs])["measures"]}
    assert m["Correction rate"].value == 75.0


def test_serializes():
    out = prompt_patterns([[_prompt(s, 1, RULE)] for s in "abc"])
    json.dumps({**out, "measures": [m.to_dict() for m in out["measures"]]})
