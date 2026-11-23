"""Junie CLI per-session format (sessions/<id>/events.jsonl + summary.json).
Synthetic data shaped like Junie CLI 26.11 output — no real session content."""
import json

from agentlens.adapters import JunieAdapter
from agentlens.schema import Role


def _session(root, sid="session-1"):
    d = root / "sessions" / sid
    d.mkdir(parents=True)
    (d / "summary.json").write_text(json.dumps({
        "sessionId": sid, "createdAt": 1795342309565, "projectDir": "/home/dev/app",
        "taskName": "Fix add() so tests pass", "lifecycle": "FAILED"}))
    blob = json.dumps({"lastAgentState": {"issue": {"description": "Fix the bug in calc.py"}}})
    (d / "state.json").write_text(json.dumps({"kind": "SessionA2uxEvent", "event": {
        "agentEvent": {"kind": "AgentStateUpdatedEvent", "blob": blob}}}))
    agent = {"kind": "MainAgent", "id": "main"}
    lines = [
        {"kind": "TaskStartedEvent", "taskId": "t1", "timestampMs": 1795342309627},
        {"kind": "SessionA2uxEvent", "timestampMs": 1795342389717, "event": {"agentEvent": {
            "kind": "ContextWindowReportEvent", "agent": agent, "used": 2050, "size": 40960}}},
        {"kind": "SessionA2uxEvent", "timestampMs": 1795342390000, "event": {"agentEvent": {
            "kind": "LlmResponseMetadataEvent", "agent": agent, "modelUsage": [{
                "model": "claude-sonnet-5-5", "cost": 0.0, "inputTokens": 1000,
                "cacheInputTokens": 800, "cacheCreateTokens": 50, "outputTokens": 120}]}}},
        {"kind": "SessionA2uxEvent", "timestampMs": 1795342400000, "event": {"agentEvent": {
            "kind": "AgentFailureEvent", "agent": agent, "message": "Failed to pass review"}}},
        "not json",
    ]
    (d / "events.jsonl").write_text("\n".join(
        x if isinstance(x, str) else json.dumps(x) for x in lines) + "\n")


def test_junie_cli_session_is_discovered_and_parsed(tmp_path, monkeypatch):
    _session(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "local.json").write_text('{"id": "x"}')  # must not be a source
    monkeypatch.setenv("JUNIE_HOME", str(tmp_path))
    a = JunieAdapter()
    sources = a.discover()
    assert [s.path.name for s in sources] == ["events.jsonl"]
    evs = list(a.parse(sources[0]))
    user = [e for e in evs if e.role is Role.USER]
    asst = [e for e in evs if e.role is Role.ASSISTANT]
    assert user[0].text == "Fix the bug in calc.py" and user[0].ts.startswith("2026-11-")
    assert len(asst) == 1
    u = asst[0].usage
    assert (u.input, u.cache_read, u.cache_write, u.output) == (200, 800, 50, 120)
    assert asst[0].cost_usd > 0 and asst[0].context_limit == 40960
    assert asst[0].repo_label and evs[0].session_id == "session-1"
    assert any(e.text and e.text.startswith("Junie failed") for e in evs)
    assert [e.seq for e in evs] == sorted({e.seq for e in evs})  # unique, ordered


def test_empty_or_broken_session_files_do_not_crash(tmp_path, monkeypatch):
    d = tmp_path / "sessions" / "s2"
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text("garbage\n")
    (d / "state.json").write_text("{not json")
    monkeypatch.setenv("JUNIE_HOME", str(tmp_path))
    a = JunieAdapter()
    assert list(a.parse(a.discover()[0])) == []
