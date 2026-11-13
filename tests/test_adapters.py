"""Golden tests for adapters.

Agent log formats drift — Codex's especially. These tests are the tripwire: when
a vendor changes their shape, CI tells us, not a GitHub issue three weeks later.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import (  # noqa: E402
    ClaudeCodeAdapter,
    CodexAdapter,
    CopilotCliAdapter,
    GeminiCliAdapter,
    GooseAdapter,
    JunieAdapter,
    OpenCodeAdapter,
)
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.schema import Role, ToolKind  # noqa: E402
from generate import (  # noqa: E402
    build,
    build_copilot_sqlite,
    build_gemini,
    build_goose_jsonl,
    build_goose_sqlite,
    build_opencode,
)


@pytest.fixture(scope="module")
def claude_files(tmp_path_factory) -> list[Path]:
    return build(tmp_path_factory.mktemp("claude") / "projects")


def parse_all(adapter, paths):
    out = []
    for p in paths:
        out.extend(adapter.parse(SourceFile.of(p)))
    return out


def test_claude_code_parses_every_fixture(claude_files):
    events = parse_all(ClaudeCodeAdapter(), claude_files)
    assert len(events) > 80, "fixtures should produce a substantial event stream"
    assert {e.provider.value for e in events} == {"claude-code"}


def test_claude_code_separates_prompts_from_tool_results(claude_files):
    """The single most important correctness property.

    Claude Code carries tool results on `type: "user"` lines. Counting those as
    human prompts silently destroys every behavioural metric in the product.
    """
    events = parse_all(ClaudeCodeAdapter(), claude_files)
    prompts = [e for e in events if e.role is Role.USER]
    results = [e for e in events if e.role is Role.TOOL_RESULT]
    assert results, "tool results must be recognised"
    assert prompts, "real prompts must be recognised"
    for p in prompts:
        assert p.text and "tool_use_id" not in p.text


def test_claude_code_token_accounting(claude_files):
    events = parse_all(ClaudeCodeAdapter(), claude_files)
    assistants = [e for e in events if e.role is Role.ASSISTANT and e.usage]
    assert assistants
    for e in assistants:
        assert e.usage.total > 0
        assert e.cost_usd > 0, f"priced model {e.model} must yield a cost"
        # Context occupancy must never exceed the window; a >100% reading is
        # the classic symptom of double-counting cache reads as fresh input.
        if e.context_tokens and e.context_limit:
            assert e.context_tokens <= e.context_limit * 1.05


def test_event_ids_are_stable_and_unique(claude_files):
    a = parse_all(ClaudeCodeAdapter(), claude_files)
    b = parse_all(ClaudeCodeAdapter(), claude_files)
    assert [e.event_id for e in a] == [e.event_id for e in b], "ids must be deterministic"
    assert len({e.event_id for e in a}) == len(a), "ids must be unique"


def test_tool_normalization_reclassifies_shell():
    from agentlens.schema import normalize_tool

    assert normalize_tool("Bash", "pytest -q") is ToolKind.TEST
    assert normalize_tool("Bash", "git status") is ToolKind.GIT
    assert normalize_tool("Bash", "ls -la") is ToolKind.BASH
    assert normalize_tool("Read") is ToolKind.READ
    assert normalize_tool("apply_patch") is ToolKind.EDIT
    assert normalize_tool("totally_unknown_tool") is ToolKind.OTHER


def test_codex_treats_cumulative_counters_as_deltas(tmp_path):
    """Codex reports running totals. Summing them naively inflates cost by
    roughly the square of the turn count."""
    p = tmp_path / "sessions" / "rollout.jsonl"
    p.parent.mkdir(parents=True)
    lines = [
        {"type": "session_meta", "payload": {"cwd": "/x", "model": "gpt-5.1-codex"}},
        {"type": "event_msg", "payload": {"type": "token_count",
         "info": {"total_token_usage": {"input_tokens": 1000, "output_tokens": 100}}}},
        {"type": "event_msg", "payload": {"type": "token_count",
         "info": {"total_token_usage": {"input_tokens": 2500, "output_tokens": 260}}}},
        {"type": "event_msg", "payload": {"type": "token_count",
         "info": {"total_token_usage": {"input_tokens": 4000, "output_tokens": 400}}}},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines))

    events = list(CodexAdapter().parse(SourceFile.of(p)))
    usages = [e.usage for e in events if e.usage]
    assert len(usages) == 3
    assert [u.input for u in usages] == [1000, 1500, 1500]
    assert sum(u.input for u in usages) == 4000, "must equal the final cumulative total"


def test_codex_input_tokens_include_cached_so_cached_is_not_billed_twice(tmp_path):
    p = tmp_path / "rollout.jsonl"
    p.write_text(json.dumps({"type": "event_msg", "payload": {"type": "token_count",
        "info": {"total_token_usage": {"input_tokens": 1000, "cached_input_tokens": 800,
                                       "output_tokens": 50}}}}))
    (u,) = [e.usage for e in CodexAdapter().parse(SourceFile.of(p)) if e.usage]
    assert (u.input, u.cache_read, u.output) == (200, 800, 50)


def test_junie_finds_sessions_regardless_of_key_name(tmp_path):
    """Junie has moved this key between IDE releases; we locate it by shape."""
    for key in ("sessions", "history", "conversations"):
        p = tmp_path / key / "history.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({key: [{
            "id": "s1", "projectPath": "/proj", "title": "Task",
            "steps": [
                {"role": "user", "content": "do the thing", "timestamp": "2026-08-01T10:00:00"},
                {"role": "assistant", "content": "done", "model": "claude-sonnet-4-5",
                 "usage": {"input": 900, "output": 120}, "timestamp": "2026-08-01T10:00:09"},
            ],
        }]}))
        events = list(JunieAdapter().parse(SourceFile.of(p)))
        roles = {e.role for e in events}
        assert Role.USER in roles and Role.ASSISTANT in roles, f"failed for key {key!r}"
        assert any(e.cost_usd > 0 for e in events)


def test_repo_paths_are_hashed_not_stored(claude_files):
    events = parse_all(ClaudeCodeAdapter(), claude_files)
    for e in events:
        if e.repo_id:
            assert "/" not in e.repo_id and len(e.repo_id) == 16


def test_adapters_survive_corrupt_input(tmp_path):
    p = tmp_path / "broken.jsonl"
    p.write_text('{"type":"user"}\nNOT JSON AT ALL\n{"type":"assistant"\n')
    # Must not raise — a half-written session file is the normal case when the
    # agent is still running.
    assert isinstance(list(ClaudeCodeAdapter().parse(SourceFile.of(p))), list)


# --- Phase 2 adapters -------------------------------------------------------


def test_opencode_parses_one_file_per_message(tmp_path):
    paths = build_opencode(tmp_path / "storage")
    adapter = OpenCodeAdapter()
    events = []
    for p in paths:
        events.extend(adapter.parse(SourceFile.of(p)))
    assert len(events) >= 4  # 1 user + 2 assistant + 1 tool_call
    assert {e.session_id for e in events} == {"oc-session-1"}
    assert any(e.role is Role.USER for e in events)
    assert any(e.role is Role.ASSISTANT and e.usage and e.usage.total > 0 for e in events)
    assert any(e.role is Role.TOOL_CALL for e in events)


def test_opencode_seq_is_stable_across_separate_ingest_runs(tmp_path):
    """One session spans many files (one per message), unlike every other
    adapter here. A per-parse-call counter starting at 0 would collide the
    moment a new message arrives after AgentLens has already ingested and
    exited once — this is the bug `seq_from_ts` exists to prevent."""
    storage = tmp_path / "storage"
    paths = build_opencode(storage)
    adapter = OpenCodeAdapter()

    # Simulate run 1: ingest only the messages that exist right now.
    run1 = [ev for p in paths for ev in adapter.parse(SourceFile.of(p))]
    max_seq_run1 = max(e.seq for e in run1)

    # Simulate run 2, a fresh process, seeing one new message file.
    import json as _json

    new_msg = storage / "message" / "oc-session-1" / "msg_99.json"
    new_msg.write_text(
        _json.dumps(
            {
                "id": "msg_99", "sessionID": "oc-session-1", "role": "assistant",
                "time": "2026-08-10T09:05:00Z", "cwd": "/home/dev/svc-handler",
                "content": [{"type": "text", "text": "Later message."}],
                "model": {"name": "claude-sonnet-4-5"},
                "tokens": {"input": 1000, "output": 50}, "cost": 0,
            }
        )
    )
    fresh_adapter = OpenCodeAdapter()  # new instance == new process, no shared state
    run2_new_event = list(fresh_adapter.parse(SourceFile.of(new_msg)))[0]

    assert run2_new_event.seq > max_seq_run1, (
        "a message ingested in a later run must sort after everything from an "
        "earlier run, or session event order silently corrupts on incremental ingest"
    )


def test_gemini_cli_parses_array_of_turns(tmp_path):
    paths = build_gemini(tmp_path)
    events = list(GeminiCliAdapter().parse(SourceFile.of(paths[0])))
    assert any(e.role is Role.USER for e in events)
    assistants = [e for e in events if e.role is Role.ASSISTANT]
    assert assistants and assistants[0].usage.total > 0
    assert assistants[0].cost_usd > 0


def test_gemini_cli_marked_experimental():
    assert GeminiCliAdapter().schema_version.endswith("-experimental")


def test_gemini_cli_skips_records_with_no_recognized_role(tmp_path):
    """Real-usage regression, 2026-08-16: `~/.gemini` isn't exclusive to
    Gemini CLI — Google Antigravity also stores data there (tool-schema
    manifests, artifact/task metadata), none of which carry a `role` field.
    A real user's history had files like these fabricated into fake "Gemini
    CLI sessions" (`Role.USER`, text pulled from whatever content/text/parts
    field happened to exist) even though none of it was ever a conversation
    turn or ever carried usage data. The fix must skip any record with no
    recognized role outright, regardless of what other fields it happens to
    carry, rather than guessing a role whenever text is present."""
    import json as _json

    chat_dir = tmp_path / "tmp" / "proj-hash-1" / "chats"
    chat_dir.mkdir(parents=True, exist_ok=True)
    p = chat_dir / "not_a_chat.json"
    p.write_text(_json.dumps([
        # A tool-schema manifest — no role field, no text-shaped field either.
        {"name": "cross_repo_search_tool", "description": "Search across repos",
         "parameters": {"type": "object"}},
        # No role field, but DOES have a field the parser actually reads
        # (`content`) — this is the shape that would previously have been
        # fabricated into a fake Role.USER "prompt".
        {"artifactType": "task", "content": "Task finished with result: done",
         "updatedAt": "2026-08-11T14:00:00Z"},
    ]))
    events = list(GeminiCliAdapter().parse(SourceFile.of(p)))
    assert events == [], (
        "records with no recognized role must be skipped entirely, not "
        "turned into a fabricated Role.USER session"
    )


def test_gemini_cli_still_parses_real_conversation_turns(tmp_path):
    """The fix must not be so aggressive it breaks genuine Gemini CLI logs —
    same fixture shape as `test_gemini_cli_parses_array_of_turns`, just
    guarding the negative-case fix above didn't overcorrect."""
    paths = build_gemini(tmp_path)
    events = list(GeminiCliAdapter().parse(SourceFile.of(paths[0])))
    assert any(e.role is Role.USER for e in events)
    assert any(e.role is Role.ASSISTANT for e in events)


def test_copilot_cli_introspects_unknown_sqlite_schema(tmp_path):
    """No public schema exists for session-store.db, so this adapter has to
    find its own columns. The test's synthetic db uses different column names
    than a naive hardcoded query would guess, on purpose."""
    db = build_copilot_sqlite(tmp_path)
    events = list(CopilotCliAdapter().parse(SourceFile.of(db)))
    assert events, "introspection must find the messages table without being told its name"
    assert any(e.role is Role.USER for e in events)
    assistants = [e for e in events if e.role is Role.ASSISTANT]
    assert assistants
    assert assistants[0].usage is not None and assistants[0].usage.total > 0
    assert assistants[0].model == "gpt-5.1-codex"


def test_copilot_cli_marked_experimental():
    assert CopilotCliAdapter().schema_version.endswith("-experimental")


def test_goose_sqlite_and_legacy_jsonl_both_parse(tmp_path):
    sqlite_events = list(GooseAdapter().parse(SourceFile.of(build_goose_sqlite(tmp_path))))
    jsonl_events = list(GooseAdapter().parse(SourceFile.of(build_goose_jsonl(tmp_path))))

    assert sqlite_events, "SQLite (Goose >= 1.10) branch must extract something"
    assert any(e.role is Role.USER for e in sqlite_events)

    assert jsonl_events, "legacy JSONL (Goose < 1.10) branch must still work"
    assert any(e.role is Role.ASSISTANT and e.usage for e in jsonl_events)


def test_goose_marked_experimental():
    assert GooseAdapter().schema_version.endswith("-experimental")


def test_sqlite_adapters_degrade_gracefully_on_garbage_input(tmp_path):
    """A locked, mid-write, or simply-not-SQLite file must not crash ingest."""
    not_a_db = tmp_path / "session-store.db"
    not_a_db.write_bytes(b"this is not a sqlite file")
    assert list(CopilotCliAdapter().parse(SourceFile.of(not_a_db))) == []

    not_a_db2 = tmp_path / "sessions.db"
    not_a_db2.write_bytes(b"also not sqlite")
    assert list(GooseAdapter().parse(SourceFile.of(not_a_db2))) == []


def test_all_seven_adapters_registered():
    from agentlens.adapters import ADAPTERS

    assert len(ADAPTERS) == 7
    assert len({a.provider for a in ADAPTERS}) == 7  # no duplicate provider


def _codex_rollout() -> list[dict]:
    """Current Codex shape: everything wrapped in response_item / event_msg."""
    ri = lambda p: {"timestamp": "2026-10-01T10:00:00Z", "type": "response_item", "payload": p}  # noqa: E731
    return [
        {"timestamp": "2026-10-01T10:00:00Z", "type": "session_meta",
         "payload": {"id": "sess-1", "cwd": "/x", "model": "gpt-5.1-codex"}},
        ri({"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "<environment_context>cwd</environment_context>"}]}),
        ri({"type": "message", "role": "user", "content": [{"type": "input_text",
            "text": "# AGENTS.md instructions for /x\n\n<INSTRUCTIONS>\nbe terse\n</INSTRUCTIONS>"}]}),
        ri({"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "fix the failing test"}]}),
        {"timestamp": "2026-10-01T10:00:01Z", "type": "event_msg",
         "payload": {"type": "user_message", "message": "fix the failing test"}},
        ri({"type": "custom_tool_call", "name": "apply_patch", "call_id": "c1",
            "input": "*** Begin Patch\n*** Update File: src/a.py\n@@\n-x\n+y\n*** End Patch"}),
        ri({"type": "custom_tool_call_output", "call_id": "c1",
            "output": json.dumps({"output": "ok", "metadata": {"exit_code": 0}})}),
        ri({"type": "function_call", "name": "exec_command", "call_id": "c2",
            "arguments": json.dumps({"cmd": "pytest -q", "workdir": "/x"})}),
        ri({"type": "function_call_output", "call_id": "c2",
            "output": "Exit code: 1\nWall time: 1s\nOutput:\nFAILED test_a"}),
        {"timestamp": "2026-10-01T10:00:02Z", "type": "token_usage_record",
         "payload": {"usage": {"input_tokens": 999999, "output_tokens": 999999}}},
    ]


@pytest.mark.parametrize("compressed", [False, True])
def test_codex_current_rollout_format(tmp_path, compressed):
    lines = "\n".join(json.dumps(r) for r in _codex_rollout())
    if compressed:
        zstd = pytest.importorskip("compression.zstd")
        p = tmp_path / "sessions" / "rollout-x.jsonl.zst"
        p.parent.mkdir()
        p.write_bytes(zstd.compress(lines.encode()))
    else:
        p = tmp_path / "sessions" / "rollout-x.jsonl"
        p.parent.mkdir()
        p.write_text(lines)

    a = CodexAdapter()
    a.roots = lambda: [tmp_path / "sessions"]
    assert [s.path for s in a.discover()] == [p]
    events = list(a.parse(SourceFile.of(p)))

    assert {e.session_id for e in events} == {"sess-1"}
    users = [e.text for e in events if e.role is Role.USER]
    assert users == ["fix the failing test"]  # injected context + event_msg dup skipped
    calls = [e.tool for e in events if e.role is Role.TOOL_CALL]
    assert [(t.kind, t.target_path) for t in calls] == [
        (ToolKind.EDIT, "src/a.py"), (ToolKind.TEST, None)]
    results = [e.tool for e in events if e.role is Role.TOOL_RESULT]
    assert [(t.kind, t.exit_code) for t in results] == [
        (ToolKind.EDIT, 0), (ToolKind.TEST, 1)]
    assert not any(e.usage for e in events)  # token_usage_record not double-billed


def test_harness_messages_are_not_prompts(tmp_path):
    """Compaction summaries and agent-injected instructions are not prompts
    the person wrote; counting them skews prompt mining and prompts/task."""
    p = tmp_path / "s.jsonl"
    base = {"sessionId": "s", "timestamp": "2026-10-01T10:00:00Z", "cwd": "/x"}
    p.write_text("\n".join(json.dumps({**base, **r}) for r in [
        {"type": "user", "isCompactSummary": True,
         "message": {"role": "user", "content": "This session is being continued..."}},
        {"type": "user", "message": {"role": "user", "content": "real prompt"}},
    ]))
    users = [e.text for e in ClaudeCodeAdapter().parse(SourceFile.of(p)) if e.role is Role.USER]
    assert users == ["real prompt"]

    rollout = _codex_rollout()
    rollout.insert(1, {"timestamp": "2026-10-01T10:00:00Z", "type": "response_item", "payload": {
        "type": "message", "role": "user", "content": [{"type": "input_text",
        "text": "The following is the Codex agent history added since your last approval"}]}})
    q = tmp_path / "rollout-y.jsonl"
    q.write_text("\n".join(json.dumps(r) for r in rollout))
    users = [e.text for e in CodexAdapter().parse(SourceFile.of(q)) if e.role is Role.USER]
    assert users == ["fix the failing test"]
