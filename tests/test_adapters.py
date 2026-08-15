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
