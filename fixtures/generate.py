#!/usr/bin/env python3
"""Synthetic fixture generator.

**No real session data is ever committed to this repository.** Agent history
contains repo paths, hostnames, client names and occasionally credentials. The
fixtures here are generated, deterministic, and deliberately contain each waste
pattern the detectors look for, so the golden tests actually test something.

    python fixtures/generate.py            # write fixtures/
    python fixtures/generate.py --demo DIR # a full fake ~/.claude to point at
"""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODELS = ["claude-opus-4-5-20251101", "claude-sonnet-4-5-20250929", "claude-haiku-4-5-20251001"]


class Writer:
    """Emits Claude Code-shaped JSONL."""

    def __init__(self, session_id: str, cwd: str, start: datetime):
        self.sid = session_id
        self.cwd = cwd
        self.t = start
        self.lines: list[dict] = []
        self.prev = None
        self.cache = 0

    def _tick(self, seconds: int = 7) -> str:
        self.t += timedelta(seconds=seconds)
        return self.t.isoformat() + "Z"

    def _base(self, typ: str) -> dict:
        u = str(uuid.uuid4())
        rec = {
            "type": typ, "uuid": u, "parentUuid": self.prev, "sessionId": self.sid,
            "timestamp": self._tick(), "cwd": self.cwd, "gitBranch": "main",
            "version": "2.0.0",
        }
        self.prev = u
        return rec

    def user(self, text: str) -> None:
        r = self._base("user")
        r["message"] = {"role": "user", "content": [{"type": "text", "text": text}]}
        self.lines.append(r)

    def assistant(self, text: str, model: str, inp: int, out: int, tools=None) -> list[str]:
        """`inp` is the TOTAL context size for this turn, not fresh input.

        Real agents cache: most of the window is billed as a cheap cache read
        and only the delta is fresh input. Modelling it any other way produces
        the nonsense "757% context full" readings that make a dashboard
        untrustworthy.
        """
        r = self._base("assistant")
        content: list[dict] = []
        if text:
            content.append({"type": "text", "text": text})
        ids: list[str] = []
        for name, tin in tools or []:
            tid = "toolu_" + uuid.uuid4().hex[:16]
            ids.append(tid)
            content.append({"type": "tool_use", "id": tid, "name": name, "input": tin})
        total = max(inp, self.cache)
        cache_read = min(self.cache, total)
        fresh = max(0, total - cache_read)
        cache_write = fresh if cache_read == 0 else 0
        self.cache = total + out
        r["message"] = {
            "role": "assistant", "model": model, "content": content,
            "usage": {
                "input_tokens": fresh, "output_tokens": out,
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
            },
        }
        self.lines.append(r)
        return ids

    def tool_fail(self, tid: str, error: str) -> None:
        """A failing tool result — this is what the loop detector reads."""
        self.tool_result(tid, error, is_error=True)

    def tool_result(self, tid: str, content: str, is_error: bool = False) -> None:
        r = self._base("user")
        r["message"] = {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tid, "content": content,
                 "is_error": is_error}
            ],
        }
        self.lines.append(r)

    def compaction(self) -> None:
        r = self._base("system")
        r["subtype"] = "compact_boundary"
        r["isCompactSummary"] = True
        r["content"] = "This conversation was compacted to free context."
        self.lines.append(r)

    def summary(self, text: str) -> None:
        r = self._base("summary")
        r["summary"] = text
        self.lines.append(r)

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for line in self.lines:
                fh.write(json.dumps(line) + "\n")


def session_blind_retry(start: datetime) -> Writer:
    """Three near-identical prompts with no new evidence between them."""
    w = Writer("11111111-1111-4111-8111-111111111111", "/home/dev/checkout-service", start)
    w.summary("Fix the failing checkout webhook")
    w.user("the webhook handler is broken, can you fix it")
    ids = w.assistant("Let me look.", MODELS[0], 4200, 380,
                      [("Grep", {"pattern": "webhook"}), ("Read", {"file_path": "/home/dev/checkout-service/src/webhook.py"})])
    w.tool_result(ids[0], "src/webhook.py:12\nsrc/routes.py:88")
    w.tool_result(ids[1], "def handle(req):\n" + "    ...\n" * 60)
    w.assistant("I changed the signature check.", MODELS[0], 9800, 620,
                [("Edit", {"file_path": "/home/dev/checkout-service/src/webhook.py"})])

    w.user("still not working, please fix it")
    w.assistant("Let me try a different approach.", MODELS[0], 15200, 700,
                [("Edit", {"file_path": "/home/dev/checkout-service/src/webhook.py"})])

    w.user("that didn't work either, it's still broken")
    w.assistant("Trying once more.", MODELS[0], 21400, 810,
                [("Edit", {"file_path": "/home/dev/checkout-service/src/webhook.py"})])

    # The turn that finally works — because it carries evidence.
    w.user("I ran `pytest tests/test_webhook.py` and got: "
           "AssertionError: expected 200 got 401 at test_webhook.py line 42. "
           "Expected the handler to accept a valid HMAC signature.")
    w.assistant("That's the header casing. Fixed.", MODELS[0], 24000, 400,
                [("Edit", {"file_path": "/home/dev/checkout-service/src/webhook.py"})])
    return w


def session_output_flood(start: datetime) -> Writer:
    """One unpiped command dumps 120 KB into a long session."""
    w = Writer("22222222-2222-4222-8222-222222222222", "/home/dev/web-app", start)
    w.summary("Audit dependency tree for duplicates")
    w.user("check our dependency tree in @package.json for duplicate versions")
    ids = w.assistant("Checking.", MODELS[1], 3000, 200, [("Bash", {"command": "npm ls --all"})])
    w.tool_result(ids[0], "node_modules tree\n" + ("├── some-package@1.2.3\n" * 4000))
    for i in range(6):
        w.assistant(f"Analysing branch {i}.", MODELS[1], 32000 + i * 1500, 400)
    return w


def session_vague_and_reread(start: datetime) -> Writer:
    w = Writer("33333333-3333-4333-8333-333333333333", "/home/dev/api-gateway", start)
    w.summary("Clean up the auth module")
    w.user("clean up the auth stuff")
    ids = w.assistant("Looking around.", MODELS[0], 2800, 300, [
        ("Glob", {"pattern": "**/*auth*"}),
        ("Grep", {"pattern": "authenticate"}),
        ("Read", {"file_path": "/home/dev/api-gateway/src/auth.py"}),
        ("Read", {"file_path": "/home/dev/api-gateway/src/middleware.py"}),
    ])
    w.tool_result(ids[0], "src/auth.py\nsrc/middleware.py")
    w.tool_result(ids[1], "src/auth.py:20\nsrc/auth.py:55")
    w.tool_result(ids[2], "x = 1\n" * 900)
    w.tool_result(ids[3], "y = 2\n" * 600)

    for _ in range(3):
        i2 = w.assistant("Re-checking the file.", MODELS[0], 12000, 500,
                         [("Read", {"file_path": "/home/dev/api-gateway/src/auth.py"})])
        w.tool_result(i2[0], "x = 1\n" * 900)

    for _ in range(9):
        w.assistant("Editing.", MODELS[0], 14000, 400,
                    [("Edit", {"file_path": "/home/dev/api-gateway/src/auth.py"})])
    return w


def session_compaction(start: datetime) -> Writer:
    w = Writer("44444444-4444-4444-8444-444444444444", "/home/dev/data-pipeline", start)
    w.summary("Migrate the ETL pipeline to the new scheduler")
    w.user("migrate @etl/pipeline.py to the new scheduler; `pytest etl/` must pass")
    for i in range(14):
        w.assistant(f"Step {i}.", MODELS[0], 40000 + i * 9000, 900,
                    [("Edit", {"file_path": "/home/dev/data-pipeline/etl/pipeline.py"})])
    w.compaction()
    for i in range(4):
        w.assistant(f"Continuing {i}.", MODELS[0], 30000, 700)
    return w


def session_clean(start: datetime) -> Writer:
    """A well-run session. Proves the detectors stay quiet when they should."""
    w = Writer("55555555-5555-4555-8555-555555555555", "/home/dev/billing", start)
    w.summary("Add proration to subscription upgrades")
    w.user("In @src/billing/proration.py add a prorate() that splits a mid-cycle "
           "upgrade by remaining days; `pytest tests/test_proration.py -q` must pass.")
    ids = w.assistant("On it.", MODELS[1], 3400, 900, [
        ("Read", {"file_path": "/home/dev/billing/src/billing/proration.py"}),
        ("Edit", {"file_path": "/home/dev/billing/src/billing/proration.py"}),
        ("Bash", {"command": "pytest tests/test_proration.py -q"}),
    ])
    w.tool_result(ids[0], "def existing(): pass\n")
    w.tool_result(ids[1], "ok")
    w.tool_result(ids[2], "3 passed in 0.4s")
    w.assistant("Done — 3 tests pass.", MODELS[1], 5200, 220)
    return w


def session_loop(start: datetime) -> Writer:
    """The model failing the same way five times and never fixing it.

    Note the prompts here are *well written* — they name the file and the
    acceptance check. That's deliberate: a loop is not a prompting failure, and
    this fixture proves the detector fires on model behaviour rather than on
    sloppy input.
    """
    w = Writer("66666666-6666-4666-8666-666666666666", "/home/dev/payments-api", start)
    w.summary("Fix failing signature verification")
    w.user(
        "In @src/payments/verify.py make verify_signature() accept the v2 header format; "
        "`pytest tests/test_verify.py -q` must pass."
    )
    # Same underlying failure, five times, with volatile bits differing each
    # round (line numbers, durations) so the signature normaliser is exercised.
    for i in range(5):
        ids = w.assistant(
            f"Attempt {i + 1}: adjusting the header parser.",
            MODELS[0], 18000 + i * 6000, 500,
            [
                ("Edit", {"file_path": "/home/dev/payments-api/src/payments/verify.py"}),
                ("Bash", {"command": "pytest tests/test_verify.py -q"}),
            ],
        )
        w.tool_result(ids[0], "ok")
        w.tool_fail(
            ids[1],
            "Traceback (most recent call last):\n"
            f'  File "/home/dev/payments-api/tests/test_verify.py", line {40 + i * 3}, '
            "in test_v2_header\n"
            "    assert verify_signature(req) is True\n"
            f"AssertionError: expected True, got False (ran in 0.{i}3s)",
        )
    return w


def session_converged_iteration(start: datetime) -> Writer:
    """Healthy iteration: the same failure three times, then it passes.

    This is the control for the loop detector. Normal debugging looks almost
    exactly like a loop right up until it works, and a detector that can't tell
    them apart would fire on every productive session and be worthless.
    """
    w = Writer("77777777-7777-4777-8777-777777777777", "/home/dev/search-svc", start)
    w.summary("Fix off-by-one in pagination")
    w.user(
        "In @src/search/paginate.py fix the off-by-one on the last page; "
        "`pytest tests/test_paginate.py -q` must pass."
    )
    for i in range(3):
        ids = w.assistant(
            f"Attempt {i + 1}.", MODELS[1], 6000 + i * 1500, 300,
            [
                ("Edit", {"file_path": "/home/dev/search-svc/src/search/paginate.py"}),
                ("Bash", {"command": "pytest tests/test_paginate.py -q"}),
            ],
        )
        w.tool_result(ids[0], "ok")
        w.tool_fail(
            ids[1],
            f"  File \"/home/dev/search-svc/tests/test_paginate.py\", line {12 + i}, "
            "in test_last_page\n"
            "AssertionError: expected 10 items, got 9",
        )
    ids = w.assistant(
        "Found it — the slice bound was exclusive.", MODELS[1], 12000, 400,
        [
            ("Edit", {"file_path": "/home/dev/search-svc/src/search/paginate.py"}),
            ("Bash", {"command": "pytest tests/test_paginate.py -q"}),
        ],
    )
    w.tool_result(ids[0], "ok")
    w.tool_result(ids[1], "4 passed in 0.3s")  # converged — detector must stay quiet
    return w


def session_recurring_next_month(start: datetime) -> Writer:
    """The same failure as `session_loop`, in a different session.

    This is what makes `CrossSessionLoopDetector` fire: the knowledge from the
    first session didn't persist anywhere, so it was rediscovered (and paid
    for) again.
    """
    w = Writer("88888888-8888-4888-8888-888888888888", "/home/dev/payments-api", start)
    w.summary("Signature verification failing again")
    w.user(
        "In @src/payments/verify.py the v2 header check regressed; "
        "`pytest tests/test_verify.py -q` must pass."
    )
    for i in range(3):
        ids = w.assistant(
            f"Attempt {i + 1}.", MODELS[0], 15000 + i * 4000, 450,
            [
                ("Edit", {"file_path": "/home/dev/payments-api/src/payments/verify.py"}),
                ("Bash", {"command": "pytest tests/test_verify.py -q"}),
            ],
        )
        w.tool_result(ids[0], "ok")
        w.tool_fail(
            ids[1],
            "Traceback (most recent call last):\n"
            f'  File "/home/dev/payments-api/tests/test_verify.py", line {51 + i}, '
            "in test_v2_header\n"
            "    assert verify_signature(req) is True\n"
            f"AssertionError: expected True, got False (ran in 1.{i}1s)",
        )
    return w


BUILDERS = [
    session_blind_retry,
    session_output_flood,
    session_vague_and_reread,
    session_compaction,
    session_clean,
    session_loop,
    session_converged_iteration,
    session_recurring_next_month,
]


def build(target: Path) -> list[Path]:
    random.seed(7)
    base = datetime(2026, 8, 4, 9, 30, 0)
    written = []
    for i, fn in enumerate(BUILDERS):
        w = fn(base + timedelta(days=i, hours=i))
        # Claude Code encodes the cwd into the directory name.
        enc = w.cwd.replace("/", "-")
        path = target / enc / f"{w.sid}.jsonl"
        w.write(path)
        written.append(path)
    return written


def build_opencode(storage_root: Path) -> list[Path]:
    """One JSON file per message under storage/message/<sessionID>/msg_*.json —
    exercises the "one session, many files" ingest path and its timestamp-based
    sequencing (see `Adapter.seq_from_ts`). `storage_root` is the OpenCode
    ``storage/`` directory itself, e.g. ``$OPENCODE_HOME/storage``."""
    sid = "oc-session-1"
    msg_dir = storage_root / "message" / sid
    msg_dir.mkdir(parents=True, exist_ok=True)
    base = datetime(2026, 8, 10, 9, 0, 0)
    written = []

    turns = [
        ("user", None, "In @src/handler.py make parse() reject malformed JSON; "
                        "`pytest tests/test_handler.py -q` must pass.", None),
        ("assistant", "claude-sonnet-4-5", "On it.", [
            {"type": "tool", "tool": "read", "state": {"input": {"filePath": "src/handler.py"}}},
        ]),
        ("assistant", "claude-sonnet-4-5", "Added validation and a test.", None),
    ]
    for i, (role, model, text, parts) in enumerate(turns):
        ts = (base + timedelta(seconds=i * 12)).isoformat() + "Z"
        rec = {
            "id": f"msg_{i}", "sessionID": sid, "role": role, "time": ts,
            "cwd": "/home/dev/svc-handler",
            "content": [{"type": "text", "text": text}],
        }
        if role == "assistant":
            rec["model"] = {"name": model}
            rec["tokens"] = {"input": 4200 + i * 500, "output": 300, "cache": {"read": 0, "write": 0}}
            rec["cost"] = 0  # OpenCode stores cost:0; AgentLens recomputes it
        if parts:
            rec["parts"] = [{"type": "text", "text": text}] + parts
        p = msg_dir / f"msg_{i}.json"
        p.write_text(json.dumps(rec))
        written.append(p)
    return written


def build_gemini(target: Path) -> list[Path]:
    """~/.gemini/tmp/<hash>/chats/<file>.json — array-of-turns shape, one of
    the plausible formats this experimental adapter tolerates."""
    chat_dir = target / "tmp" / "proj-hash-1" / "chats"
    chat_dir.mkdir(parents=True, exist_ok=True)
    base = datetime(2026, 8, 11, 14, 0, 0)
    turns = [
        {"role": "user", "timestamp": base.isoformat() + "Z",
         "parts": [{"text": "explain @src/config.py"}]},
        {"role": "model", "timestamp": (base + timedelta(seconds=8)).isoformat() + "Z",
         "model": "gemini-3-pro", "parts": [{"text": "It loads env vars."}],
         "usageMetadata": {"promptTokenCount": 3000, "candidatesTokenCount": 250,
                            "cachedContentTokenCount": 0}},
    ]
    p = chat_dir / "chat-1.json"
    p.write_text(json.dumps(turns))
    return [p]


def build_copilot_sqlite(target: Path) -> Path:
    """A synthetic session-store.db shaped like GitHub's documented
    description (messages with role/content/tools/timestamp) — schema is our
    best guess, clearly marked experimental in the adapter itself."""
    import sqlite3

    target.mkdir(parents=True, exist_ok=True)
    db = target / "session-store.db"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, "
        "content TEXT, model TEXT, created_at TEXT)"
    )
    base = datetime(2026, 8, 12, 10, 0, 0)
    rows = [
        ("cop-1", "user", "fix the flaky test in @tests/test_retry.py", None,
         base.isoformat() + "Z"),
        ("cop-1", "assistant",
         json.dumps({"text": "Added a retry with backoff.",
                     "usage": {"prompt_tokens": 5200, "completion_tokens": 400},
                     "model": "gpt-5.1-codex"}),
         "gpt-5.1-codex", (base + timedelta(seconds=9)).isoformat() + "Z"),
    ]
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, model, created_at) VALUES (?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def build_goose_sqlite(target: Path) -> Path:
    """A synthetic sessions.db for Goose >= 1.10 — same caveat as Copilot's:
    schema is inferred from Goose's documented description, not confirmed."""
    import sqlite3

    target.mkdir(parents=True, exist_ok=True)
    db = target / "sessions.db"
    if db.exists():
        db.unlink()
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE conversation_messages (id INTEGER PRIMARY KEY, session_id TEXT, "
        "role TEXT, content TEXT, created_at TEXT)"
    )
    base = datetime(2026, 8, 13, 8, 0, 0)
    rows = [
        ("goose-1", "user", "summarize recent changes in @CHANGELOG.md", base.isoformat() + "Z"),
        ("goose-1", "assistant", "Three entries added this week.",
         (base + timedelta(seconds=6)).isoformat() + "Z"),
    ]
    conn.executemany(
        "INSERT INTO conversation_messages (session_id, role, content, created_at) "
        "VALUES (?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def build_goose_jsonl(target: Path) -> Path:
    """Pre-1.10 Goose: one JSONL file per session."""
    target.mkdir(parents=True, exist_ok=True)
    p = target / "goose-legacy-1.jsonl"
    base = datetime(2026, 8, 9, 8, 0, 0)
    lines = [
        {"role": "user", "timestamp": base.isoformat() + "Z", "content": "list open TODOs"},
        {"role": "assistant", "timestamp": (base + timedelta(seconds=5)).isoformat() + "Z",
         "model": "gpt-4.1-mini", "content": "Found 2 TODOs.",
         "usage": {"input_tokens": 1200, "output_tokens": 80}},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines))
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic agent fixtures.")
    ap.add_argument("--demo", metavar="DIR",
                    help="build a fake CLAUDE_CONFIG_DIR you can point AgentLens at")
    args = ap.parse_args()

    if args.demo:
        home = Path(args.demo).expanduser()
        cc_paths = build(home / ".claude" / "projects")
        oc_paths = build_opencode(home / ".opencode" / "storage")
        gm_paths = build_gemini(home / ".gemini")
        build_copilot_sqlite(home / ".copilot")
        build_goose_sqlite(home / ".goose" / "sessions")
        build_goose_jsonl(home / ".goose" / "sessions")
        total = len(cc_paths) + len(oc_paths) + len(gm_paths) + 3
        print(f"Wrote {total} synthetic session file(s) across 6 agents to {home}")
        print(
            "\nTry it:\n"
            f"  export CLAUDE_CONFIG_DIR={home}/.claude\n"
            f"  export OPENCODE_HOME={home}/.opencode\n"
            f"  export GEMINI_HOME={home}/.gemini\n"
            f"  export COPILOT_HOME={home}/.copilot\n"
            f"  export GOOSE_HOME={home}/.goose/sessions\n"
            "  export AGENTLENS_HOME=/tmp/agentlens-demo\n"
            "  agentlens doctor && agentlens ingest && agentlens waste"
        )
    else:
        target = ROOT / "claude-code"
        paths = build(target)
        print(f"Wrote {len(paths)} fixture file(s) to {target}")


if __name__ == "__main__":
    main()
