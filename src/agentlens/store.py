"""Local SQLite store.

Everything lives in ``~/.agentlens/agentlens.db``. Nothing leaves the machine.

Why a database and not just re-parsing on demand: you want month-over-month
history and incremental ingest. Re-parsing gigabytes of JSONL on every page load
is exactly why most local dashboards feel sluggish and get abandoned.

Ingest is idempotent via ``Event.event_id``, so re-running ``ingest`` is always
safe and never double-counts.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

from .schema import Event, Provider, Role, Session, ToolKind, Usage

SCHEMA_VERSION = 2

_DDL = """
CREATE TABLE IF NOT EXISTS events (
    event_id       TEXT PRIMARY KEY,
    session_id     TEXT NOT NULL,
    provider       TEXT NOT NULL,
    seq            INTEGER NOT NULL,
    ts             TEXT,
    role           TEXT NOT NULL,
    model          TEXT,
    tok_input      INTEGER DEFAULT 0,
    tok_output     INTEGER DEFAULT 0,
    tok_cache_read INTEGER DEFAULT 0,
    tok_cache_write INTEGER DEFAULT 0,
    tok_reasoning  INTEGER DEFAULT 0,
    cost_usd       REAL DEFAULT 0,
    text           TEXT,
    tool_kind      TEXT,
    tool_raw_name  TEXT,
    tool_in_bytes  INTEGER DEFAULT 0,
    tool_out_bytes INTEGER DEFAULT 0,
    tool_exit_code INTEGER,
    tool_target    TEXT,
    tool_error_text TEXT,
    context_tokens INTEGER,
    context_limit  INTEGER,
    cwd            TEXT,
    git_branch     TEXT,
    repo_id        TEXT,
    repo_label     TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_session ON events(session_id, seq);
CREATE INDEX IF NOT EXISTS ix_events_provider ON events(provider);
CREATE INDEX IF NOT EXISTS ix_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS ix_events_role ON events(role);
CREATE INDEX IF NOT EXISTS ix_events_repo ON events(repo_id);

CREATE TABLE IF NOT EXISTS sources (
    path           TEXT PRIMARY KEY,
    provider       TEXT NOT NULL,
    size_bytes     INTEGER,
    mtime          REAL,
    schema_version TEXT,
    ingested_at    TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def default_db_path() -> Path:
    env = os.environ.get("AGENTLENS_HOME")
    home = Path(env) if env else Path.home() / ".agentlens"
    home.mkdir(parents=True, exist_ok=True)
    return home / "agentlens.db"


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        #: True when this open triggered a destructive schema upgrade, so the
        #: CLI can tell the user to re-ingest instead of silently showing them
        #: an empty database.
        self.migrated_from: int | None = self._migrate()
        self.conn.executescript(_DDL)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self.conn.commit()

    def _migrate(self) -> int | None:
        """Upgrade an older database by discarding it.

        This is safe precisely because `events` is a **derived cache** — the
        agents' own history files on disk are the source of truth and are never
        written to. Re-parsing is idempotent (`event_id` is a content hash), so
        dropping and repopulating loses nothing except the time it takes.

        Writing real column-by-column migrations here would be ceremony around
        data we can always regenerate, and every migration path would be one
        more thing to get subtly wrong.
        """
        try:
            row = self.conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.Error:
            return None  # brand-new database, nothing to migrate
        if row is None:
            return None
        try:
            found = int(row["value"])
        except (TypeError, ValueError):
            found = 0
        if found >= SCHEMA_VERSION:
            return None
        self.conn.executescript("DROP TABLE IF EXISTS events; DROP TABLE IF EXISTS sources;")
        self.conn.commit()
        return found

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- writing --------------------------------------------------------

    #: Explicit column list rather than a positional `VALUES (?,?,?…)`.
    #: The positional form silently mis-maps every column after an insertion
    #: point the moment someone adds a field to the DDL — which is exactly what
    #: schema v2 did.
    _EVENT_COLUMNS = (
        "event_id", "session_id", "provider", "seq", "ts", "role", "model",
        "tok_input", "tok_output", "tok_cache_read", "tok_cache_write", "tok_reasoning",
        "cost_usd", "text",
        "tool_kind", "tool_raw_name", "tool_in_bytes", "tool_out_bytes",
        "tool_exit_code", "tool_target", "tool_error_text",
        "context_tokens", "context_limit",
        "cwd", "git_branch", "repo_id", "repo_label",
    )

    def add_events(self, events: Iterable[Event]) -> int:
        rows = []
        for e in events:
            u = e.usage or Usage()
            t = e.tool
            rows.append(
                (
                    e.event_id, e.session_id, e.provider.value, e.seq, e.ts, e.role.value,
                    e.model,
                    u.input, u.output, u.cache_read, u.cache_write, u.reasoning,
                    e.cost_usd, e.text,
                    t.kind.value if t else None,
                    t.raw_name if t else None,
                    t.input_bytes if t else 0,
                    t.output_bytes if t else 0,
                    t.exit_code if t else None,
                    t.target_path if t else None,
                    t.error_text if t else None,
                    e.context_tokens, e.context_limit,
                    e.cwd, e.git_branch, e.repo_id, e.repo_label,
                )
            )
        if not rows:
            return 0
        cols = ", ".join(self._EVENT_COLUMNS)
        placeholders = ", ".join("?" * len(self._EVENT_COLUMNS))
        cur = self.conn.executemany(
            f"INSERT OR IGNORE INTO events ({cols}) VALUES ({placeholders})", rows
        )
        self.conn.commit()
        return cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

    def source_is_current(self, path: str, size: int, mtime: float, schema_version: str) -> bool:
        row = self.conn.execute(
            "SELECT size_bytes, mtime, schema_version FROM sources WHERE path = ?", (path,)
        ).fetchone()
        if not row:
            return False
        return (
            row["size_bytes"] == size
            and abs((row["mtime"] or 0) - mtime) < 1e-6
            and row["schema_version"] == schema_version
        )

    def mark_source(
        self, path: str, provider: str, size: int, mtime: float, schema_version: str
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,datetime('now'))",
            (path, provider, size, mtime, schema_version),
        )
        self.conn.commit()

    def reset(self) -> None:
        self.conn.executescript("DELETE FROM events; DELETE FROM sources;")
        self.conn.commit()

    # --- reading --------------------------------------------------------

    def _where(
        self,
        provider: str | None,
        since: str | None,
        repo_id: str | None,
        session_id: str | None = None,
    ):
        clauses, params = [], []
        if provider:
            clauses.append("provider = ?")
            params.append(provider)
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        if repo_id:
            clauses.append("repo_id = ?")
            params.append(repo_id)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), params

    def session_ids(
        self,
        provider: str | None = None,
        since: str | None = None,
        repo_id: str | None = None,
        session_id: str | None = None,
    ) -> list[str]:
        w, p = self._where(provider, since, repo_id, session_id)
        rows = self.conn.execute(
            f"SELECT DISTINCT session_id FROM events{w} ORDER BY session_id", p
        ).fetchall()
        return [r["session_id"] for r in rows]

    def projects(self) -> list[dict]:
        """Distinct projects for the scope filter.

        `repo_label` is the display name (a basename, never a path — see
        `Event.repo_label`); `repo_id` stays the value the filter actually
        queries on.
        """
        rows = self.conn.execute(
            "SELECT repo_id, "
            "       COALESCE(MAX(repo_label), '') AS label, "
            "       COUNT(DISTINCT session_id) AS sessions, "
            "       COALESCE(SUM(cost_usd), 0) AS cost_usd, "
            "       MAX(ts) AS last_seen "
            "FROM events WHERE repo_id IS NOT NULL "
            "GROUP BY repo_id ORDER BY cost_usd DESC"
        ).fetchall()
        out = [
            {
                "repo_id": r["repo_id"],
                "label": r["label"] or r["repo_id"][:8],
                "sessions": r["sessions"],
                "cost_usd": round(r["cost_usd"] or 0.0, 6),
                "last_seen": r["last_seen"] or "",
            }
            for r in rows
        ]
        # Two distinct projects can share a basename ("api" under two different
        # parents), and the agents with no project concept all label by agent
        # name. Identical entries in a picker are unusable, so disambiguate the
        # collisions — and only the collisions.
        seen: dict[str, int] = {}
        for p in out:
            seen[p["label"]] = seen.get(p["label"], 0) + 1
        for p in out:
            if seen[p["label"]] > 1:
                p["label"] = f"{p['label']} ({p['repo_id'][:6]})"
        return out

    def sessions_index(
        self,
        provider: str | None = None,
        since: str | None = None,
        repo_id: str | None = None,
        limit: int = 300,
    ) -> list[dict]:
        """Lightweight session list for the session dropdown — deliberately not
        `sessions()`, which reconstructs every event of every session just to
        populate a picker."""
        w, p = self._where(provider, since, repo_id)
        rows = self.conn.execute(
            "SELECT session_id, "
            "       MAX(provider) AS provider, "
            "       MIN(ts) AS started_at, "
            "       COALESCE(SUM(cost_usd), 0) AS cost_usd, "
            "       COALESCE(MAX(repo_label), '') AS label "
            f"FROM events{w} "
            "GROUP BY session_id ORDER BY cost_usd DESC LIMIT ?",
            [*p, limit],
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                "provider": r["provider"],
                "started_at": r["started_at"] or "",
                "cost_usd": round(r["cost_usd"] or 0.0, 6),
                "label": r["label"] or "",
            }
            for r in rows
        ]

    def resolve_project(self, needle: str | None) -> str | None:
        """Accept either a repo_id or a human label from the CLI.

        Nobody types a 16-char hash, so `--project checkout-service` has to
        work; matching is case-insensitive and falls back to a unique prefix.
        """
        if not needle:
            return None
        wanted = needle.strip().lower()
        projects = self.projects()
        for pr in projects:
            if pr["repo_id"] == needle:
                return pr["repo_id"]
        for pr in projects:
            if pr["label"].lower() == wanted:
                return pr["repo_id"]
        matches = [pr for pr in projects if pr["label"].lower().startswith(wanted)]
        if len(matches) == 1:
            return matches[0]["repo_id"]
        return needle  # let it match nothing rather than silently widening scope

    def events_for(self, session_id: str) -> list[Event]:
        rows = self.conn.execute(
            "SELECT * FROM events WHERE session_id = ? ORDER BY seq", (session_id,)
        ).fetchall()
        return [_row_to_event(r) for r in rows]

    def iter_sessions(
        self,
        provider: str | None = None,
        since: str | None = None,
        repo_id: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[tuple[str, list[Event]]]:
        for sid in self.session_ids(provider, since, repo_id, session_id):
            yield sid, self.events_for(sid)

    def sessions(
        self,
        provider: str | None = None,
        since: str | None = None,
        repo_id: str | None = None,
        session_id: str | None = None,
    ) -> list[Session]:
        from .analytics.rollups import summarize_session

        return [
            summarize_session(evs)
            for _, evs in self.iter_sessions(provider, since, repo_id, session_id)
        ]

    def counts(self) -> dict[str, int]:
        row = self.conn.execute(
            "SELECT COUNT(*) n, COUNT(DISTINCT session_id) s, "
            "COALESCE(SUM(cost_usd),0) c FROM events"
        ).fetchone()
        return {"events": row["n"], "sessions": row["s"], "cost_usd": row["c"]}

    def distinct_models(self) -> list[str]:
        """Every model name actually present in history — cheap because it's
        an indexed-column query, not a full event reconstruction. Used to
        drive Prompt Lab's "same prompt across models" comparison without
        paying to rebuild every `Event` just to read one column off it."""
        return [
            r["model"]
            for r in self.conn.execute(
                "SELECT DISTINCT model FROM events WHERE model IS NOT NULL "
                "ORDER BY model"
            ).fetchall()
        ]

    def providers(self) -> list[str]:
        return [
            r["provider"]
            for r in self.conn.execute(
                "SELECT DISTINCT provider FROM events ORDER BY provider"
            ).fetchall()
        ]

    def model_token_totals(self) -> dict[str, int]:
        """Total tokens billed per model, across every event that named one.

        Used to find models the pricing table has never heard of — those
        events still get cost recomputed (see `pricing.cost_of`), but an
        unrecognised model silently prices at $0, so a heavy-usage model this
        table doesn't know about would otherwise undercount total spend with
        no visible sign anything was skipped. See `cmd_doctor`'s pricing
        coverage check."""
        return {
            r["model"]: r["toks"]
            for r in self.conn.execute(
                "SELECT model, SUM(tok_input + tok_output + tok_cache_read "
                "+ tok_cache_write + tok_reasoning) AS toks FROM events "
                "WHERE model IS NOT NULL GROUP BY model"
            ).fetchall()
        }


def _row_to_event(r: sqlite3.Row) -> Event:
    from .schema import ToolInfo

    tool = None
    if r["tool_kind"]:
        tool = ToolInfo(
            kind=ToolKind(r["tool_kind"]),
            raw_name=r["tool_raw_name"] or "",
            input_bytes=r["tool_in_bytes"] or 0,
            output_bytes=r["tool_out_bytes"] or 0,
            exit_code=r["tool_exit_code"],
            target_path=r["tool_target"],
            error_text=r["tool_error_text"],
        )
    usage = Usage(
        input=r["tok_input"] or 0,
        output=r["tok_output"] or 0,
        cache_read=r["tok_cache_read"] or 0,
        cache_write=r["tok_cache_write"] or 0,
        reasoning=r["tok_reasoning"] or 0,
    )
    return Event(
        session_id=r["session_id"],
        provider=Provider(r["provider"]),
        seq=r["seq"],
        ts=r["ts"] or "",
        role=Role(r["role"]),
        model=r["model"],
        usage=usage if usage.total else None,
        cost_usd=r["cost_usd"] or 0.0,
        text=r["text"],
        tool=tool,
        context_tokens=r["context_tokens"],
        context_limit=r["context_limit"],
        cwd=r["cwd"],
        git_branch=r["git_branch"],
        repo_id=r["repo_id"],
        repo_label=r["repo_label"],
    )
