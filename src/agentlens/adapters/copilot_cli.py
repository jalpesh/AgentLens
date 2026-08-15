"""GitHub Copilot CLI adapter.

⚠️ EXPERIMENTAL. GitHub's own docs confirm sessions are persisted under
``~/.copilot/session-state/`` with "structured data also maintained" in a
SQLite database at ``~/.copilot/session-store.db``, and that a session
contains "your prompts, Copilot's responses, the tools that were used, and
details of files that were modified." The table and column schema of that
database is not published anywhere at the time this was written.

Rather than hardcode a schema we can't verify — and that GitHub can change
without notice, same risk class as Cursor's ``state.vscdb`` — this adapter
introspects the database at read time: it looks for a table whose columns
look like a message log (role/content/timestamp-shaped names), and pulls
what it can out of each row, including digging into any JSON-text column for
nested usage/model fields.

Included anyway, at the user's explicit request, because Copilot CLI has by
far the largest install base of any agent covered here. If it finds nothing
on your database, run ``PRAGMA table_info(<table>)`` yourself against
``session-store.db`` and open an issue with the output — no real data
required, just column names.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from ..schema import Event, Provider, Role, Usage, hash_repo
from ..pricing import context_limit_of, cost_of
from .base import Adapter, SourceFile

_TABLE_HINTS = ("message", "event", "turn", "chat", "log", "session")
_ROLE_COLS = ("role", "type", "sender", "author")
_CONTENT_COLS = ("content", "text", "body", "message", "prompt", "response")
_TS_COLS = ("timestamp", "created_at", "createdat", "time", "ts")
_SESSION_COLS = ("session_id", "sessionid", "session", "conversation_id", "thread_id")
_MODEL_COLS = ("model", "model_id", "modelid")


class CopilotCliAdapter(Adapter):
    provider = Provider.COPILOT_CLI
    display_name = "GitHub Copilot CLI"
    schema_version = "2-experimental"

    def roots(self) -> list[Path]:
        env = os.environ.get("COPILOT_HOME")
        base = Path(env) if env else Path.home() / ".copilot"
        return [base]

    def discover(self) -> list[SourceFile]:
        out: list[SourceFile] = []
        for root in self.roots():
            db = root / "session-store.db"
            if db.exists() and db.stat().st_size > 0:
                out.append(SourceFile.of(db))
        return out

    def parse(self, source: SourceFile) -> Iterator[Event]:
        conn = self.open_sqlite_ro(source.path)
        if conn is None:
            return
        try:
            table, cols = self._find_message_table(conn)
            if not table:
                return

            role_col = self._pick(cols, _ROLE_COLS)
            content_col = self._pick(cols, _CONTENT_COLS)
            ts_col = self._pick(cols, _TS_COLS)
            session_col = self._pick(cols, _SESSION_COLS)
            model_col = self._pick(cols, _MODEL_COLS)
            if not content_col:
                return  # nothing worth extracting without a content column

            select_cols = ", ".join(f'"{c}"' for c in cols)
            try:
                rows = conn.execute(f'SELECT {select_cols} FROM "{table}"').fetchall()
            except Exception:
                return

            for row in rows:
                r = dict(row)
                content = r.get(content_col)
                text, usage, model = self._extract(content, r, model_col)
                if not text and usage is None:
                    continue

                session_id = str(r.get(session_col) or source.path.stem)
                ts = str(r.get(ts_col) or "")
                role_raw = str(r.get(role_col) or "").lower() if role_col else ""
                role = Role.ASSISTANT if "assist" in role_raw or "model" in role_raw else Role.USER
                if not role_col:
                    role = Role.ASSISTANT if usage else Role.USER

                yield Event(
                    session_id=session_id, provider=self.provider,
                    seq=self.seq_from_ts(ts), ts=ts, role=role,
                    model=model, usage=usage,
                    cost_usd=cost_of(model, usage) if usage else 0.0,
                    text=text or None,
                    context_tokens=(usage.input + usage.cache_read) if usage else None,
                    context_limit=context_limit_of(model) if model else None,
                    repo_id=hash_repo(str(source.path)),
                    # This format records no working directory, so sessions
                    # can't be attributed to a project. Label them by agent
                    # rather than showing a hash fragment in the project
                    # filter and implying a project we don't actually know.
                    repo_label=self.display_name,
                )
        finally:
            conn.close()

    # --- introspection helpers ------------------------------------------

    def _find_message_table(self, conn) -> tuple[str | None, list[str]]:
        best: tuple[str, list[str]] | None = None
        for name in self.sqlite_tables(conn):
            low = name.lower()
            if not any(h in low for h in _TABLE_HINTS):
                continue
            cols = self.sqlite_columns(conn, name)
            cols_low = [c.lower() for c in cols]
            score = sum(1 for c in cols_low if any(h in c for h in _CONTENT_COLS))
            if score and (best is None or score > len(best[1])):
                best = (name, cols)
        return best if best else (None, [])

    @staticmethod
    def _pick(cols: list[str], candidates: tuple[str, ...]) -> str | None:
        low = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand in low:
                return low[cand]
        for c_low, c in low.items():
            if any(cand in c_low for cand in candidates):
                return c
        return None

    def _extract(self, content, row: dict, model_col: str | None):
        """Content may be plain text or a JSON blob carrying nested usage and
        model fields — try JSON first, fall back to treating it as text."""
        usage = None
        model = row.get(model_col) if model_col else None
        text = ""

        if isinstance(content, (bytes, bytearray)):
            try:
                content = content.decode("utf-8", errors="replace")
            except Exception:
                content = None

        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    parsed = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    text = self.text_of(
                        self.dig(parsed, "text", "content", "message")
                    ) or str(parsed.get("text") or "")
                    usage_raw = self.dig(parsed, "usage", "tokenUsage")
                    if isinstance(usage_raw, dict):
                        usage = Usage(
                            input=int(self.dig(usage_raw, "input", "prompt_tokens") or 0),
                            output=int(self.dig(usage_raw, "output", "completion_tokens") or 0),
                            cache_read=int(self.dig(usage_raw, "cached") or 0),
                        )
                    model = model or self.dig(parsed, "model")
                else:
                    text = stripped
            else:
                text = content

        return text, usage, model
