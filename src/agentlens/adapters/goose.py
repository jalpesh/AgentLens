"""Goose adapter.

⚠️ EXPERIMENTAL, and a two-format adapter because the vendor changed formats
underneath it:

* **Goose ≥ 1.10** stores sessions in a SQLite database, ``sessions.db``, at
  ``~/.local/share/goose/sessions/`` (macOS/Linux) or
  ``%APPDATA%\\Block\\goose\\data\\sessions\\`` (Windows). The documented
  contents are session metadata, conversation messages with role information,
  tool calls with arguments/results, and "token usage statistics" — but not
  published column names, so this branch introspects the schema the same way
  the Copilot CLI adapter does.
* **Goose < 1.10** wrote one ``.jsonl`` file per session in the same
  directory. That branch is a normal JSONL parser.

Both branches are attempted; whichever finds files wins. If your Goose
install uses neither shape, `agentlens doctor -v` will show 0 files found —
that's the honest failure mode for an unverified format, not a crash.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

from ..schema import (
    Event,
    Provider,
    Role,
    ToolInfo,
    Usage,
    hash_repo,
    normalize_tool,
)
from ..pricing import context_limit_of, cost_of
from .base import Adapter, SourceFile

_ROLE_MAP = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM}
_TABLE_HINTS = ("message", "conversation", "session")
_ROLE_COLS = ("role", "sender")
_CONTENT_COLS = ("content", "text", "body")
_TS_COLS = ("timestamp", "created_at", "createdat", "time")
_SESSION_COLS = ("session_id", "sessionid", "session")
_MODEL_COLS = ("model", "model_id")


class GooseAdapter(Adapter):
    provider = Provider.GOOSE
    display_name = "Goose"
    schema_version = "2-experimental"

    def roots(self) -> list[Path]:
        env = os.environ.get("GOOSE_HOME")
        if env:
            return [Path(env)]
        if sys.platform.startswith("win"):
            appdata = os.environ.get("APPDATA")
            base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
            return [base / "Block" / "goose" / "data" / "sessions"]
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
        return [base / "goose" / "sessions"]

    def discover(self) -> list[SourceFile]:
        out: list[SourceFile] = []
        for root in self.roots():
            if not root.exists():
                continue
            db = root / "sessions.db"
            if db.exists() and db.stat().st_size > 0:
                out.append(SourceFile.of(db))
            for p in self.safe_glob(root, "*.jsonl", self):
                if p.stat().st_size > 0:
                    out.append(SourceFile.of(p))
        return sorted(out, key=lambda s: s.mtime)

    def parse(self, source: SourceFile) -> Iterator[Event]:
        if source.path.suffix == ".db":
            yield from self._parse_sqlite(source)
        else:
            yield from self._parse_jsonl(source)

    # --- Goose >= 1.10: SQLite -------------------------------------------

    def _parse_sqlite(self, source: SourceFile) -> Iterator[Event]:
        conn = self.open_sqlite_ro(source.path)
        if conn is None:
            return
        try:
            table, cols = self._find_message_table(conn)
            if not table:
                return
            content_col = self._pick(cols, _CONTENT_COLS)
            if not content_col:
                return
            role_col = self._pick(cols, _ROLE_COLS)
            ts_col = self._pick(cols, _TS_COLS)
            session_col = self._pick(cols, _SESSION_COLS)
            model_col = self._pick(cols, _MODEL_COLS)

            select_cols = ", ".join(f'"{c}"' for c in cols)
            try:
                rows = conn.execute(f'SELECT {select_cols} FROM "{table}"').fetchall()
            except Exception:
                return

            for row in rows:
                r = dict(row)
                text = r.get(content_col)
                text = text if isinstance(text, str) else self.text_of(text)
                if not text or not text.strip():
                    continue
                role_raw = str(r.get(role_col) or "").lower() if role_col else ""
                role = _ROLE_MAP.get(role_raw, Role.USER)
                session_id = str(r.get(session_col) or source.path.stem)
                ts = str(r.get(ts_col) or "")
                model = r.get(model_col) if model_col else None

                yield Event(
                    session_id=session_id, provider=self.provider,
                    seq=self.seq_from_ts(ts), ts=ts, role=role,
                    model=model if role is Role.ASSISTANT else None,
                    text=text,
                    repo_id=hash_repo(str(source.path)),
                    # No working directory in this format — label by agent
                    # rather than implying a project we can't determine.
                    repo_label=self.display_name,
                )
        finally:
            conn.close()

    def _find_message_table(self, conn) -> tuple[str | None, list[str]]:
        best: tuple[str, list[str]] | None = None
        for name in self.sqlite_tables(conn):
            if not any(h in name.lower() for h in _TABLE_HINTS):
                continue
            cols = self.sqlite_columns(conn, name)
            score = sum(1 for c in cols if any(h in c.lower() for h in _CONTENT_COLS))
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

    # --- Goose < 1.10: JSONL ----------------------------------------------

    def _parse_jsonl(self, source: SourceFile) -> Iterator[Event]:
        session_id = source.path.stem
        repo_id = hash_repo(str(source.path))
        seq = 0
        for rec in self.iter_jsonl(source.path):
            seq += 1
            role_raw = str(self.dig(rec, "role") or "").lower()
            role = _ROLE_MAP.get(role_raw, Role.SYSTEM)
            ts = str(self.dig(rec, "timestamp", "created_at") or "")
            model = self.dig(rec, "model")
            text = self.text_of(self.dig(rec, "content", "text"))
            base = dict(
                session_id=session_id, provider=self.provider, seq=seq, ts=ts,
                repo_id=repo_id, repo_label=self.display_name,
            )

            if role is Role.ASSISTANT:
                usage_raw = self.dig(rec, "usage")
                usage = None
                if isinstance(usage_raw, dict):
                    usage = Usage(
                        input=int(self.dig(usage_raw, "input_tokens", "input") or 0),
                        output=int(self.dig(usage_raw, "output_tokens", "output") or 0),
                    )
                yield Event(
                    role=role, model=model, usage=usage,
                    cost_usd=cost_of(model, usage) if usage else 0.0,
                    text=text or None,
                    context_limit=context_limit_of(model) if model else None,
                    **base,
                )
                for call in rec.get("toolRequests") or rec.get("tool_calls") or []:
                    if not isinstance(call, dict):
                        continue
                    raw_name = str(self.dig(call, "name", "tool") or "")
                    seq += 1
                    yield Event(
                        role=Role.TOOL_CALL,
                        tool=ToolInfo(
                            kind=normalize_tool(raw_name),
                            raw_name=raw_name,
                            input_bytes=self.size_of(call.get("arguments")),
                        ),
                        session_id=session_id, provider=self.provider, seq=seq, ts=ts,
                        repo_id=repo_id, repo_label=self.display_name,
                    )
            elif text.strip():
                yield Event(role=Role.USER, text=text, **base)
