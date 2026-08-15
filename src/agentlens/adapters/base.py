"""The adapter contract.

Adding an agent means implementing this interface and committing a synthetic
fixture plus a golden test. Nothing else in the codebase changes. That is the
whole point of the normalized schema.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..schema import Event, Provider


@dataclass(frozen=True, slots=True)
class SourceFile:
    path: Path
    size_bytes: int
    mtime: float

    @classmethod
    def of(cls, path: Path) -> "SourceFile":
        st = path.stat()
        return cls(path=path, size_bytes=st.st_size, mtime=st.st_mtime)


class Adapter(ABC):
    """Read one agent's on-disk history and emit normalized events."""

    provider: Provider
    #: Bump when the parse logic changes meaningfully; used to invalidate
    #: previously-ingested rows so users get corrected numbers on upgrade.
    schema_version: str = "1"
    #: Human label for CLI output.
    display_name: str = ""

    @abstractmethod
    def roots(self) -> list[Path]:
        """Candidate directories to search, most likely first."""

    @abstractmethod
    def discover(self) -> list[SourceFile]:
        """Locate every history file this adapter can parse."""

    @abstractmethod
    def parse(self, source: SourceFile) -> Iterator[Event]:
        """Stream events out of one file.

        Must not read the whole file into memory. Real users have multi-hundred
        megabyte session directories.
        """

    def detect(self) -> bool:
        """Cheap probe: is this agent installed on this machine at all?

        Deliberately permission-tolerant: a directory that exists but can't be
        *listed* still counts as "found" so `doctor` can tell the user to fix
        permissions rather than reporting the agent as not installed.
        """
        for r in self.roots():
            try:
                if r.exists():
                    return True
            except PermissionError:
                return True
        return False

    #: Populated by discover() when a root exists but couldn't be read, so
    #: `doctor` can print OS-specific guidance instead of silently finding
    #: nothing. See `permission_hint()` below.
    last_permission_error: Path | None = None

    def permission_hint(self) -> str | None:
        """OS-specific remediation text for a permission-denied root.

        AgentLens is a plain local process with no special privileges by
        design — it relies on ordinary OS file permissions, which differ
        enough across platforms that a flat "permission denied" is not
        actionable on its own.
        """
        if not self.last_permission_error:
            return None
        p = str(self.last_permission_error)
        import sys

        if sys.platform == "darwin":
            return (
                f"macOS blocked reading {p}. If your terminal is sandboxed "
                "(iTerm/Terminal under System Settings > Privacy & Security > "
                "Files and Folders), grant it Full Disk Access, or run "
                "`chmod -R u+rX` on the folder if you own it but it's unreadable."
            )
        if sys.platform.startswith("win"):
            return (
                f"Windows blocked reading {p}. If Controlled Folder Access is "
                "on (Windows Security > Virus & threat protection > "
                "Ransomware protection), add python.exe / agentlens.exe to the "
                "allowed apps list, or check the folder's Properties > "
                "Security tab for your user's read permission."
            )
        return (
            f"Permission denied reading {p}. Check ownership and mode with "
            "`ls -la` and `chmod -R u+rX` the folder if you own it."
        )

    # --- shared helpers -------------------------------------------------

    @staticmethod
    def safe_glob(root: Path, pattern: str, adapter: "Adapter") -> list[Path]:
        """rglob that degrades to an empty result on permission errors instead
        of crashing `ingest` for every agent because one directory is locked
        down."""
        try:
            return [p for p in root.rglob(pattern) if p.is_file()]
        except PermissionError:
            adapter.last_permission_error = root
            return []

    @staticmethod
    def dig(obj: Any, *names: str) -> Any:
        """First non-None hit for any of `names`, searched at the top level and
        one level deep. Agents rename the same concept ("input_tokens" vs
        "input" vs "prompt_tokens") across releases; this is how adapters stay
        readable instead of a wall of `.get(...) or .get(...) or .get(...)`."""
        if not isinstance(obj, dict):
            return None
        for n in names:
            if obj.get(n) is not None:
                return obj[n]
        for v in obj.values():
            if isinstance(v, dict):
                for n in names:
                    if v.get(n) is not None:
                        return v[n]
        return None

    @staticmethod
    def iter_jsonl(path: Path) -> Iterable[dict]:
        """Line-delimited JSON, resilient to the truncated last line you get
        when a session is still being written to."""
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except OSError:
            return

    @staticmethod
    def text_of(content) -> str:
        """Flatten the several shapes a message body arrives in."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") in (None, "text", "input_text", "output_text"):
                        parts.append(str(block.get("text", "")))
            return "\n".join(p for p in parts if p)
        if isinstance(content, dict):
            return str(content.get("text", ""))
        return ""

    @staticmethod
    def open_sqlite_ro(path: Path):
        """Read-only connection, so a buggy adapter can never corrupt the
        agent's own live database. Returns None on any failure — a locked or
        mid-write SQLite file must degrade gracefully, not crash ingest."""
        import sqlite3

        try:
            uri = f"file:{path.as_posix()}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=2.0)
            conn.row_factory = sqlite3.Row
            # `SELECT 1` alone never touches the file's page structure, so a
            # plain-text file with no SQLite header would connect "successfully"
            # and only fail later, mid-parse. Reading sqlite_master forces
            # SQLite to actually validate the file header now, while we can
            # still fail closed cleanly.
            conn.execute("SELECT name FROM sqlite_master LIMIT 1")
            return conn
        except sqlite3.Error:
            return None

    @staticmethod
    def sqlite_tables(conn) -> list[str]:
        import sqlite3

        try:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        except sqlite3.Error:
            return []
        return [r["name"] for r in rows]

    @staticmethod
    def sqlite_columns(conn, table: str) -> list[str]:
        try:
            rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
        except Exception:
            return []
        return [r["name"] for r in rows]

    @staticmethod
    def seq_from_ts(ts: object, local_index: int = 0) -> int:
        """A `seq` value derived from wall-clock time instead of a per-call
        counter.

        Most adapters map one session to one file, so a counter starting at 0
        in `parse()` is scoped correctly. Adapters where one session spans many
        files (OpenCode: one file per message) can't use a counter — ingest
        runs incrementally across separate process invocations, and a counter
        that resets to 0 each run would assign message 6 the same `seq` as
        message 1, scrambling event order the moment two ingest runs happen
        days apart. Deriving `seq` from the message's own timestamp instead
        makes ordering stable across runs and idempotent by construction.

        `local_index` disambiguates multiple events inside one message
        (assistant text, then N tool calls) that share one timestamp.
        """
        millis = 0
        if isinstance(ts, (int, float)):
            millis = int(ts if ts > 10**12 else ts * 1000)
        elif isinstance(ts, str) and ts:
            try:
                from datetime import datetime

                s = ts.replace("Z", "+00:00")
                millis = int(datetime.fromisoformat(s).timestamp() * 1000)
            except ValueError:
                try:
                    millis = int(float(ts) * (1000 if float(ts) < 10**12 else 1))
                except ValueError:
                    millis = 0
        return millis * 1000 + local_index

    @staticmethod
    def read_json(path: Path) -> dict | list | None:
        """A single JSON document (not JSONL). Returns None rather than
        raising — a half-written file while the agent is still running must
        not abort the whole ingest."""
        try:
            return json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def size_of(obj) -> int:
        if obj is None:
            return 0
        if isinstance(obj, str):
            return len(obj.encode("utf-8", errors="replace"))
        try:
            return len(json.dumps(obj, default=str).encode("utf-8"))
        except (TypeError, ValueError):
            return 0
