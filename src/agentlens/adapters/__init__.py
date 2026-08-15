"""Adapter registry.

Phase 1 shipped Claude Code, Codex and Junie — full support, formats verified
against public docs and reference implementations.

Phase 2 adds Gemini CLI, OpenCode, GitHub Copilot CLI and Goose. Of these,
OpenCode's plain-JSON-per-file layout is documented well enough to ship as
full support. Gemini CLI's storage location is documented but its file format
is not, so it's marked experimental. Copilot CLI and Goose (>=1.10) turned out
to store sessions in undocumented SQLite databases — the same complexity class
as Cursor/Windsurf — so they're marked experimental too: built defensively via
schema introspection rather than hardcoded queries, included because Copilot
CLI in particular has the largest install base of any agent here, but liable
to silently find nothing if a vendor update changes the schema.

Cursor and Windsurf remain Phase 3 — multiple undocumented, multi-gigabyte
SQLite stores per agent, changing without notice. Aider is not yet scheduled.
"""

from __future__ import annotations

from .base import Adapter, SourceFile
from .claude_code import ClaudeCodeAdapter
from .codex import CodexAdapter
from .copilot_cli import CopilotCliAdapter
from .gemini_cli import GeminiCliAdapter
from .goose import GooseAdapter
from .junie import JunieAdapter
from .opencode import OpenCodeAdapter
from ..schema import Provider

ADAPTERS: list[Adapter] = [
    ClaudeCodeAdapter(),
    CodexAdapter(),
    JunieAdapter(),
    OpenCodeAdapter(),
    GeminiCliAdapter(),
    CopilotCliAdapter(),
    GooseAdapter(),
]

#: Adapters whose format is not verified against public documentation or a
#: reference implementation. `doctor` and the README both surface this rather
#: than letting "installed" quietly imply "reliable".
EXPERIMENTAL: set[Provider] = {
    a.provider for a in ADAPTERS if a.schema_version.endswith("-experimental")
}

BY_PROVIDER: dict[Provider, Adapter] = {a.provider: a for a in ADAPTERS}


def get(name: str) -> Adapter | None:
    try:
        return BY_PROVIDER.get(Provider(name))
    except ValueError:
        return None


def installed() -> list[Adapter]:
    """Adapters whose agent is actually present on this machine."""
    return [a for a in ADAPTERS if a.detect()]


__all__ = [
    "Adapter",
    "SourceFile",
    "ADAPTERS",
    "EXPERIMENTAL",
    "BY_PROVIDER",
    "get",
    "installed",
    "ClaudeCodeAdapter",
    "CodexAdapter",
    "JunieAdapter",
    "OpenCodeAdapter",
    "GeminiCliAdapter",
    "CopilotCliAdapter",
    "GooseAdapter",
]
