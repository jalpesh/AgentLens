"""Gemini CLI adapter.

⚠️ EXPERIMENTAL — Google documents *what* Gemini CLI records (prompts,
responses, tool executions, token usage including cache) and *where*
(``~/.gemini/tmp/<project-hash>/chats/``), but not the on-disk file format or
field names. This adapter is written defensively against several plausible
shapes (JSON array, JSONL, and an OpenAI-style ``choices`` envelope) and
degrades to zero events rather than raising when none match.

``~/.gemini`` is not exclusively Gemini CLI's directory. Google Antigravity —
a separate IDE product built on Gemini — also stores data there: tool-schema
manifests, artifact/task metadata (plans, walkthroughs, task summaries as
JSON), and its actual conversation "memory" as Protocol Buffer files, which
this adapter cannot read and makes no attempt to (undocumented, binary,
versioned — reverse-engineering it is out of scope; see the project's
"never guess a format" rule). Confirmed via a real user's history 2026-08-15:
every file under their ``~/.gemini`` was one of these two Antigravity shapes,
none were genuine Gemini CLI conversation logs — and because neither shape
carries a ``role`` field, every one of them was previously getting fabricated
into a fake "session" with a synthetic ``Role.USER`` turn built from whatever
text field happened to exist. Records with no recognized role are now
skipped outright rather than guessed at.

If it silently finds nothing on your machine: run
``agentlens doctor -v`` and file an issue with one sanitised sample file —
that is exactly the feedback loop that turns "experimental" into "full"
support, the same way the Claude Code and Codex adapters got there.
"""

from __future__ import annotations

import os
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
    repo_label_of,
)
from ..pricing import context_limit_of, cost_of
from .base import Adapter, SourceFile

_ROLE_MAP = {
    "user": Role.USER,
    "human": Role.USER,
    "model": Role.ASSISTANT,
    "assistant": Role.ASSISTANT,
    "function": Role.TOOL_RESULT,
    "tool": Role.TOOL_RESULT,
}


class GeminiCliAdapter(Adapter):
    provider = Provider.GEMINI_CLI
    display_name = "Gemini CLI"
    schema_version = "2-experimental"

    def roots(self) -> list[Path]:
        env = os.environ.get("GEMINI_HOME")
        base = Path(env) if env else Path.home() / ".gemini"
        return [base / "tmp", base / "chats", base]

    def discover(self) -> list[SourceFile]:
        out: dict[Path, SourceFile] = {}
        for root in self.roots():
            if not root.exists():
                continue
            for pattern in ("*.json", "*.jsonl"):
                for p in self.safe_glob(root, pattern, self):
                    if p.stat().st_size > 0 and p not in out:
                        out[p] = SourceFile.of(p)
        return sorted(out.values(), key=lambda s: s.mtime)

    def parse(self, source: SourceFile) -> Iterator[Event]:
        # Try JSONL first (one turn per line); fall back to a single JSON
        # document containing a `messages`/`history`/`chats` array. Both
        # shapes are attested by different Gemini CLI releases in the wild.
        records = list(self.iter_jsonl(source.path))
        if not records:
            doc = self.read_json(source.path)
            if isinstance(doc, list):
                records = [r for r in doc if isinstance(r, dict)]
            elif isinstance(doc, dict):
                for key in ("messages", "history", "chats", "turns"):
                    v = doc.get(key)
                    if isinstance(v, list):
                        records = [r for r in v if isinstance(r, dict)]
                        break
                else:
                    records = [doc]
        if not records:
            return

        session_id = source.path.stem
        cwd = self._project_dir_from_path(source.path)
        repo_id = hash_repo(cwd)
        seq = 0

        for rec in records:
            seq += 1
            ts = str(self.dig(rec, "timestamp", "ts", "createdAt") or "")
            model = self.dig(rec, "model", "modelId")
            role_raw = str(self.dig(rec, "role", "type") or "").lower()
            role = _ROLE_MAP.get(role_raw)
            if role is None:
                # No recognized role at all — most likely not a conversation
                # turn. `~/.gemini` isn't exclusively Gemini CLI's directory:
                # Google Antigravity (a separate IDE product built on Gemini)
                # also stores data there — tool-schema manifests
                # (`{"name","description","parameters"}`) and artifact/task
                # metadata (`{"artifactType","summary","updatedAt"}`) — none
                # of which carry a role field, none of which are chat logs.
                # Previously this fell through to the `else` branch below and
                # got fabricated into a `Role.USER` "prompt" out of whatever
                # text field happened to exist, which is how a real user's
                # history ended up with 30 "Gemini CLI sessions" that were
                # actually Antigravity task-completion notices and artifact
                # summaries — none ever carrying usage data, since only
                # genuine assistant turns do, which inflated the session
                # count while contributing nothing to any dollar figure.
                # Skipping outright is more honest than guessing at a role
                # for a record with none. Found via real usage data,
                # 2026-08-16.
                continue

            base = dict(
                session_id=session_id, provider=self.provider, seq=seq, ts=ts,
                cwd=cwd, repo_id=repo_id, repo_label=repo_label_of(cwd),
                raw_id=str(rec.get("id") or ""),
            )

            usage_raw = self.dig(rec, "usage", "tokenUsage", "usageMetadata")
            usage = None
            if isinstance(usage_raw, dict):
                usage = Usage(
                    input=int(
                        self.dig(usage_raw, "input", "inputTokens", "promptTokenCount") or 0
                    ),
                    output=int(
                        self.dig(usage_raw, "output", "outputTokens", "candidatesTokenCount")
                        or 0
                    ),
                    cache_read=int(
                        self.dig(usage_raw, "cached", "cachedContentTokenCount") or 0
                    ),
                )

            if role is Role.ASSISTANT:
                text = self.text_of(rec.get("content") or rec.get("text") or rec.get("parts"))
                yield Event(
                    role=role, model=model, usage=usage,
                    cost_usd=cost_of(model, usage),
                    text=text or None,
                    context_tokens=(usage.input + usage.cache_read) if usage else None,
                    context_limit=context_limit_of(model),
                    **base,
                )
                for call in rec.get("toolCalls") or rec.get("functionCalls") or []:
                    if not isinstance(call, dict):
                        continue
                    raw_name = str(self.dig(call, "name", "tool") or "")
                    seq += 1
                    yield Event(
                        role=Role.TOOL_CALL, model=model,
                        tool=ToolInfo(
                            kind=normalize_tool(raw_name),
                            raw_name=raw_name,
                            input_bytes=self.size_of(call.get("args")),
                            target_path=self.dig(call.get("args") or {}, "path", "file_path"),
                        ),
                        session_id=session_id, provider=self.provider, seq=seq, ts=ts,
                        cwd=cwd, repo_id=repo_id, repo_label=repo_label_of(cwd),
                    )
            elif role is Role.TOOL_RESULT:
                raw_name = str(self.dig(rec, "name", "tool") or "")
                yield Event(
                    role=role,
                    tool=ToolInfo(
                        kind=normalize_tool(raw_name),
                        raw_name=raw_name,
                        output_bytes=self.size_of(rec.get("response") or rec.get("content")),
                    ),
                    **base,
                )
            else:
                text = self.text_of(rec.get("content") or rec.get("text") or rec.get("parts"))
                if text.strip():
                    yield Event(role=Role.USER, text=text, **base)

    @staticmethod
    def _project_dir_from_path(path: Path) -> str | None:
        """Extract the per-project hash from ``.../tmp/<hash>/chats/<file>``.

        Anchored on the LAST "tmp" component, not the first. A naive
        `parts.index("tmp")` matches the wrong segment whenever the whole tree
        lives under a system temp directory — which is exactly what happens on
        a machine using `GEMINI_HOME=/tmp/...`, and it silently mislabels every
        project as "tmp"'s child.
        """
        parts = path.parts
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "tmp" and i + 1 < len(parts):
                return parts[i + 1]
        return None
