"""Junie (JetBrains) adapter.

Two sources under ``$JUNIE_HOME`` (default ``~/.junie``):

1. **Junie CLI** (current): one folder per session,
   ``sessions/<id>/{events.jsonl, summary.json, state.json, transcript.md}``.
   Verified against Junie CLI 26.11. We read the request (``state.json`` →
   ``issue.description``), token usage per model call
   (``LlmResponseMetadataEvent``), context-window fill
   (``ContextWindowReportEvent``) and failures. Tool steps are not parsed yet,
   so outcomes show as unknown rather than guessed.
2. **Legacy** ``history.json``: a single JSON document whose shape moved between
   IDE releases, so its session array is located structurally.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

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

_SESSION_KEYS = ("sessions", "history", "tasks", "conversations", "items", "entries")
_STEP_KEYS = ("steps", "messages", "events", "turns", "requests", "calls")


class JunieAdapter(Adapter):
    provider = Provider.JUNIE
    display_name = "Junie"
    schema_version = "3"

    def roots(self) -> list[Path]:
        env = os.environ.get("JUNIE_HOME")
        return [Path(env) if env else Path.home() / ".junie"]

    def discover(self) -> list[SourceFile]:
        out: list[SourceFile] = []
        for root in self.roots():
            if not root.exists():
                continue
            for name in ("history.json", "sessions.json"):
                p = root / name
                if p.is_file() and p.stat().st_size > 0:
                    out.append(SourceFile.of(p))
            for p in root.glob("**/*.json"):
                if p.is_file() and p.stat().st_size > 0 and p.name in ("history.json",):
                    sf = SourceFile.of(p)
                    if sf not in out:
                        out.append(sf)
            for p in sorted(root.glob("sessions/*/events.jsonl")):
                if p.stat().st_size > 0:
                    out.append(SourceFile.of(p))
        return out

    # --- Junie CLI sessions --------------------------------------------

    @staticmethod
    def _ts(ms) -> str:
        try:
            return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).isoformat()
        except (TypeError, ValueError, OverflowError, OSError):
            return ""

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            return d if isinstance(d, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _parse_cli_session(self, events_path: Path) -> Iterator[Event]:
        folder = events_path.parent
        summary = self._read_json(folder / "summary.json")
        sid = str(summary.get("sessionId") or folder.name)
        cwd = summary.get("projectDir")
        base = dict(session_id=sid, provider=self.provider, cwd=cwd,
                    repo_id=hash_repo(cwd), repo_label=repo_label_of(cwd))
        seq = 0

        # The request lives in the last agent state, stored as a JSON string.
        request = ""
        state = self._read_json(folder / "state.json")
        blob = ((state.get("event") or {}).get("agentEvent") or {}).get("blob")
        if isinstance(blob, str):
            try:
                issue = (json.loads(blob).get("lastAgentState") or {}).get("issue") or {}
                request = str(issue.get("description") or "")
            except (json.JSONDecodeError, AttributeError):
                pass
        start = self._ts(summary.get("createdAt"))
        if request.strip():
            seq += 1
            yield Event(seq=seq, role=Role.USER, ts=start, text=request, raw_id="request", **base)
        if summary.get("taskName"):
            seq += 1
            yield Event(seq=seq, role=Role.SYSTEM, ts=start, text=str(summary["taskName"]),
                        raw_id="title", **base)

        ctx: tuple[int, int] | None = None
        try:
            lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        for i, line in enumerate(lines):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = ((d.get("event") or {}).get("agentEvent") or {}) if isinstance(d, dict) else {}
            kind, ts = a.get("kind"), self._ts(d.get("timestampMs"))
            if kind == "ContextWindowReportEvent":
                ctx = (int(a.get("used") or 0), int(a.get("size") or 0))
            elif kind == "LlmResponseMetadataEvent":
                for j, m in enumerate(a.get("modelUsage") or []):
                    cached = int(m.get("cacheInputTokens") or 0)
                    # inputTokens includes the cached part (seen: 6216 in, 6158 cached).
                    usage = Usage(input=max(0, int(m.get("inputTokens") or 0) - cached),
                                  output=int(m.get("outputTokens") or 0), cache_read=cached,
                                  cache_write=int(m.get("cacheCreateTokens") or 0))
                    model = m.get("model")
                    seq += 1
                    yield Event(seq=seq, role=Role.ASSISTANT, ts=ts, model=model, usage=usage,
                                cost_usd=cost_of(model, usage),
                                context_tokens=ctx[0] if ctx else usage.input + usage.cache_read,
                                context_limit=(ctx[1] if ctx and ctx[1] else
                                               context_limit_of(model)),
                                raw_id=f"llm-{i}-{j}", **base)
            elif kind == "AgentFailureEvent":
                seq += 1
                yield Event(seq=seq, role=Role.SYSTEM, ts=ts,
                            text=f"Junie failed: {a.get('message') or a.get('errorCode') or ''}",
                            raw_id=f"fail-{i}", **base)

    # --- structural helpers --------------------------------------------

    @staticmethod
    def _find_sessions(doc: Any) -> list[dict]:
        """Junie has moved this key around between releases. Find it by shape."""
        if isinstance(doc, list):
            return [d for d in doc if isinstance(d, dict)]
        if not isinstance(doc, dict):
            return []
        for k in _SESSION_KEYS:
            v = doc.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        # Fall back to the longest list-of-dicts anywhere one level down.
        best: list[dict] = []
        for v in doc.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and len(v) > len(best):
                best = v
        return best

    @staticmethod
    def _find_steps(sess: dict) -> list[dict]:
        for k in _STEP_KEYS:
            v = sess.get(k)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        return []

    def parse(self, source: SourceFile) -> Iterator[Event]:
        if source.path.name == "events.jsonl":
            yield from self._parse_cli_session(source.path)
            return
        try:
            doc = json.loads(source.path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            return

        for si, sess in enumerate(self._find_sessions(doc)):
            sid = str(
                sess.get("id") or sess.get("sessionId") or sess.get("uuid") or f"junie-{si}"
            )
            cwd = sess.get("projectPath") or sess.get("project") or sess.get("cwd")
            title = sess.get("title") or sess.get("name") or sess.get("task")
            repo_id = hash_repo(cwd)
            seq = 0

            base = dict(
                session_id=sid,
                provider=self.provider,
                cwd=cwd,
                repo_id=repo_id,
                repo_label=repo_label_of(cwd),
            )

            if title:
                seq += 1
                yield Event(
                    seq=seq,
                    role=Role.SYSTEM,
                    ts=str(sess.get("createdAt") or sess.get("timestamp") or ""),
                    text=str(title),
                    raw_id="title",
                    **base,
                )

            for step in self._find_steps(sess):
                ts = str(step.get("timestamp") or step.get("createdAt") or step.get("time") or "")
                model = step.get("model") or step.get("modelId") or sess.get("model")
                role_raw = str(step.get("role") or step.get("type") or "").lower()
                sbase = dict(ts=ts, raw_id=str(step.get("id") or ""), **base)

                u = step.get("usage") or step.get("tokens") or {}
                usage = None
                if isinstance(u, dict) and u:
                    usage = Usage(
                        input=int(u.get("input") or u.get("inputTokens") or u.get("prompt") or 0),
                        output=int(
                            u.get("output") or u.get("outputTokens") or u.get("completion") or 0
                        ),
                        cache_read=int(u.get("cacheRead") or u.get("cachedTokens") or 0),
                        cache_write=int(u.get("cacheWrite") or 0),
                    )

                if "user" in role_raw or "prompt" in role_raw or "request" in role_raw:
                    text = self.text_of(step.get("content") or step.get("text") or step.get("prompt"))
                    if text.strip():
                        seq += 1
                        yield Event(seq=seq, role=Role.USER, text=text, **sbase)
                    continue

                if "tool" in role_raw or step.get("toolName"):
                    raw_name = str(step.get("toolName") or step.get("tool") or "")
                    command = step.get("command") if isinstance(step.get("command"), str) else None
                    seq += 1
                    yield Event(
                        seq=seq,
                        role=Role.TOOL_CALL,
                        model=model,
                        text=command,
                        tool=ToolInfo(
                            kind=normalize_tool(raw_name, command),
                            raw_name=raw_name,
                            input_bytes=self.size_of(step.get("input")),
                            output_bytes=self.size_of(step.get("output")),
                            target_path=step.get("path") or step.get("file"),
                        ),
                        **sbase,
                    )
                    continue

                if "compact" in role_raw:
                    seq += 1
                    yield Event(seq=seq, role=Role.COMPACTION, **sbase)
                    continue

                text = self.text_of(step.get("content") or step.get("text") or step.get("response"))
                seq += 1
                yield Event(
                    seq=seq,
                    role=Role.ASSISTANT,
                    model=model,
                    usage=usage,
                    cost_usd=cost_of(model, usage),
                    text=text or None,
                    context_tokens=(usage.input + usage.cache_read) if usage else None,
                    context_limit=context_limit_of(model),
                    **sbase,
                )
