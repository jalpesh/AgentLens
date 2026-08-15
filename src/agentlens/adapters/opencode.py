"""OpenCode adapter.

Source: ``~/.local/share/opencode/storage/`` (XDG data dir; overridable via
``OPENCODE_HOME``) — one JSON file per message under
``storage/message/<sessionID>/msg_*.json``, and one per session under
``storage/session/<projectHash>/<sessionID>.json``.

This is the best-documented of the Phase 2 formats — a plain JSON-per-file
layout rather than a database — so it ships as full support rather than
experimental. OpenCode itself stores ``cost: 0`` on message records and
expects the reader to price tokens independently, which is what this adapter
does via the shared pricing table.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

from ..schema import (
    ERROR_TEXT_LIMIT,
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

_ROLE_MAP = {"user": Role.USER, "assistant": Role.ASSISTANT, "system": Role.SYSTEM}


class OpenCodeAdapter(Adapter):
    provider = Provider.OPENCODE
    display_name = "OpenCode"
    schema_version = "2"

    def roots(self) -> list[Path]:
        env = os.environ.get("OPENCODE_HOME")
        if env:
            return [Path(env) / "storage"]
        xdg = os.environ.get("XDG_DATA_HOME")
        base = Path(xdg) if xdg else Path.home() / ".local" / "share"
        return [base / "opencode" / "storage"]

    def discover(self) -> list[SourceFile]:
        out: list[SourceFile] = []
        for root in self.roots():
            msg_dir = root / "message"
            if not msg_dir.exists():
                continue
            for p in self.safe_glob(msg_dir, "*.json", self):
                if p.stat().st_size > 0:
                    out.append(SourceFile.of(p))
        return sorted(out, key=lambda s: s.mtime)

    def parse(self, source: SourceFile) -> Iterator[Event]:
        rec = self.read_json(source.path)
        if not isinstance(rec, dict):
            return

        # storage/message/<sessionID>/msg_<id>.json
        session_id = source.path.parent.name or self.dig(rec, "sessionID", "session_id") or "unknown"
        role_raw = str(self.dig(rec, "role") or "").lower()
        role = _ROLE_MAP.get(role_raw, Role.SYSTEM)
        ts = str(self.dig(rec, "time", "created", "createdAt") or "")
        # OpenCode namespaces model under providerID/modelID rather than a
        # single string; fall back to a flat `model` field if present.
        model = self.dig(rec, "model") or self.dig(rec, "modelID")
        if isinstance(model, dict):
            model = model.get("name") or model.get("id")
        cwd = self.dig(rec, "cwd", "directory", "path")
        repo_id = hash_repo(cwd)
        raw_id = str(self.dig(rec, "id", "messageID") or source.path.stem)

        # One file per MESSAGE, not per session — see `seq_from_ts` docstring
        # for why a local counter would silently corrupt ordering the moment
        # ingest runs incrementally across two separate CLI invocations.
        base = dict(
            session_id=str(session_id), provider=self.provider, ts=ts,
            cwd=cwd, repo_id=repo_id, repo_label=repo_label_of(cwd),
            raw_id=raw_id,
        )

        text = self.text_of(
            self.dig(rec, "content", "text", "parts") or rec.get("parts")
        )

        usage_raw = self.dig(rec, "tokens", "usage")
        usage = None
        if isinstance(usage_raw, dict):
            usage = Usage(
                input=int(self.dig(usage_raw, "input") or 0),
                output=int(self.dig(usage_raw, "output") or 0),
                cache_read=int(self.dig(usage_raw, "cache", "read") or 0)
                if not isinstance(usage_raw.get("cache"), dict)
                else int(usage_raw["cache"].get("read") or 0),
                cache_write=int(usage_raw.get("cache", {}).get("write") or 0)
                if isinstance(usage_raw.get("cache"), dict)
                else 0,
                reasoning=int(self.dig(usage_raw, "reasoning") or 0),
            )

        local_index = 0
        if role is Role.ASSISTANT:
            yield Event(
                role=role, model=model, usage=usage,
                cost_usd=cost_of(model, usage) if usage else float(rec.get("cost") or 0.0),
                text=text or None,
                context_tokens=(usage.input + usage.cache_read) if usage else None,
                context_limit=context_limit_of(model),
                seq=self.seq_from_ts(ts, local_index),
                **base,
            )
            local_index += 1
        elif text.strip():
            yield Event(role=Role.USER, text=text, seq=self.seq_from_ts(ts, local_index), **base)
            local_index += 1

        # Tool invocations, when present, ride alongside the message as parts
        # with `type == "tool"` (OpenCode's own convention for tool-use parts).
        parts = rec.get("parts")
        if isinstance(parts, list):
            for part in parts:
                if not isinstance(part, dict) or part.get("type") != "tool":
                    continue
                raw_name = str(self.dig(part, "tool", "name") or "")
                state = part.get("state") or {}
                # Failures only — see ToolInfo.error_text.
                status = str(state.get("status") or "").lower()
                failed = status in ("error", "failed") or bool(state.get("error"))
                err_text = (
                    (self.text_of(state.get("error")) or self.text_of(state.get("output")))[
                        :ERROR_TEXT_LIMIT
                    ]
                    or None
                    if failed
                    else None
                )
                yield Event(
                    role=Role.TOOL_CALL, model=model,
                    tool=ToolInfo(
                        kind=normalize_tool(raw_name),
                        raw_name=raw_name,
                        input_bytes=self.size_of(state.get("input")),
                        output_bytes=self.size_of(state.get("output")),
                        exit_code=1 if failed else 0,
                        error_text=err_text,
                        target_path=self.dig(state.get("input") or {}, "filePath", "path"),
                    ),
                    seq=self.seq_from_ts(ts, local_index),
                    **base,
                )
                local_index += 1
