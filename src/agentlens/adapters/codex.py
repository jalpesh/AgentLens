"""Codex CLI adapter.

Source: ``$CODEX_HOME`` (default ``~/.codex``) → ``sessions/``,
``archived_sessions/``, and older flat rollout directories.

Codex writes JSONL "rollouts". The format is a moving target — even ccusage
flags Codex support as experimental "while the Codex CLI log format continues to
evolve" — so this adapter is written defensively: it tolerates several known
envelope shapes and reports **cumulative** token counters as per-turn deltas.

That delta handling is the part people get wrong. Codex reports running totals
for a session; naively summing them inflates cost by roughly the number of turns
squared.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

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


def _dig(obj: Any, *names: str) -> Any:
    """First non-None hit for any of `names`, searched one level deep."""
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


class CodexAdapter(Adapter):
    provider = Provider.CODEX
    display_name = "Codex CLI"
    schema_version = "2"

    def roots(self) -> list[Path]:
        home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
        return [home / "sessions", home / "archived_sessions", home]

    def discover(self) -> list[SourceFile]:
        out: dict[Path, SourceFile] = {}
        for root in self.roots():
            if not root.exists():
                continue
            for p in root.rglob("*.jsonl"):
                if p.is_file() and p.stat().st_size > 0 and p not in out:
                    out[p] = SourceFile.of(p)
        return sorted(out.values(), key=lambda s: s.mtime)

    def parse(self, source: SourceFile) -> Iterator[Event]:
        session_id = source.path.stem
        seq = 0
        cwd: str | None = None
        model: str | None = None
        # Codex reports cumulative counters; we emit deltas.
        prev = Usage()

        for rec in self.iter_jsonl(source.path):
            ts = rec.get("timestamp") or rec.get("ts") or ""
            payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else rec
            rtype = str(
                rec.get("type") or payload.get("type") or payload.get("role") or ""
            ).lower()

            cwd = _dig(payload, "cwd", "workdir", "cwd_path") or cwd
            model = _dig(payload, "model", "model_slug", "model_name") or model
            sid = _dig(payload, "session_id", "id", "conversation_id") or session_id
            repo_id = hash_repo(cwd)

            base = dict(
                session_id=str(sid),
                provider=self.provider,
                ts=str(ts),
                cwd=cwd,
                repo_id=repo_id,
                repo_label=repo_label_of(cwd),
                raw_id=str(rec.get("id") or payload.get("id") or ""),
            )

            # --- token accounting -------------------------------------
            info = _dig(payload, "token_usage", "usage", "total_token_usage", "info")
            if isinstance(info, dict):
                inner = info.get("total_token_usage") or info.get("last_token_usage") or info
                if isinstance(inner, dict):
                    cum = Usage(
                        input=int(inner.get("input_tokens") or inner.get("input") or 0),
                        output=int(inner.get("output_tokens") or inner.get("output") or 0),
                        cache_read=int(
                            inner.get("cached_input_tokens")
                            or inner.get("cache_read_input_tokens")
                            or 0
                        ),
                        reasoning=int(inner.get("reasoning_output_tokens") or 0),
                    )
                    # Detect whether these are cumulative or per-turn.
                    is_cumulative = cum.total >= prev.total and prev.total > 0
                    delta = (
                        Usage(
                            input=max(0, cum.input - prev.input),
                            output=max(0, cum.output - prev.output),
                            cache_read=max(0, cum.cache_read - prev.cache_read),
                            reasoning=max(0, cum.reasoning - prev.reasoning),
                        )
                        if is_cumulative
                        else cum
                    )
                    if cum.total >= prev.total:
                        prev = cum

                    if delta.total > 0:
                        seq += 1
                        yield Event(
                            seq=seq,
                            role=Role.ASSISTANT,
                            model=model,
                            usage=delta,
                            cost_usd=cost_of(model, delta),
                            context_tokens=cum.input + cum.cache_read or None,
                            context_limit=context_limit_of(model),
                            **base,
                        )
                        continue

            # --- messages ----------------------------------------------
            if rtype in ("user_message", "user", "message") and (
                payload.get("role") in (None, "user") or rtype == "user_message"
            ):
                text = self.text_of(payload.get("content") or payload.get("message"))
                if text.strip():
                    seq += 1
                    yield Event(seq=seq, role=Role.USER, text=text, **base)
                continue

            if rtype in ("agent_message", "assistant"):
                text = self.text_of(payload.get("content") or payload.get("message"))
                if text.strip():
                    seq += 1
                    yield Event(
                        seq=seq,
                        role=Role.ASSISTANT,
                        model=model,
                        text=text,
                        context_limit=context_limit_of(model),
                        **base,
                    )
                continue

            # --- tools --------------------------------------------------
            if "function_call" in rtype or "tool_call" in rtype or rtype == "exec_command_begin":
                raw_name = str(
                    _dig(payload, "name", "tool_name", "function") or "exec_command"
                )
                args = payload.get("arguments") or payload.get("input") or payload.get("command")
                command = args if isinstance(args, str) else None
                if isinstance(args, list):
                    command = " ".join(str(a) for a in args)
                seq += 1
                yield Event(
                    seq=seq,
                    role=Role.TOOL_CALL,
                    model=model,
                    text=command,
                    tool=ToolInfo(
                        kind=normalize_tool(raw_name, command),
                        raw_name=raw_name,
                        input_bytes=self.size_of(args),
                        target_path=_dig(payload, "path", "file_path"),
                    ),
                    **base,
                )
                continue

            if "output" in rtype and ("function" in rtype or "exec" in rtype or "tool" in rtype):
                out = payload.get("output") or payload.get("stdout") or payload.get("result")
                exit_code = payload.get("exit_code")
                stderr = payload.get("stderr")
                # Failures only — see ToolInfo.error_text.
                err_text = None
                if exit_code not in (None, 0) or stderr:
                    err_text = (
                        self.text_of(stderr) or self.text_of(out)
                    )[:ERROR_TEXT_LIMIT] or None
                seq += 1
                yield Event(
                    seq=seq,
                    role=Role.TOOL_RESULT,
                    tool=ToolInfo(
                        kind=normalize_tool(str(_dig(payload, "name", "tool_name") or "")),
                        raw_name=str(_dig(payload, "name", "tool_name") or ""),
                        output_bytes=self.size_of(out),
                        error_text=err_text,
                        exit_code=exit_code,
                        duration_ms=payload.get("duration_ms"),
                    ),
                    **base,
                )
                continue

            if "compact" in rtype:
                seq += 1
                yield Event(seq=seq, role=Role.COMPACTION, **base)
