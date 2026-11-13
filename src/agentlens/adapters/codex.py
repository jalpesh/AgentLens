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

import json
import os
import re
from pathlib import Path
from typing import Any, Iterator

from ..schema import (
    ERROR_TEXT_LIMIT,
    Event,
    Provider,
    Role,
    ToolInfo,
    ToolKind,
    Usage,
    hash_repo,
    normalize_tool,
    repo_label_of,
)
from ..pricing import context_limit_of, cost_of
from .base import Adapter, SourceFile

try:  # stdlib on Python 3.14+; older Pythons simply skip compressed rollouts
    from compression import zstd as _zstd
except ImportError:  # pragma: no cover
    _zstd = None

_EXIT_RE = re.compile(r"(?:exit code|exited with code)[:\s]+(-?\d+)", re.I)
_PATCH_FILE_RE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File: (.+)$", re.M)
#: Harness-injected user messages: <environment_context>, <command-name>, …
#: and the "# AGENTS.md instructions … <INSTRUCTIONS>" block Codex prepends.
#: Also context-continuation summaries and the auto-approval reviewer's
#: "The following is the Codex agent history …" requests.
_INJECTED = re.compile(r"^\s*(<|# AGENTS\.md instructions|This session is being continued "
                       r"from a previous conversation|The following is the Codex agent history)")
_SHELL_TOOLS = {"exec_command", "shell", "shell_command", "local_shell_call", "container.exec"}


def _kind(raw_name: str, command: str | None) -> ToolKind:
    if raw_name == "apply_patch":
        return ToolKind.EDIT
    if raw_name in _SHELL_TOOLS:
        return normalize_tool("bash", command)
    return normalize_tool(raw_name, command)


def _command_and_target(args: Any) -> tuple[str | None, str | None]:
    """Codex tool arguments: a JSON string ({"cmd": ...} / {"command": [...]}),
    a raw apply_patch body, or already-decoded values."""
    if isinstance(args, str):
        m = _PATCH_FILE_RE.search(args)
        if m:
            return None, m.group(1).strip()
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return args, None
    if isinstance(args, dict):
        cmd = args.get("cmd") or args.get("command")
        target = args.get("path") or args.get("file_path")
        if isinstance(cmd, list):
            # ["bash", "-lc", "pytest -q"] -> the script is what was run
            cmd = " ".join(str(c) for c in cmd)
        return (cmd if isinstance(cmd, str) else None), target
    if isinstance(args, list):
        return " ".join(str(a) for a in args), None
    return None, None


def _exit_code_of(out: Any) -> int | None:
    if isinstance(out, str):
        try:
            decoded = json.loads(out)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            out = decoded
        else:
            m = _EXIT_RE.search(out[:400])
            return int(m.group(1)) if m else None
    if isinstance(out, dict):
        meta = out.get("metadata") or {}
        code = meta.get("exit_code", out.get("exit_code"))
        if isinstance(code, int):
            return code
        return _exit_code_of(out.get("output")) if isinstance(out.get("output"), str) else None
    return None


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
    schema_version = "3"

    def roots(self) -> list[Path]:
        home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
        return [home / "sessions", home / "archived_sessions", home]

    def discover(self) -> list[SourceFile]:
        out: dict[Path, SourceFile] = {}
        patterns = ["*.jsonl"] + (["*.jsonl.zst"] if _zstd else [])
        for root in self.roots():
            if not root.exists():
                continue
            for pattern in patterns:
                for p in root.rglob(pattern):
                    if (p.is_file() and p.stat().st_size > 0 and p not in out
                            and p.name != "session_index.jsonl"):
                        out[p] = SourceFile.of(p)
        return sorted(out.values(), key=lambda s: s.mtime)

    def _records(self, path: Path) -> Iterator[dict]:
        if path.suffix != ".zst":
            yield from self.iter_jsonl(path)
            return
        try:
            with _zstd.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        except (OSError, _zstd.ZstdError):
            return

    def parse(self, source: SourceFile) -> Iterator[Event]:
        # Codex now zstd-compresses finished rollouts in place:
        # `x.jsonl` -> `x.jsonl.zst`. Same session, same id either way.
        session_id = source.path.name.split(".jsonl")[0]
        seq = 0
        cwd: str | None = None
        model: str | None = None
        # Codex reports cumulative counters; we emit deltas.
        prev = Usage()
        # call_id -> (raw_name, command, target) so a result is classified
        # (TEST vs BASH) by the command that produced it.
        pending: dict[str, tuple[str, str | None, str | None]] = {}
        saw_response_messages = False

        for rec in self._records(source.path):
            ts = rec.get("timestamp") or rec.get("ts") or ""
            payload = rec.get("payload") if isinstance(rec.get("payload"), dict) else rec
            outer = str(rec.get("type") or "").lower()
            # `response_item` / `event_msg` are envelopes; the real record
            # type is the payload's.
            if outer in ("response_item", "event_msg") or not outer:
                rtype = str(payload.get("type") or payload.get("role") or "").lower()
            else:
                rtype = outer

            cwd = _dig(payload, "cwd", "workdir", "cwd_path") or cwd
            model = _dig(payload, "model", "model_slug", "model_name") or model
            if outer == "session_meta" and payload.get("id"):
                session_id = str(payload["id"])
            repo_id = hash_repo(cwd)

            base = dict(
                session_id=session_id,
                provider=self.provider,
                ts=str(ts),
                cwd=cwd,
                repo_id=repo_id,
                repo_label=repo_label_of(cwd),
                raw_id=str(rec.get("id") or payload.get("id") or payload.get("call_id") or ""),
            )

            # Per-response usage duplicates what `token_count` already
            # reports cumulatively; counting both double-bills.
            if outer == "token_usage_record":
                continue

            # --- token accounting -------------------------------------
            info = _dig(payload, "token_usage", "usage", "total_token_usage", "info")
            if isinstance(info, dict):
                inner = info.get("total_token_usage") or info.get("last_token_usage") or info
                if isinstance(inner, dict):
                    cached = int(
                        inner.get("cached_input_tokens")
                        or inner.get("cache_read_input_tokens")
                        or 0
                    )
                    cum = Usage(
                        # OpenAI's input_tokens already includes the cached
                        # part; billing both double-counts every cache hit.
                        input=max(0, int(inner.get("input_tokens") or inner.get("input") or 0)
                                  - cached),
                        output=int(inner.get("output_tokens") or inner.get("output") or 0),
                        cache_read=cached,
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
            if rtype == "message":
                role = payload.get("role")
                text = self.text_of(payload.get("content"))
                saw_response_messages = True
                # Harness-injected context (<environment_context>,
                # <command-name>, ...) arrives as a user message; it is not a
                # prompt the person wrote.
                # ponytail: leading-marker check; refine if real prompts start with "<".
                if role == "user" and text.strip() and not _INJECTED.match(text):
                    seq += 1
                    yield Event(seq=seq, role=Role.USER, text=text, **base)
                elif role == "assistant" and text.strip():
                    seq += 1
                    yield Event(seq=seq, role=Role.ASSISTANT, model=model, text=text,
                                context_limit=context_limit_of(model), **base)
                continue

            # Older rollouts carry messages only as event_msg; newer ones carry
            # both, so the event_msg copy is skipped once response items appear.
            if rtype in ("user_message", "user", "agent_message", "assistant"):
                if not saw_response_messages:
                    text = self.text_of(payload.get("content") or payload.get("message"))
                    is_user = rtype in ("user_message", "user")
                    if text.strip() and not (is_user and _INJECTED.match(text)):
                        seq += 1
                        yield Event(
                            seq=seq,
                            role=Role.USER if is_user else Role.ASSISTANT,
                            model=None if is_user else model,
                            text=text,
                            context_limit=None if is_user else context_limit_of(model),
                            **base,
                        )
                continue

            # --- tools --------------------------------------------------
            if rtype in ("function_call", "custom_tool_call", "local_shell_call") or (
                "tool_call" in rtype and not rtype.endswith("output")
            ) or rtype == "exec_command_begin":
                raw_name = str(
                    _dig(payload, "name", "tool_name", "function") or "exec_command"
                )
                args = payload.get("arguments") or payload.get("input") or payload.get("command")
                command, target = _command_and_target(args)
                if payload.get("call_id"):
                    pending[str(payload["call_id"])] = (raw_name, command, target)
                seq += 1
                yield Event(
                    seq=seq,
                    role=Role.TOOL_CALL,
                    model=model,
                    text=command,
                    tool=ToolInfo(
                        kind=_kind(raw_name, command),
                        raw_name=raw_name,
                        input_bytes=self.size_of(args),
                        target_path=target or _dig(payload, "path", "file_path"),
                    ),
                    **base,
                )
                continue

            if rtype.endswith("output") or rtype == "exec_command_end":
                raw_name, command, target = pending.pop(
                    str(payload.get("call_id") or ""),
                    (str(_dig(payload, "name", "tool_name") or ""), None, None),
                )
                out = payload.get("output") or payload.get("stdout") or payload.get("result")
                exit_code = payload.get("exit_code")
                if exit_code is None:
                    exit_code = _exit_code_of(out)
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
                        kind=_kind(raw_name, command),
                        raw_name=raw_name,
                        output_bytes=self.size_of(out),
                        error_text=err_text,
                        exit_code=exit_code,
                        duration_ms=payload.get("duration_ms"),
                        target_path=target,
                    ),
                    **base,
                )
                continue

            if "compact" in rtype:
                seq += 1
                yield Event(seq=seq, role=Role.COMPACTION, **base)
