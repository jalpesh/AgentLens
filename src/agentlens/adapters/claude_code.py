"""Claude Code adapter.

Source: ``~/.claude/projects/<encoded-cwd>/<session-id>.jsonl``

One JSON object per line, append-only. Assistant lines carry ``message.usage``
with ``input_tokens`` / ``output_tokens`` / ``cache_creation_input_tokens`` /
``cache_read_input_tokens``. Tool calls appear as ``tool_use`` blocks inside
assistant messages; their results come back as ``tool_result`` blocks on the
following ``type: "user"`` line, matched by ``tool_use_id``.

This is by a distance the richest and best-behaved of the agent formats, which
is why it is the reference implementation.
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


class ClaudeCodeAdapter(Adapter):
    provider = Provider.CLAUDE_CODE
    display_name = "Claude Code"
    schema_version = "2"

    def roots(self) -> list[Path]:
        env = os.environ.get("CLAUDE_CONFIG_DIR")
        candidates = [Path(env) if env else Path.home() / ".claude"]
        return [c / "projects" for c in candidates]

    def discover(self) -> list[SourceFile]:
        out: list[SourceFile] = []
        for root in self.roots():
            if not root.exists():
                continue
            for p in root.rglob("*.jsonl"):
                if p.is_file() and p.stat().st_size > 0:
                    out.append(SourceFile.of(p))
        return sorted(out, key=lambda s: s.mtime)

    def parse(self, source: SourceFile) -> Iterator[Event]:
        session_id = source.path.stem
        seq = 0
        cwd: str | None = None
        branch: str | None = None
        # tool_use_id -> (raw_name, input_bytes, target_path) so the result line
        # can be attributed back to the call that produced it.
        pending: dict[str, tuple[str, int, str | None]] = {}

        for rec in self.iter_jsonl(source.path):
            rtype = rec.get("type")
            cwd = rec.get("cwd") or cwd
            branch = rec.get("gitBranch") or branch
            ts = rec.get("timestamp") or ""
            raw_id = rec.get("uuid")
            repo_id = hash_repo(cwd)

            base = dict(
                session_id=rec.get("sessionId") or session_id,
                provider=self.provider,
                ts=ts,
                cwd=cwd,
                git_branch=branch,
                repo_id=repo_id,
                repo_label=repo_label_of(cwd),
                raw_id=raw_id,
            )

            if rtype == "assistant":
                msg = rec.get("message") or {}
                model = msg.get("model")
                u = msg.get("usage") or {}
                usage = Usage(
                    input=int(u.get("input_tokens") or 0),
                    output=int(u.get("output_tokens") or 0),
                    cache_write=int(u.get("cache_creation_input_tokens") or 0),
                    cache_read=int(u.get("cache_read_input_tokens") or 0),
                )
                text = self.text_of(msg.get("content"))

                # Context occupancy is everything currently resident in the
                # window: fresh input + cached input + what we just emitted.
                ctx = usage.input + usage.cache_read + usage.cache_write + usage.output

                seq += 1
                yield Event(
                    seq=seq,
                    role=Role.ASSISTANT,
                    model=model,
                    usage=usage,
                    cost_usd=cost_of(model, usage),
                    text=text or None,
                    context_tokens=ctx or None,
                    context_limit=context_limit_of(model),
                    **base,
                )

                for block in msg.get("content") or []:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    raw_name = str(block.get("name") or "")
                    tin = block.get("input") or {}
                    in_bytes = self.size_of(tin)
                    target = tin.get("file_path") or tin.get("path") or tin.get("notebook_path")
                    command = tin.get("command") if isinstance(tin.get("command"), str) else None
                    tid = str(block.get("id") or "")
                    if tid:
                        pending[tid] = (raw_name, in_bytes, target)

                    seq += 1
                    yield Event(
                        seq=seq,
                        role=Role.TOOL_CALL,
                        model=model,
                        text=command or None,
                        tool=ToolInfo(
                            kind=normalize_tool(raw_name, command),
                            raw_name=raw_name,
                            input_bytes=in_bytes,
                            target_path=target,
                        ),
                        **base,
                    )

            elif rtype == "user":
                msg = rec.get("message") or {}
                content = msg.get("content")

                # A "user" line is either a real human turn or the carrier for
                # tool results. Telling them apart matters: counting tool
                # results as prompts destroys every behavioural metric.
                results = [
                    b
                    for b in (content or [])
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                ]
                if results:
                    for block in results:
                        tid = str(block.get("tool_use_id") or "")
                        raw_name, in_bytes, target = pending.pop(tid, ("", 0, None))
                        out_bytes = self.size_of(block.get("content"))
                        is_err = bool(block.get("is_error"))
                        # Failures only — see ToolInfo.error_text. A successful
                        # command's stdout is not worth storing and is exactly
                        # the sort of thing users don't expect a local tool to
                        # keep.
                        err_text = (
                            self.text_of(block.get("content"))[:ERROR_TEXT_LIMIT] or None
                            if is_err
                            else None
                        )
                        seq += 1
                        yield Event(
                            seq=seq,
                            role=Role.TOOL_RESULT,
                            tool=ToolInfo(
                                kind=normalize_tool(raw_name),
                                raw_name=raw_name,
                                input_bytes=in_bytes,
                                output_bytes=out_bytes,
                                exit_code=1 if is_err else 0,
                                target_path=target,
                                error_text=err_text,
                            ),
                            **base,
                        )
                    continue

                text = self.text_of(content)
                if not text.strip():
                    continue
                seq += 1
                yield Event(seq=seq, role=Role.USER, text=text, **base)

            elif rtype == "system":
                sub = str(rec.get("subtype") or "")
                content = self.text_of(rec.get("content"))
                looks_compacted = (
                    "compact" in sub.lower()
                    or bool(rec.get("isCompactSummary"))
                    or "conversation was compacted" in content.lower()
                )
                seq += 1
                yield Event(
                    seq=seq,
                    role=Role.COMPACTION if looks_compacted else Role.SYSTEM,
                    text=content or None,
                    **base,
                )

            elif rtype == "summary":
                # Claude Code writes a title record for the session.
                seq += 1
                yield Event(
                    seq=seq,
                    role=Role.SYSTEM,
                    text=str(rec.get("summary") or "") or None,
                    **base,
                )
