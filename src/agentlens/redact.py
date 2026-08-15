"""Redaction.

Given how easily a screenshot of a tool like this leaks an employer's repo
names, infra hostnames and client identifiers, redaction is not a checkbox here
— it is applied by default on every path that leaves the machine (export,
share, benchmark output).

The rule: the numbers survive, the identifiers don't.
"""

from __future__ import annotations

import re
from typing import Any

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Secrets first — never let these through, even in "raw" mode.
    (re.compile(r"\b(sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,})\b"),
     "<REDACTED:token>"),
    (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"), "<REDACTED:aws-key>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"),
     "<REDACTED:private-key>"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<REDACTED:email>"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{16,}=*"), "Bearer <REDACTED>"),
    (re.compile(r'(?i)\b(password|passwd|secret|api[_-]?key|token)\s*[=:]\s*\S+'),
     r"\1=<REDACTED>"),
    # Identifiers — the ones that leak who you work for.
    (re.compile(r"\bhttps?://[^\s)\"']+"), "<REDACTED:url>"),
    (re.compile(r"(?:/Users|/home)/[^/\s]+"), "~"),
    (re.compile(r"[A-Z]:\\Users\\[^\\\s]+", re.I), "~"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<REDACTED:ip>"),
]

#: Applied even when redaction is "off". Secrets are never optional.
_ALWAYS = _PATTERNS[:6]


def redact_text(text: str | None, full: bool = True) -> str | None:
    if not text:
        return text
    out = text
    for pat, repl in (_PATTERNS if full else _ALWAYS):
        out = pat.sub(repl, out)
    return out


def redact_path(path: str | None) -> str | None:
    """Keep the leaf so findings stay readable, drop the tree that identifies
    the org."""
    if not path:
        return path
    leaf = re.split(r"[/\\]", path)[-1]
    return f".../{leaf}" if leaf else None


def redact_obj(obj: Any, full: bool = True) -> Any:
    """Recursively redact a JSON-serialisable structure."""
    if isinstance(obj, str):
        return redact_text(obj, full)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("cwd", "target_path", "tool_target"):
                out[k] = redact_path(v) if isinstance(v, str) else v
            elif k in ("repo_id", "event_id", "session_id"):
                out[k] = v  # already hashes / opaque ids
            else:
                out[k] = redact_obj(v, full)
        return out
    if isinstance(obj, list):
        return [redact_obj(v, full) for v in obj]
    return obj
