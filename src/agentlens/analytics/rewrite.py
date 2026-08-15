"""Prompt rewriting.

Two modes, and the distinction is deliberate:

* **`scaffold()` — deterministic, offline, always available.** Not a generic
  fill-in-the-blanks template: it is populated from the user's *own session*.
  The file they were actually editing, the test command their project actually
  runs, the error text that actually followed. A template that already knows
  most of the answers is a different thing from a form.

* **`llm_rewrite()` — opt-in, requires a key, never automatic.** Genuinely
  better output when you want it, at the cost of a network call. It fires only
  when the user passes `--llm`, and its absence changes nothing else.

The default install has no network code at all. See SECURITY.md — that claim is
precise, and this module is the only reason it needs to be.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, asdict
from typing import Any

from ..schema import Event, Role, ToolKind

_TEST_COMMAND_HINTS = (
    "pytest", "npm test", "npm run test", "yarn test", "pnpm test", "go test",
    "cargo test", "mvn test", "gradle test", "jest", "vitest", "tox", "rspec",
    "phpunit", "dotnet test",
)


@dataclass(slots=True)
class Scaffold:
    text: str
    #: Which blanks we managed to fill from real session data, so the UI can
    #: show the user what's inferred versus what they still need to supply.
    filled: dict[str, str]
    placeholders: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nearest_file(events: list[Event], seq: int | None) -> str | None:
    """The file the agent touched closest to this prompt.

    Searches outward from the prompt rather than taking the first file in the
    session — in a long session those are rarely the same thing.
    """
    candidates = [
        e for e in events
        if e.tool and e.tool.target_path and e.role is Role.TOOL_CALL
    ]
    if not candidates:
        return None
    if seq is None:
        return candidates[0].tool.target_path
    nearest = min(candidates, key=lambda e: abs(e.seq - seq))
    return nearest.tool.target_path


def detect_test_command(events: list[Event]) -> str | None:
    """The test command this project actually runs, taken from its own history.

    Only ever reads **tool** events. An earlier version fell back to scanning
    any event's text, which happily matched a user sentence like "I ran
    `pytest tests/test_x.py` and got AssertionError…" and then presented that
    whole sentence as the acceptance command. Prompts describe commands;
    they are not commands.
    """
    for e in events:
        if not e.tool or e.tool.kind is not ToolKind.TEST:
            continue
        cmd = (e.text or "").strip()
        if cmd:
            return cmd.splitlines()[0][:120]

    for e in events:
        # Bash calls that weren't classified as TEST (an unusual runner, say).
        if not e.tool or e.tool.kind is not ToolKind.BASH:
            continue
        text = (e.text or "").strip()
        if any(hint in text.lower() for hint in _TEST_COMMAND_HINTS):
            return text.splitlines()[0][:120]
    return None


def _nearest_error(events: list[Event], seq: int | None) -> str | None:
    fails = [e for e in events if e.tool and e.tool.failed and e.tool.error_text]
    if not fails:
        return None
    if seq is None:
        return fails[0].tool.error_text
    after = [e for e in fails if seq is None or e.seq >= seq]
    chosen = after[0] if after else fails[-1]
    return chosen.tool.error_text


def scaffold(
    prompt: str,
    events: list[Event] | None = None,
    seq: int | None = None,
    issues: list[str] | None = None,
) -> Scaffold:
    """Build a rewrite skeleton, pre-filling everything the session can tell us.

    `issues` are `Issue.code` values from `score_prompt`; they decide which
    sections are worth including. A prompt that already names a file doesn't
    need a FILE line adding to it.
    """
    events = events or []
    issues = issues or []
    filled: dict[str, str] = {}
    placeholders: list[str] = []

    file_hint = _nearest_file(events, seq)
    test_hint = detect_test_command(events)
    error_hint = _nearest_error(events, seq)

    # Keep the user's own words as the intent line — the rewrite is about
    # adding the missing specifics, not about paraphrasing what they meant.
    intent = " ".join((prompt or "").split()) or "<what should change>"

    lines: list[str] = []
    if file_hint:
        filled["file"] = file_hint
        lines.append(f"In @{file_hint}, {intent}")
    else:
        placeholders.append("<FILE>")
        lines.append(f"In @<FILE>, {intent}")

    if "no_acceptance" in issues or not issues:
        if test_hint:
            filled["acceptance"] = test_hint
            lines.append(f"Acceptance: `{test_hint}` must pass.")
        else:
            placeholders.append("<ACCEPTANCE CHECK>")
            lines.append("Acceptance: <ACCEPTANCE CHECK — the command or observable outcome>")

    if error_hint:
        condensed = " ".join(error_hint.split())[:300]
        filled["observed"] = condensed
        lines.append(f"Observed: {condensed}")

    return Scaffold(text="\n".join(lines), filled=filled, placeholders=placeholders)


# --- opt-in LLM path --------------------------------------------------------

_PROVIDERS = {
    "anthropic": ("ANTHROPIC_API_KEY", "https://api.anthropic.com/v1/messages"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1/chat/completions"),
}


def llm_available() -> str | None:
    """Which provider, if any, has a key in the environment. No network call."""
    for name, (env, _) in _PROVIDERS.items():
        if os.environ.get(env):
            return name
    return None


def llm_rewrite(prompt: str, scaffold_text: str, model: str | None = None) -> str:
    """Rewrite via an LLM. Only ever called behind an explicit `--llm` flag.

    Imports its HTTP client inside the function so that the default install
    genuinely has no network code loaded — importing `agentlens` must not pull
    in an HTTP stack for the 99% of runs that never touch this path.
    """
    provider = llm_available()
    if not provider:
        raise RuntimeError(
            "No API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY, or drop "
            "--llm to use the offline scaffold."
        )

    import json
    import urllib.request

    instruction = (
        "Rewrite this coding-agent prompt so it names the target file, states the "
        "change precisely, and gives an acceptance check. Keep the user's intent "
        "exactly. Return only the rewritten prompt.\n\n"
        f"Original:\n{prompt}\n\nDetected context:\n{scaffold_text}"
    )

    if provider == "anthropic":
        key_env, url = _PROVIDERS["anthropic"]
        body = {
            "model": model or "claude-sonnet-4-5",
            "max_tokens": 700,
            "messages": [{"role": "user", "content": instruction}],
        }
        headers = {
            "content-type": "application/json",
            "x-api-key": os.environ[key_env],
            "anthropic-version": "2023-06-01",
        }
    else:
        key_env, url = _PROVIDERS["openai"]
        body = {
            "model": model or "gpt-5-mini",
            "messages": [{"role": "user", "content": instruction}],
        }
        headers = {
            "content-type": "application/json",
            "authorization": f"Bearer {os.environ[key_env]}",
        }

    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read())

    if provider == "anthropic":
        blocks = payload.get("content") or []
        return "\n".join(b.get("text", "") for b in blocks if isinstance(b, dict)).strip()
    choices = payload.get("choices") or []
    return (choices[0]["message"]["content"] if choices else "").strip()


_WHITESPACE = re.compile(r"\n{3,}")


def tidy(text: str) -> str:
    return _WHITESPACE.sub("\n\n", (text or "").strip())
