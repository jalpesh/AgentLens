"""Agents & Skills — suggestions with a receipt.

The hard rule, and the whole reason this module is worth having:

    **No suggestion without a triggering finding or a fingerprint fact.**

"Here are ten good MCP servers" is content anyone can write, has no connection
to the user's data, and is indistinguishable from a blog post. "Add this hook,
because `edit_without_test` cost you $0.52 across 23 edits" can only be
produced by something that has read your history. Every `Suggestion` therefore
carries `triggered_by`, and anything that can't name its trigger doesn't get
generated.

## Delivery

This module *generates* file content. It never writes it. The CLI writes
(`agentlens skills init --write`), where the user is already in a trusted
terminal; the dashboard previews and downloads. The dashboard has no write
endpoint at all — see the README's security note. A browser-reachable
arbitrary-file-write endpoint is not something a privacy-positioned tool should
ship for the convenience of skipping a copy-paste.

## Schemas

Skill, hook and command formats were checked against current Claude Code docs
rather than guessed. Inventing a plausible-looking hook schema produces a file
that silently does nothing, which is worse than producing no file — the user
believes they're protected and isn't.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Any

from .detectors.base import Finding
from .fingerprint import ProjectProfile


@dataclass(slots=True)
class Artifact:
    """A file we can generate, ready to write or download."""

    path: str  # suggested location, relative to the project root
    content: str
    language: str = "markdown"  # for syntax highlighting in the UI

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Suggestion:
    id: str
    kind: str  # skill | hook | command | mcp | guideline
    title: str
    why: str  # cites the finding and its dollar cost
    #: Detector name, "fingerprint:<fact>", or "metric:<fact>" (an
    #: Engineering-tab rollup, e.g. discovery share) — never empty for a
    #: *triggered* suggestion. The one deliberate exception is the generic,
    #: org-wide bucket produced by `generic_suggestions()` below: those are
    #: not derived from the user's data at all, so they leave this "" rather
    #: than inventing a fake trigger to satisfy the type. `suggest()` itself
    #: never returns one with an empty trigger — see the assertion at the
    #: bottom of that function.
    triggered_by: str
    artifact: Artifact | None = None
    #: Set when we're describing a category rather than naming a package,
    #: because we couldn't verify a specific one exists. Better an honest
    #: category than a confident recommendation for something imaginary.
    unverified_category: bool = False
    evidence_usd: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["artifact"] = self.artifact.to_dict() if self.artifact else None
        return d


# --- artifact builders ------------------------------------------------------


def _skill(name: str, description: str, body: str) -> Artifact:
    """A Claude Code / Agent Skills `SKILL.md`.

    Verified against current docs: skills live at
    `.claude/skills/<name>/SKILL.md`; frontmatter sits between `---` markers;
    all fields are optional but `description` is what Claude uses to decide
    when to load the skill.
    """
    return Artifact(
        path=f".claude/skills/{name}/SKILL.md",
        content=f"---\nname: {name}\ndescription: {description}\n---\n\n{body.strip()}\n",
    )


def _command(name: str, description: str, body: str) -> Artifact:
    """A slash command. `.claude/commands/<name>.md` still works and is the
    simpler shape when there are no supporting files."""
    return Artifact(
        path=f".claude/commands/{name}.md",
        content=f"---\ndescription: {description}\n---\n\n{body.strip()}\n",
    )


def _post_edit_hook(test_command: str) -> Artifact:
    """A PostToolUse hook that runs the project's tests after edits.

    Event name, matcher syntax and handler shape verified against current
    hooks documentation: `PostToolUse` with a `matcher` of `"Edit|Write"` and
    a `command`-type handler.
    """
    payload = {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "Edit|Write",
                    "hooks": [
                        {
                            "type": "command",
                            "command": test_command,
                            "timeout": 120,
                            "statusMessage": "Running tests after edit…",
                        }
                    ],
                }
            ]
        }
    }
    return Artifact(
        path=".claude/settings.json",
        # ensure_ascii=False: this file is read by humans, and "\u2026" where
        # an ellipsis should be looks like a bug in the generator.
        content=json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        language="json",
    )


def _mcp_stub(server_name: str, command: str, args: list[str]) -> Artifact:
    return Artifact(
        path=".mcp.json",
        content=json.dumps(
            {"mcpServers": {server_name: {"command": command, "args": args}}},
            indent=2, ensure_ascii=False,
        )
        + "\n",
        language="json",
    )


# --- the engine -------------------------------------------------------------


#: Discovery share above this — search/grep calls as a fraction of all agent
#: tool steps — is the same signal the Engineering tab surfaces as a tile;
#: past this point a code-graph index earns its keep. Chosen conservatively:
#: some discovery is normal, a session that's *mostly* discovery is not.
DISCOVERY_SHARE_TRIGGER_PCT = 30.0


def suggest(
    findings: list[Finding],
    profile: ProjectProfile | None = None,
    metrics: dict | None = None,
) -> list[Suggestion]:
    """Map (findings × project profile × rollup metrics) onto concrete,
    generatable fixes.

    `metrics` is the same Engineering-tab rollup dict `rollups.engineering_metrics`
    produces — optional, and only consulted for the one metric-triggered
    suggestion below (discovery share), so every existing caller that doesn't
    pass it keeps behaving exactly as before.
    """
    profile = profile or ProjectProfile()
    metrics = metrics or {}
    by_name = {f.detector: f for f in findings}
    out: list[Suggestion] = []

    def cost(detector: str) -> float:
        f = by_name.get(detector)
        return f.wasted_usd if f else 0.0

    def cite(detector: str, extra: str = "") -> str:
        f = by_name.get(detector)
        if not f:
            return extra
        return (
            f"{f.title} cost you ${f.wasted_usd:.2f} across {f.occurrences} "
            f"occurrence(s). {extra}".strip()
        )

    # --- edit_without_test → a hook that runs the tests for you -------------
    if "edit_without_test" in by_name and profile.test_command:
        out.append(
            Suggestion(
                id="post-edit-tests",
                kind="hook",
                title="Run your tests automatically after every edit",
                why=cite(
                    "edit_without_test",
                    f"This project runs `{profile.test_command}` — wire it to fire on "
                    "each edit so a broken change surfaces immediately instead of "
                    "three turns later.",
                ),
                triggered_by="edit_without_test",
                artifact=_post_edit_hook(profile.test_command),
                evidence_usd=cost("edit_without_test"),
            )
        )

    # --- loops / blind retries → a /debug command that forces evidence ------
    if {"loop", "blind_retry", "target_churn"} & set(by_name):
        trigger = next(
            d for d in ("loop", "blind_retry", "target_churn") if d in by_name
        )
        out.append(
            Suggestion(
                id="debug-command",
                kind="command",
                title="A /debug command that forces the evidence the model needs",
                why=cite(
                    trigger,
                    "The fix is always the same three facts, so make them a template "
                    "instead of remembering them under pressure.",
                ),
                triggered_by=trigger,
                artifact=_command(
                    "debug",
                    "Structured debugging: state the command, the observed output and "
                    "the expected output before proposing a fix.",
                    """
Before proposing any fix, fill in and confirm all three:

- **Command run:** $1
- **Observed output:** (paste it verbatim — do not summarise)
- **Expected output:** (what should have happened instead)

Then:

1. State your diagnosis in one sentence, and the evidence that supports it.
2. Only after the diagnosis, make the smallest change that tests it.
3. Re-run the command and report the actual result.

If two attempts fail, stop editing and produce a plan instead. Repeated
patching without a diagnosis costs more each round, because the whole
conversation is re-sent every time.
""",
                ),
                evidence_usd=cost(trigger),
            )
        )

    # --- cross-session recurrence → a skill so it stops recurring -----------
    if "cross_session_loop" in by_name:
        f = by_name["cross_session_loop"]
        sample = f.evidence[0].quote if f.evidence else "the recurring failure"
        out.append(
            Suggestion(
                id="recurring-fix-skill",
                kind="skill",
                title="Capture the failure you keep re-solving as a skill",
                why=cite(
                    "cross_session_loop",
                    "It has already come back across sessions — writing it down is the "
                    "only thing that stops it coming back again.",
                ),
                triggered_by="cross_session_loop",
                artifact=_skill(
                    "known-failures",
                    "Known failures in this project and their resolutions. Use when a "
                    "test or build fails with a familiar-looking error.",
                    f"""
## Known failures

Record each recurring failure here the moment you solve it, so the next
session starts from the answer instead of rediscovering it.

### {sample[:120]}

- **Symptom:** (the error, trimmed to its identifying line)
- **Cause:** (what was actually wrong)
- **Fix:** (the change that resolved it)
- **Check:** (the command that proves it's fixed)

> Generated by AgentLens because this failure appeared in more than one
> session. Fill in the cause and fix while you still remember them.
""",
                ),
                evidence_usd=cost("cross_session_loop"),
            )
        )

    # --- uncached docs → cache the answer once ------------------------------
    if "uncached_docs" in by_name:
        out.append(
            Suggestion(
                id="docs-cache-skill",
                kind="skill",
                title="Cache the documentation you keep re-fetching",
                why=cite(
                    "uncached_docs",
                    "Fetched pages arrive as raw markup and are billed in full every "
                    "time, then re-sent as context for the rest of the session.",
                ),
                triggered_by="uncached_docs",
                artifact=_skill(
                    "project-reference",
                    "Cached answers to questions about this project's dependencies and "
                    "APIs. Use before fetching external documentation.",
                    """
## Cached reference

Before fetching a page, check whether the answer is already here. After
fetching one, add the answer here in a few lines.

### <topic>

- **Question:** …
- **Answer:** … (the 5–20 lines that mattered, not the whole page)
- **Source:** <url>

> Generated by AgentLens because the same pages were fetched repeatedly.
""",
                ),
                evidence_usd=cost("uncached_docs"),
            )
        )

    # --- whole-file re-reads → a code index -------------------------------
    if "whole_file_reread" in by_name:
        out.append(
            Suggestion(
                id="code-index-mcp",
                kind="mcp",
                title="Add a semantic code-index MCP server",
                why=cite(
                    "whole_file_reread",
                    "An index returns the relevant ~50 lines instead of pushing whole "
                    "files into context and leaving them there.",
                ),
                triggered_by="whole_file_reread",
                # No specific package is named: AgentLens does not ship
                # recommendations for software it hasn't verified exists and
                # works. The category is real and the config shape is correct;
                # the user chooses the implementation.
                unverified_category=True,
                artifact=_mcp_stub("code-index", "<mcp-server-command>", ["<args>"]),
                evidence_usd=cost("whole_file_reread"),
            )
        )

    # --- fingerprint-driven: linters in the loop ---------------------------
    linters = profile.linters()
    if linters and profile.primary_language:
        out.append(
            Suggestion(
                id="lint-guideline",
                kind="guideline",
                title=f"Put {profile.primary_language} linting in the agent's loop",
                why=(
                    f"Your sessions are mostly {profile.primary_language} "
                    f"({profile.sample_events} events). Agents that lint before "
                    "declaring done produce fewer rework rounds."
                ),
                triggered_by=f"fingerprint:{profile.primary_language}",
                artifact=Artifact(
                    path="AGENTS.md",
                    content=(
                        "## Verification\n\n"
                        "Before declaring a task complete, run:\n\n```bash\n"
                        + "\n".join(linters)
                        + (f"\n{profile.test_command}" if profile.test_command else "")
                        + "\n```\n"
                    ),
                ),
                evidence_usd=0.0,
            )
        )

    # --- compaction → delegate instead of growing one session --------------
    if "compaction_thrash" in by_name:
        out.append(
            Suggestion(
                id="delegation-guideline",
                kind="guideline",
                title="Split long tasks instead of letting one session fill up",
                why=cite("compaction_thrash"),
                triggered_by="compaction_thrash",
                artifact=Artifact(
                    path="AGENTS.md",
                    content=(
                        "## Session hygiene\n\n"
                        "- Start a new session when the topic changes; carry forward "
                        "three lines of state, not the whole transcript.\n"
                        "- For multi-part work, delegate each part as its own task "
                        "rather than growing one conversation until it compacts.\n"
                    ),
                ),
                evidence_usd=cost("compaction_thrash"),
            )
        )

    # --- model routing ------------------------------------------------------
    if "model_mismatch" in by_name:
        out.append(
            Suggestion(
                id="model-routing-guideline",
                kind="guideline",
                title="Write down which model handles which kind of task",
                why=cite("model_mismatch"),
                triggered_by="model_mismatch",
                artifact=Artifact(
                    path="AGENTS.md",
                    content=(
                        "## Model routing\n\n"
                        "- Mechanical work (renames, formatting, boilerplate, "
                        "single-file edits with a clear spec): cheapest capable model.\n"
                        "- Design, debugging and anything needing a plan: frontier "
                        "model.\n"
                    ),
                ),
                evidence_usd=cost("model_mismatch"),
            )
        )

    # --- discovery share → the same code-graph suggestion the Engineering
    # tab links to, triggered on a rollup metric rather than a Finding. -----
    discovery_pct = metrics.get("discovery_share_pct") or 0.0
    if discovery_pct >= DISCOVERY_SHARE_TRIGGER_PCT:
        out.append(
            Suggestion(
                id="discovery-tools",
                kind="mcp",
                title="Give the agent a code index instead of grepping cold",
                why=(
                    f"{discovery_pct:.0f}% of agent tool calls in this scope are "
                    "search/grep — that's the agent re-discovering your codebase "
                    "instead of acting on it, and every one of those calls is "
                    "billed. codegraph builds a local index it can query "
                    "directly; ast-grep gives structural search sharper than "
                    "text grep for the ones you still run by hand."
                ),
                triggered_by="metric:discovery_share",
                artifact=_mcp_stub(
                    "codegraph", "npx", ["-y", "@colbymchenry/codegraph"]
                ),
                evidence_usd=0.0,
            )
        )

    # Every suggestion must be able to name its trigger. This is an assertion
    # about the design, not defensive coding — if it ever fails, something has
    # started inventing generic advice.
    out = [s for s in out if s.triggered_by]
    return sorted(out, key=lambda s: -s.evidence_usd)


def get(suggestions: list[Suggestion], sid: str) -> Suggestion | None:
    return next((s for s in suggestions if s.id == sid), None)


# --- the generic bucket ------------------------------------------------------
#
# Everything above requires a Finding or a fingerprint fact — that's the
# module's whole reason to exist. This section is the deliberate exception,
# reinstated alongside it rather than instead of it:
#
#     "Generic best practices — not derived from your data. Useful for most
#     projects regardless of what we found."
#
# Structurally and visually separate everywhere it renders (own heading, own
# card style in the dashboard, own section in the CLI) so it never gets
# mistaken for a triggered suggestion, and it never sets `triggered_by` —
# doing so would be inventing a receipt this section explicitly doesn't have.
# Every package named here was verified to exist this session (see
# PLAN_PHASE4.md); `code-review-graph` was investigated and found ambiguous
# (four unrelated packages share the name) so it is deliberately left out
# rather than guessed at.


def generic_suggestions() -> list[Suggestion]:
    """Static, org-wide suggestions — same three tools every time, regardless
    of the user's history. Never call `suggest()`'s output filter on this
    list; it exists specifically to have no trigger."""
    return [
        Suggestion(
            id="generic-ast-grep",
            kind="mcp",
            title="ast-grep for structural search and lint/rewrite",
            why=(
                "Structural, AST-aware search and rewrite across a codebase — "
                "finds \"every call to this function with this argument shape\" "
                "in a way text grep can't, and works as both a CLI and an "
                "editor/agent-callable tool."
            ),
            triggered_by="",
            unverified_category=False,
            artifact=Artifact(
                path="README-ast-grep.md",
                content=(
                    "## ast-grep\n\nInstall: `npm install -g @ast-grep/cli` "
                    "(or `brew install ast-grep`).\n\nUse for structural code "
                    "search and codemods: `sg --pattern '$FN($ARGS)' --lang python`.\n"
                ),
            ),
        ),
        Suggestion(
            id="generic-codegraph",
            kind="mcp",
            title="codegraph — a local code-graph MCP server",
            why=(
                "Builds a local, queryable graph of your codebase (symbols, "
                "call sites, imports) that an agent can query directly instead "
                "of re-discovering the same structure by grepping every session."
            ),
            triggered_by="",
            unverified_category=False,
            artifact=_mcp_stub("codegraph", "npx", ["-y", "@colbymchenry/codegraph"]),
        ),
        Suggestion(
            id="generic-codebase-memory",
            kind="mcp",
            title="codebase-memory-mcp — persistent code intelligence",
            why=(
                "An MCP server that keeps a persistent index of the codebase "
                "across sessions, so context an agent already worked out "
                "doesn't have to be rediscovered from scratch next time."
            ),
            triggered_by="",
            unverified_category=False,
            artifact=_mcp_stub(
                "codebase-memory", "npx", ["-y", "codebase-memory-mcp"]
            ),
        ),
    ]
