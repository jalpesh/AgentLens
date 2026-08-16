"""The normalized event schema.

Every supported agent — however it stores its history — reduces to a stream of
``Event`` objects. This module is the contract that makes a new adapter a
~150-line file instead of a rewrite. It has no I/O and no third-party imports
on purpose: everything else in the package depends on it, and it depends on
nothing.

The schema is deliberately lossy. It keeps what drives cost and what reveals
behaviour, and throws away the rest.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Provider(str, Enum):
    CLAUDE_CODE = "claude-code"
    CODEX = "codex"
    JUNIE = "junie"
    GEMINI_CLI = "gemini-cli"
    AIDER = "aider"
    COPILOT_CLI = "copilot-cli"
    OPENCODE = "opencode"
    GOOSE = "goose"
    CURSOR = "cursor"
    WINDSURF = "windsurf"
    #: Not a real adapter — no discover()/parse() implementation exists for
    #: this value. It tags session-level data a person typed into the
    #: dashboard's manual-import form for a tool AgentLens doesn't (yet)
    #: support, so it can be counted in totals/charts while staying visibly
    #: and structurally distinct from anything a real adapter produced (see
    #: `web/server.py`'s manual-session endpoint and the "manual" badge in
    #: the dashboard). Carries no turn-by-turn events, so it must never
    #: surface a detector finding — there is nothing to find a pattern in.
    MANUAL = "manual"


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    SYSTEM = "system"
    COMPACTION = "compaction"


class ToolKind(str, Enum):
    """Normalized tool taxonomy.

    Agents name their tools differently (``Bash`` vs ``shell`` vs
    ``run_command``). Detectors reason over these normalized kinds so a rule
    written once fires across every provider.
    """

    READ = "read"
    EDIT = "edit"
    BASH = "bash"
    SEARCH = "search"
    WEB = "web"
    TEST = "test"
    GIT = "git"
    TASK = "task"
    OTHER = "other"


@dataclass(slots=True)
class Usage:
    """Token accounting for a single assistant turn.

    ``cache_read`` and ``cache_write`` are split out because prompt caching is
    where the money actually is on Anthropic and OpenAI models. Two sessions
    with identical total token counts can differ several-fold in cost purely on
    cache hit rate. Any tool that collapses these into one number is wrong.
    """

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    reasoning: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cache_read + self.cache_write + self.reasoning

    @property
    def billable_input(self) -> int:
        """Input tokens excluding cache reads, which are priced separately."""
        return self.input

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            cache_read=self.cache_read + other.cache_read,
            cache_write=self.cache_write + other.cache_write,
            reasoning=self.reasoning + other.reasoning,
        )


#: Failing tool output is truncated hard before it is stored. Long enough to
#: fingerprint the failure, short enough that the database doesn't balloon and
#: the privacy surface stays small.
ERROR_TEXT_LIMIT = 600


@dataclass(slots=True)
class ToolInfo:
    kind: ToolKind = ToolKind.OTHER
    raw_name: str = ""
    input_bytes: int = 0
    output_bytes: int = 0
    exit_code: int | None = None
    duration_ms: int | None = None
    target_path: str | None = None
    #: Truncated snippet of **failing** tool output only (schema v2+).
    #:
    #: Successful output is still discarded. Loop detection needs to compare
    #: one failure against another; it never needs stdout from a command that
    #: worked. Storing only failures keeps both the database and the amount of
    #: the user's code sitting in a queryable file to a minimum.
    error_text: str | None = None

    @property
    def failed(self) -> bool:
        return bool(self.exit_code) or bool(self.error_text)


@dataclass(slots=True)
class Event:
    """One thing that happened inside an agent session."""

    session_id: str
    provider: Provider
    seq: int
    ts: str  # ISO 8601
    role: Role

    model: str | None = None
    usage: Usage | None = None
    cost_usd: float = 0.0

    text: str | None = None
    tool: ToolInfo | None = None

    context_tokens: int | None = None
    context_limit: int | None = None

    cwd: str | None = None
    git_branch: str | None = None
    #: Human-readable project name — the **basename** of ``cwd``, never a full
    #: path (schema v2+).
    #:
    #: This is a deliberate, scoped relaxation of the "never store paths" rule,
    #: and it only holds together because both halves are implemented: a
    #: basename ("checkout-service") is what makes a project filter usable at
    #: all, while a full path ("/Users/me/work/AcmeCorp/checkout-service")
    #: would leak the employer, the client and the directory layout. Shown in
    #: the local UI; **stripped by every export path**. `repo_id` remains the
    #: salted-hash join key — this field is for display only.
    repo_label: str | None = None
    repo_id: str | None = None  # salted hash, never a real path

    raw_id: str | None = None  # the source system's own id, for debugging

    # --- identity -------------------------------------------------------

    _event_id: str | None = field(default=None, repr=False, compare=False)

    @property
    def event_id(self) -> str:
        """Stable content hash.

        Ingest is idempotent because of this. You will re-scan the same JSONL
        hundreds of times while developing; without a stable id you double-count
        and stop trusting your own numbers, which kills the whole project.
        """
        if self._event_id is None:
            basis = "|".join(
                str(x)
                for x in (
                    self.provider.value,
                    self.session_id,
                    self.seq,
                    self.ts,
                    self.role.value,
                    self.raw_id or "",
                )
            )
            self._event_id = hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]
        return self._event_id

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("_event_id", None)
        d["provider"] = self.provider.value
        d["role"] = self.role.value
        if self.tool:
            d["tool"]["kind"] = self.tool.kind.value
        d["event_id"] = self.event_id
        return d


@dataclass(slots=True)
class Session:
    """Rolled-up view of one agent session."""

    session_id: str
    provider: Provider
    title: str | None = None
    started_at: str = ""
    ended_at: str = ""
    total_cost_usd: float = 0.0
    usage: Usage = field(default_factory=Usage)
    event_count: int = 0
    peak_context_pct: float = 0.0
    compaction_count: int = 0
    models: dict[str, dict[str, float]] = field(default_factory=dict)
    repo_id: str | None = None
    repo_label: str | None = None
    cwd: str | None = None


# --- helpers ------------------------------------------------------------

_TOOL_ALIASES: dict[str, ToolKind] = {
    # Claude Code
    "read": ToolKind.READ,
    "write": ToolKind.EDIT,
    "edit": ToolKind.EDIT,
    "multiedit": ToolKind.EDIT,
    "notebookedit": ToolKind.EDIT,
    "bash": ToolKind.BASH,
    "bashoutput": ToolKind.BASH,
    "grep": ToolKind.SEARCH,
    "glob": ToolKind.SEARCH,
    "webfetch": ToolKind.WEB,
    "websearch": ToolKind.WEB,
    "task": ToolKind.TASK,
    "agent": ToolKind.TASK,
    # Codex / generic
    "shell": ToolKind.BASH,
    "local_shell": ToolKind.BASH,
    "run_command": ToolKind.BASH,
    "exec_command": ToolKind.BASH,
    "apply_patch": ToolKind.EDIT,
    "view_image": ToolKind.OTHER,
    "update_plan": ToolKind.OTHER,
    # Junie / JetBrains
    "open": ToolKind.READ,
    "search_project": ToolKind.SEARCH,
    "search_replace": ToolKind.EDIT,
    "create": ToolKind.EDIT,
    "run_test": ToolKind.TEST,
    # Gemini / others
    "read_file": ToolKind.READ,
    "write_file": ToolKind.EDIT,
    "replace": ToolKind.EDIT,
    "google_web_search": ToolKind.WEB,
    "web_fetch": ToolKind.WEB,
}

_TEST_HINTS = (
    "pytest",
    "npm test",
    "npm run test",
    "yarn test",
    "pnpm test",
    "go test",
    "cargo test",
    "mvn test",
    "gradle test",
    "jest",
    "vitest",
    "unittest",
    "tox",
    "rspec",
    "phpunit",
    "dotnet test",
)

_GIT_HINTS = ("git ",)


def normalize_tool(raw_name: str, command_text: str | None = None) -> ToolKind:
    """Map a provider-specific tool name to the shared taxonomy.

    A shell tool is reclassified as TEST or GIT based on what it actually ran,
    because "did you run the tests" is a behavioural question the detectors care
    about far more than "did you invoke bash".
    """
    kind = _TOOL_ALIASES.get(raw_name.strip().lower(), ToolKind.OTHER)
    if kind is ToolKind.BASH and command_text:
        low = command_text.lower()
        if any(h in low for h in _TEST_HINTS):
            return ToolKind.TEST
        if any(low.lstrip().startswith(h) for h in _GIT_HINTS):
            return ToolKind.GIT
    return kind


def hash_repo(path: str | None, salt: str = "agentlens") -> str | None:
    """Turn a filesystem path into an opaque, stable repo identifier.

    Real paths leak employer names, client names and project codenames. We keep
    the ability to group by repo without ever storing what the repo is called.
    """
    if not path:
        return None
    return hashlib.sha256((salt + "|" + path).encode("utf-8")).hexdigest()[:16]


def repo_label_of(path: str | None) -> str | None:
    """The last path component, and nothing else.

    Deliberately paired with `hash_repo` above: that one produces the opaque
    join key, this one produces the human label a project filter needs. Keeping
    them adjacent is the point — if someone later "simplifies" this to return
    the full path, the comment on `Event.repo_label` explains why that would be
    a privacy regression rather than a convenience.
    """
    if not path:
        return None
    cleaned = str(path).replace("\\", "/").rstrip("/")
    if not cleaned:
        return None
    leaf = cleaned.rsplit("/", 1)[-1]
    return leaf or None
