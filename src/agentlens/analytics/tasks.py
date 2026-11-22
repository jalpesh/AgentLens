"""Task layer + Task Economics — the step from "tokens per prompt" to "cost per
piece of finished work".

A *task* is a unit of engineering work, which no agent records explicitly. It
is inferred here from the normalized `Event` stream, so it works identically
for every adapter (Claude Code, Codex, Junie, Gemini, Copilot, OpenCode, Goose)
and never reads anything provider-specific.

Everything about a task beyond its tokens and cost is inferred, so every
number leaves this module as a `Measure` stating its source and confidence.
Outcomes that cannot be established locally (git rework, abandonment intent)
are reported UNAVAILABLE rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from ..schema import Event, Provider, Role, ToolKind
from .prompt_patterns import _prompts
from .confidence import Confidence, Measure, derived, direct, heuristic, temper, unavailable

#: Sessions in the same repo + branch that start within this many minutes of
#: the previous one ending are treated as one task resumed (a restart, a
#: context reset, `--continue`).
# ponytail: time-gap grouping only; group by work item / commit once SDLC
# correlation exists.
RESUME_GAP_MIN = 30

_LOOP_DETECTORS = {"loop", "target_churn", "cross_session_loop"}

#: Task outcomes. Deliberately no "abandoned": intent to abandon can't be
#: told apart from "finished elsewhere" using local history alone.
VALIDATED = "validated"  # changed code, last known validation passed
UNVALIDATED = "unvalidated"  # changed code, no validation result seen
FAILING = "failing"  # changed code, last known validation failed
NO_CHANGE = "no_change"  # tool calls seen, but no edits: investigation, Q&A
UNKNOWN = "unknown"  # adapter exposed no tool events, so outcome can't be seen

#: Gaps between events longer than this count as idle, not task time.
IDLE_GAP_MIN = 30
#: Inside one session, a prompt after this long a pause starts a new task.
#: Resumed sessions run for weeks; without splitting, one "task" was 70% of
#: all spend and the median task cost $0.
SPLIT_IDLE_MIN = 120

_COMMIT = re.compile(r"\bgit\b[^|;&]*\bcommit\b")
#: First match wins. Heuristic, labelled LOW confidence wherever it's shown.
_TYPES = (
    ("bug fix", re.compile(r"\b(fix|bug|error|fail\w*|broken|crash\w*|regression|"
                           r"not working|doesn'?t work|exception|traceback)\b", re.I)),
    ("refactor", re.compile(r"\b(refactor\w*|clean ?up|rename|simplif\w*|restructur\w*|"
                            r"reorgani[sz]\w*)\b", re.I)),
    # "run the tests" is verification inside other work, not test work.
    ("tests", re.compile(r"\b(write|add|fix|increase|improve|update|missing)\s+(the\s+|more\s+)?"
                         r"(unit\s+|integration\s+|e2e\s+)?tests?\b|\btest coverage\b", re.I)),
    ("docs", re.compile(r"\b(readme|docs?|documentation|document|changelog)\b", re.I)),
    ("feature", re.compile(r"\b(add|implement\w*|build|create|support|new|introduce|make|"
                           r"update|change|improve|enable|label|generate|integrat\w*|"
                           r"upgrade|migrat\w*|deploy|install|set ?up|publish|release)\b", re.I)),
    ("investigation", re.compile(r"\b(check|why|investigat\w*|analy[sz]\w*|review|explain|"
                                 r"look (at|into)|status|what|how|compare|audit|find|research)\b",
                                 re.I)),
)
#: Prompts that carry no topic ("continue", "yes"); skipped for title and type.
_FILLER = re.compile(r"^\W*(continue|go ahead|go on|yes|yep|ok(ay)?|proceed|try again|do it|"
                     r"keep going|resume|next|sure|please continue)\b", re.I)
_TEST_FILE = re.compile(r"(^|/)(tests?/|test_|.*_test\.|.*\.test\.|.*\.spec\.)")


@dataclass(slots=True)
class Task:
    task_id: str
    provider: str
    repo_label: str | None
    session_ids: list[str]
    started_at: str
    ended_at: str
    prompts: int
    model_calls: int
    edits: int
    files_changed: int
    validations: int
    validation_failures: int
    last_validation: str | None  # "passed" | "failed" | "unknown" | None
    tokens_in: int
    tokens_out: int
    cost_usd: float
    looped: bool
    outcome: str
    active_min: float | None
    signals: list[str] = field(default_factory=list)
    title: str = ""
    task_type: str = "other"

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def completed(self) -> bool:
        return self.outcome in (VALIDATED, UNVALIDATED)

    @property
    def first_pass(self) -> bool:
        return self.completed and not self.looped and self.validation_failures == 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(tokens=self.tokens, completed=self.completed, first_pass=self.first_pass)
        return d


def _parse(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return dt.replace(tzinfo=None)


def split_session(evs: list[Event]) -> list[list[Event]]:
    """Cut one session into tasks. A new task starts at a typed prompt that
    follows a pause over `SPLIT_IDLE_MIN`, a `git commit`, or a branch change.
    Cuts only land on prompts, so every task opens with what was asked.
    ponytail: no topic-shift detection; add it if long sessions still blur."""
    prompts = {id(e) for e in _prompts(evs)}
    out: list[list[Event]] = [[]]
    pending = False  # a boundary was seen; cut at the next prompt
    prev_t, prev_branch = None, None
    for e in evs:
        t = _parse(e.ts) if e.ts else None
        if t and prev_t and (t - prev_t).total_seconds() > SPLIT_IDLE_MIN * 60:
            pending = True
        if e.git_branch and prev_branch and e.git_branch != prev_branch:
            pending = True
        if pending and id(e) in prompts and out[-1]:
            out.append([])
            pending = False
        out[-1].append(e)
        if (e.role is Role.TOOL_CALL and e.tool and e.tool.kind is ToolKind.GIT
                and e.text and _COMMIT.search(e.text)):
            pending = True
        prev_t = t or prev_t
        prev_branch = e.git_branch or prev_branch
    return [seg for seg in out if seg]


def classify(asks: list[str], edited: set[str]) -> str:
    """Keyword guess: the first of the task's opening prompts that matches
    decides, then the files it edited break the tie."""
    for text in asks:
        for name, rx in _TYPES:
            # "check X" that then edits code is change work, not investigation.
            if rx.search(text) and not (name == "investigation" and edited):
                return name
    if edited and all(_TEST_FILE.search(p) for p in edited):
        return "tests"
    if edited and all(p.lower().endswith((".md", ".rst", ".txt")) for p in edited):
        return "docs"
    return "investigation" if not edited else "other"


def group_tasks(sessions_events: list[list[Event]]) -> list[list[list[Event]]]:
    """Group sessions into tasks: same provider + repo + branch, resumed within
    `RESUME_GAP_MIN`. Sessions with no timestamps stay tasks of their own."""
    def key(evs: list[Event]):
        e = evs[0]
        return (e.provider.value, e.repo_id, e.git_branch)

    segments = [seg for evs in sessions_events if evs for seg in split_session(evs)]
    ordered = sorted(segments, key=lambda evs: evs[0].ts or "")
    tasks: list[list[list[Event]]] = []
    open_by_key: dict = {}
    for evs in ordered:
        k = key(evs)
        prev = open_by_key.get(k)
        start = _parse(evs[0].ts)
        # Only *different* sessions are joined; a session's own cuts are deliberate.
        if (prev is not None and k[1] is not None and start
                and prev[-1][0].session_id != evs[0].session_id):
            last_end = _parse(prev[-1][-1].ts)
            if last_end and 0 <= (start - last_end).total_seconds() <= RESUME_GAP_MIN * 60:
                prev.append(evs)
                continue
        group = [evs]
        tasks.append(group)
        open_by_key[k] = group
    return tasks


def build_task(group: list[list[Event]], loop_marks: set[tuple[str, str]]) -> Task:
    events = [e for evs in group for e in evs]
    sids = list(dict.fromkeys(evs[0].session_id for evs in group))
    edited: set[str] = set()
    edits = validations = failures = 0
    shell_calls: list[bool] = []
    last_validation: str | None = None
    tin = tout = 0
    for e in events:
        if e.usage:
            tin += e.usage.input + e.usage.cache_read + e.usage.cache_write
            tout += e.usage.output + e.usage.reasoning
        if not e.tool:
            continue
        if e.role is Role.TOOL_CALL and e.tool.kind is ToolKind.EDIT:
            edits += 1
            if e.tool.target_path:
                edited.add(e.tool.target_path)
        is_test = e.tool.kind is ToolKind.TEST
        if e.tool.kind in (ToolKind.BASH, ToolKind.TEST):
            # Older ingests classified a test's *result* as bash (no command
            # text on the result), so pair results with calls by order.
            # ponytail: FIFO pairing; parallel shell calls finishing out of
            # order can mis-pair — re-ingest fixes it at the source.
            if e.role is Role.TOOL_CALL:
                shell_calls.append(is_test)
            elif e.role is Role.TOOL_RESULT and shell_calls:
                is_test = shell_calls.pop(0) or is_test
        if is_test:
            if e.role is Role.TOOL_CALL:
                validations += 1
            known = e.tool.exit_code is not None or bool(e.tool.error_text)
            if known:
                last_validation = "failed" if e.tool.failed else "passed"
                failures += e.tool.failed
            elif last_validation is None:
                last_validation = "unknown"

    # A loop belongs to the task whose time span holds its evidence, not to
    # every task cut from that session.
    lo, hi = events[0].ts or "", events[-1].ts or ""
    looped = any(sid in sids and (not ts or lo <= ts <= hi) for sid, ts in loop_marks)
    signals = []
    if not any(e.tool for e in events):
        outcome = UNKNOWN
        signals.append("agent history has no tool events — outcome not observable")
    elif edits == 0:
        outcome = NO_CHANGE
        signals.append("no file edits")
    elif last_validation == "passed":
        outcome = VALIDATED
        signals.append("last validation run passed")
    elif last_validation == "failed":
        outcome = FAILING
        signals.append("last validation run failed")
    else:
        outcome = UNVALIDATED
        signals.append("code changed, no validation result seen")
    if failures:
        signals.append(f"{failures} failed validation run(s)")
    if looped:
        signals.append("loop detector fired")
    if len(group) > 1:
        signals.append(f"resumed across {len(group)} sessions")

    first = events[0]
    asks = [" ".join(e.text.split()) for e in _prompts(events)]
    topical = [a for a in asks if not _FILLER.match(a) and len(a.split()) >= 3] or asks
    title = topical[0][:100] if topical else ""
    return Task(
        task_id=sids[0],
        provider=first.provider.value,
        repo_label=next((e.repo_label for e in events if e.repo_label), None),
        session_ids=sids,
        started_at=first.ts or "",
        ended_at=events[-1].ts or "",
        prompts=sum(1 for e in events if e.role is Role.USER and e.text),
        model_calls=sum(1 for e in events if e.role is Role.ASSISTANT and e.usage),
        edits=edits,
        files_changed=len(edited),
        validations=validations,
        validation_failures=failures,
        last_validation=last_validation,
        tokens_in=tin,
        tokens_out=tout,
        cost_usd=round(sum(e.cost_usd for e in events), 6),
        looped=looped,
        outcome=outcome,
        active_min=_active_minutes(group),
        signals=signals,
        title=title,
        task_type=classify([a[:400] for a in topical[:3]], edited),
    )


def _active_minutes(group: list[list[Event]]) -> float | None:
    """Sum of gaps between consecutive events, skipping idle gaps. Wall-clock
    span would count a session left open over a weekend as days of work."""
    total, seen = 0.0, False
    for evs in group:
        times = [t for t in (_parse(e.ts) for e in evs) if t]
        for a, b in zip(times, times[1:]):
            gap = (b - a).total_seconds() / 60
            if 0 <= gap <= IDLE_GAP_MIN:
                total += gap
        seen = seen or len(times) > 1
    return round(total, 1) if seen else None


def build_tasks(sessions_events: list[list[Event]], findings: list | None = None) -> list[Task]:
    real = [evs for evs in sessions_events if evs and evs[0].provider is not Provider.MANUAL]
    looped = {
        (ev.session_id, ev.ts or "")
        for f in (findings or []) if f.detector in _LOOP_DETECTORS
        for ev in f.evidence
    }
    tasks = [build_task(g, looped) for g in group_tasks(real)]
    seen: dict[str, int] = {}
    for t in tasks:  # a session's later tasks: <session>#2, #3 …
        seen[t.task_id] = n = seen.get(t.task_id, 0) + 1
        if n > 1:
            t.task_id = f"{t.task_id}#{n}"
    return tasks


def _pct(n: int, d: int) -> float:
    return round(100.0 * n / d, 1)


def task_economics(tasks: list[Task]) -> list[Measure]:
    """Headline Task Economics. Unknown stays UNAVAILABLE, never 0."""
    n = len(tasks)
    if not n:
        return [unavailable("Tasks", "No sessions in scope.", "count")]

    done = [t for t in tasks if t.completed]
    observable = [t for t in tasks if t.outcome != UNKNOWN]
    coded = [t for t in observable if t.outcome != NO_CHANGE]
    validated_any = [t for t in coded if t.last_validation in ("passed", "failed")]
    cost = sum(t.cost_usd for t in tasks)
    tokens = sum(t.tokens for t in tasks)
    obs_cost = sum(t.cost_usd for t in observable)
    obs_tokens = sum(t.tokens for t in observable)
    durations = [t.active_min for t in tasks if t.active_min is not None]
    t_ = lambda c, s: temper(c, s)  # noqa: E731

    grouping = (f"Sessions split at a pause over {SPLIT_IDLE_MIN} min, a git commit or a "
                f"branch change; pieces resumed within {RESUME_GAP_MIN} min are joined.")
    out = [
        heuristic("Tasks", n, grouping, Confidence.MEDIUM, "count", n),
        heuristic("Tasks that changed code", len(coded),
                  f"Tasks with a file-edit tool call. {n - len(observable)} task(s) excluded: "
                  "their agent's history exposes no tool events.",
                  Confidence.HIGH, "count", len(observable)),
        heuristic("Completed tasks", len(done),
                  "Changed code and did not end on a failing validation run.",
                  t_(Confidence.MEDIUM if validated_any else Confidence.LOW, len(coded)),
                  "count", len(coded)),
    ]
    if coded:
        out.append(heuristic(
            "Completion rate", _pct(len(done), len(coded)),
            "Completed ÷ tasks that changed code. Read-only tasks excluded.",
            t_(Confidence.MEDIUM if validated_any else Confidence.LOW, len(coded)),
            "pct", len(coded)))
    else:
        out.append(unavailable("Completion rate", "No observable task changed code.", "pct"))

    out += [
        derived("Sessions / task", round(sum(len(t.session_ids) for t in tasks) / n, 2),
                grouping, t_(Confidence.MEDIUM, n), "", n),
        direct("Prompts / task", round(sum(t.prompts for t in tasks) / n, 2),
               "User prompts counted from session history.", "", n),
        direct("Tokens / task", round(tokens / n),
               "Agent-reported token usage ÷ tasks.", "tokens", n),
        derived("AI cost / task", round(cost / n, 4),
                "Token usage × pricing table ÷ tasks.", t_(Confidence.HIGH, n), "usd", n),
    ]
    if done:
        out += [
            derived("Tokens / completed task", round(obs_tokens / len(done)),
                    "Tokens of all observable tasks (incl. failed, read-only) ÷ completed.",
                    t_(Confidence.MEDIUM, len(done)), "tokens", len(done)),
            derived("AI cost / completed task", round(obs_cost / len(done), 4),
                    "Spend on observable tasks ÷ completed — failed work is part of the "
                    "price. Tasks with unknown outcome excluded.",
                    t_(Confidence.MEDIUM, len(done)), "usd", len(done)),
            heuristic("First-pass success", _pct(sum(t.first_pass for t in done), len(done)),
                      "Completed with no failed validation and no loop finding.",
                      t_(Confidence.MEDIUM if validated_any else Confidence.LOW, len(done)),
                      "pct", len(done)),
        ]
    else:
        out += [unavailable(lbl, "No completed tasks yet.", u) for lbl, u in (
            ("Tokens / completed task", "tokens"), ("AI cost / completed task", "usd"),
            ("First-pass success", "pct"))]
    out.append(heuristic("Loop rate", _pct(sum(t.looped for t in tasks), n),
                         "Tasks where a loop-family detector fired.",
                         t_(Confidence.MEDIUM if observable else Confidence.LOW, n), "pct", n))
    if validated_any:
        out.append(derived(
            "Validation success", _pct(sum(t.last_validation == "passed" for t in validated_any),
                                       len(validated_any)),
            "Code-changing tasks whose last test run passed ÷ those with a test result.",
            t_(Confidence.HIGH, len(validated_any)), "pct", len(validated_any)))
    else:
        out.append(unavailable("Validation success",
                               "No test/build result seen in any code-changing task.", "pct"))
    if durations:
        out.append(heuristic("Active time / task (min)",
                             round(sum(durations) / len(durations), 1),
                             f"Time between events, ignoring gaps over {IDLE_GAP_MIN} min.",
                             t_(Confidence.MEDIUM, len(durations)), "", len(durations)))
    else:
        out.append(unavailable("Active time / task (min)", "No usable timestamps."))
    out.append(unavailable("Rework rate", "Needs git correlation (not yet implemented)."))
    return out


def by_type(tasks: list[Task]) -> list[dict]:
    """Spend per task type, most expensive first. Types are keyword guesses."""
    agg: dict[str, dict] = {}
    for t in tasks:
        a = agg.setdefault(t.task_type, {"task_type": t.task_type, "tasks": 0, "cost_usd": 0.0,
                                         "completed": 0, "tokens": 0})
        a["tasks"] += 1
        a["cost_usd"] += t.cost_usd
        a["completed"] += t.completed
        a["tokens"] += t.tokens
    total = sum(a["cost_usd"] for a in agg.values()) or 1
    for a in agg.values():
        a["cost_usd"] = round(a["cost_usd"], 4)
        a["pct"] = round(100 * a["cost_usd"] / total, 1)
        a["cost_per_task"] = round(a["cost_usd"] / a["tasks"], 4)
    return sorted(agg.values(), key=lambda a: -a["cost_usd"])


def by_outcome(tasks: list[Task]) -> dict[str, int]:
    out = {VALIDATED: 0, UNVALIDATED: 0, FAILING: 0, NO_CHANGE: 0, UNKNOWN: 0}
    for t in tasks:
        out[t.outcome] += 1
    return out
