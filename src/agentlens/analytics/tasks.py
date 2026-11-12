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

from dataclasses import asdict, dataclass, field
from datetime import datetime

from ..schema import Event, Provider, Role, ToolKind
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


def group_tasks(sessions_events: list[list[Event]]) -> list[list[list[Event]]]:
    """Group sessions into tasks: same provider + repo + branch, resumed within
    `RESUME_GAP_MIN`. Sessions with no timestamps stay tasks of their own."""
    def key(evs: list[Event]):
        e = evs[0]
        return (e.provider.value, e.repo_id, e.git_branch)

    ordered = sorted((evs for evs in sessions_events if evs), key=lambda evs: evs[0].ts or "")
    tasks: list[list[list[Event]]] = []
    open_by_key: dict = {}
    for evs in ordered:
        k = key(evs)
        prev = open_by_key.get(k)
        start = _parse(evs[0].ts)
        if prev is not None and k[1] is not None and start:
            last_end = _parse(prev[-1][-1].ts)
            if last_end and 0 <= (start - last_end).total_seconds() <= RESUME_GAP_MIN * 60:
                prev.append(evs)
                continue
        group = [evs]
        tasks.append(group)
        open_by_key[k] = group
    return tasks


def build_task(group: list[list[Event]], looped_sessions: set[str]) -> Task:
    events = [e for evs in group for e in evs]
    sids = [evs[0].session_id for evs in group]
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

    looped = any(s in looped_sessions for s in sids)
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
        ev.session_id
        for f in (findings or []) if f.detector in _LOOP_DETECTORS
        for ev in f.evidence
    }
    return [build_task(g, looped) for g in group_tasks(real)]


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

    grouping = (f"Sessions grouped into tasks by repo + branch, resumed within "
                f"{RESUME_GAP_MIN} min.")
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


def by_outcome(tasks: list[Task]) -> dict[str, int]:
    out = {VALIDATED: 0, UNVALIDATED: 0, FAILING: 0, NO_CHANGE: 0, UNKNOWN: 0}
    for t in tasks:
        out[t.outcome] += 1
    return out
