"""Loop detector — when the model is stuck, not you.

Every other detector in this package identifies something the *user* did:
a vague prompt, an unpiped command, a re-ask with no new evidence. This one
identifies something the *model* did — the same failure, over and over, without
converging — and says so plainly.

That distinction matters more than it sounds. A tool that blames the user for
everything is a tool people stop believing, because sometimes it genuinely
isn't the prompt: the model has all the information it needs and is still
circling. Being able to say "six attempts on one error is not a prompting
problem" is what makes the *other* findings credible when they do point at the
user.

## Why not just extend `blind_retry`?

`blind_retry` is pairwise and prompt-shaped: two consecutive user turns that
look alike with no new evidence between them. A loop is neither — it's a
*cycle* spanning many turns, identified from **tool failures** rather than
prompt text, and it fires even when every prompt is well written. You can write
three flawless prompts and still be in a loop.

## Detection

Failures are fingerprinted into a **signature**: the error with everything
volatile stripped out (line numbers, addresses, temp paths, timestamps,
durations). Two failures with the same signature are "the same failure" even
though their raw text differs. Three occurrences of one signature in a session
means the model is not making progress.

## The suppression that keeps this honest

Iteration that *ends in success* is just work. If the last occurrence of a
signature is followed by a passing run, the loop converged and is suppressed
below `CONVERGED_REPORT_AT` repeats. Without this the detector fires on normal
debugging and the entire tool reads as noise — the same discipline as
`test_clean_session_produces_no_findings`, and it has its own fixture and test.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ...schema import Event, Role, ToolKind
from .base import Attempt, Detector, Escalation, Evidence, Finding, GlobalDetector

#: A signature must repeat this many times before it's a loop rather than a
#: retry. Two identical failures is a bad afternoon; three is a pattern.
LOOP_THRESHOLD = 3

#: A loop that converged is only worth reporting if it took this many rounds —
#: below it, that's just iteration and flagging it would be crying wolf.
CONVERGED_REPORT_AT = 5

#: Same file edited this many times with failures interleaved.
CHURN_EDIT_THRESHOLD = 4
CHURN_FAILURE_THRESHOLD = 2

_VOLATILE = [
    (re.compile(r"0x[0-9a-fA-F]+"), "0xADDR"),
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "UUID"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\S*"), "TS"),
    (re.compile(r"(?:/[\w.\-]+){2,}"), "PATH"),
    (re.compile(r"[A-Za-z]:\\[\w.\\\-]+"), "PATH"),
    (re.compile(r"\bline \d+\b", re.I), "line N"),
    (re.compile(r":\d+:\d+\b"), ":N:N"),
    (re.compile(r":\d+\b"), ":N"),
    (re.compile(r"\b\d+\.\d+\s*(?:s|ms|sec|seconds?)\b", re.I), "DUR"),
    (re.compile(r"\b\d+\b"), "N"),
    (re.compile(r"\s+"), " "),
]

#: The first line of a traceback is boilerplate; the *last* line carries the
#: exception type and message, which is what actually identifies the failure.
_TRACEBACK_TAIL = re.compile(r"(?:^|\n)([A-Za-z_.]*(?:Error|Exception|Failure)[^\n]*)\s*$")


#: A signature this short after stripping is not specific enough to trust as
#: "the same failure" — it's usually what's left of a generic line ("exit
#: code N", "error: N") once every digit and path has been scrubbed. Matching
#: unrelated failures together under a signature this thin is how a handful
#: of common, boring errors end up looking like one big recurring pattern
#: across every session in the store — see `_MIN_SIGNATURE_ALNUM`.
_MIN_SIGNATURE_ALNUM = 8


def signature(error_text: str | None) -> str | None:
    """Reduce a failure to a comparable shape.

    Strips everything that changes between two occurrences of the *same*
    underlying problem — line numbers, paths, addresses, timings — so that
    "AssertionError: expected 200 got 401 at test_auth.py:42" and the same
    error at :57 fingerprint identically.

    Returns None (rather than an overly generic string) when what's left
    after stripping is too thin to trust as identifying one specific failure.
    Without this floor, boring/common error shapes ("command failed",
    "exit code 1") collapse to near-identical signatures across totally
    unrelated sessions, and every downstream dollar figure derived from "this
    signature recurred" inherits that false match.
    """
    if not error_text or not error_text.strip():
        return None
    text = error_text.strip()

    # Prefer the exception line of a traceback: it's the part that identifies
    # the failure, and the stack above it is noise that varies between runs.
    tail = _TRACEBACK_TAIL.search(text)
    if tail:
        text = tail.group(1)
    else:
        text = "\n".join(text.splitlines()[:3])

    for pattern, repl in _VOLATILE:
        text = pattern.sub(repl, text)
    text = text.strip().lower()[:200]
    if not text:
        return None
    alnum = sum(1 for c in text if c.isalnum())
    if alnum < _MIN_SIGNATURE_ALNUM:
        return None
    return text


def _nearest_prompt(events: list[Event], seq: int) -> str:
    """The user prompt in force when this occurrence happened — the most
    recent one at or before it. Empty string if the occurrence precedes any
    prompt, which the caller renders as an empty excerpt rather than crashing."""
    text = ""
    for e in events:
        if e.seq > seq:
            break
        if e.role is Role.USER and e.text:
            text = e.text
    return text


def _build_attempts(events: list[Event], occurrences: list[Event]) -> list[Attempt]:
    """One `Attempt` per occurrence: the window of cost/tokens/calls between
    the previous occurrence (or session start) and this one.

    First attempt costing less than the finding's total is the sanity check
    this produces almost for free: each later window only ever adds more
    calls on top of an already-growing conversation.
    """
    attempts: list[Attempt] = []
    bounds = [events[0].seq - 1 if events else 0] + [o.seq for o in occurrences]
    for i, occ in enumerate(occurrences):
        lo, hi = bounds[i], bounds[i + 1]
        cost, toks = Detector.cost_between(events, lo, hi)
        calls = sum(
            1 for e in events if lo < e.seq <= hi and e.role is Role.TOOL_CALL
        )
        attempts.append(
            Attempt(
                when=occ.ts,
                prompt_excerpt=Detector.quote(_nearest_prompt(events, occ.seq)),
                tokens=toks,
                cost_usd=round(cost, 6),
                calls=calls,
            )
        )
    return attempts


def _escalation_ladder() -> list[Escalation]:
    """The advice changes with how deep the hole is — see `Escalation`."""
    return [
        Escalation(
            threshold=2,
            action="better_prompt",
            advice=(
                "Paste the exact error, the command that produced it, and what you "
                "expected instead. The model is guessing because it hasn't been shown "
                "the failure."
            ),
        ),
        Escalation(
            threshold=LOOP_THRESHOLD,
            action="plan_mode",
            advice=(
                "Stop patching and ask for a plan before any further edits. After three "
                "attempts the model is fixing symptoms it hasn't diagnosed — more "
                "attempts cost more and converge no faster."
            ),
        ),
        Escalation(
            threshold=CONVERGED_REPORT_AT,
            action="skill",
            advice=(
                "This is a knowledge gap, not a prompting gap. Write the resolution down "
                "as a skill or a guideline (`agentlens skills`) so the next session "
                "starts where this one ended instead of rediscovering it."
            ),
        ),
    ]


class LoopDetector(Detector):
    """Same failure, repeatedly, within one session."""

    name = "loop"
    title = "The model looping on the same failure"
    min_usd = 0.0  # a loop is worth reporting on repetition alone

    def run(self, events: list[Event]) -> Finding | None:
        failures: dict[str, list[Event]] = defaultdict(list)
        for e in events:
            if e.role not in (Role.TOOL_RESULT, Role.TOOL_CALL) or not e.tool:
                continue
            if not e.tool.failed:
                continue
            sig = signature(e.tool.error_text)
            if sig:
                failures[sig].append(e)

        if not failures:
            return None

        wasted = 0.0
        tokens = 0
        evidence: list[Evidence] = []
        worst_repeats = 0
        worst_attempts: list[Attempt] = []

        for sig, occurrences in failures.items():
            repeats = len(occurrences)
            if repeats < LOOP_THRESHOLD:
                continue

            converged = self._converged_after(events, occurrences[-1])
            if converged and repeats < CONVERGED_REPORT_AT:
                # Iteration that ended in success. Normal work — say nothing.
                continue

            first, last = occurrences[0].seq, occurrences[-1].seq
            cost, toks = self.cost_between(events, first - 1, last)
            # The first attempt is legitimate; the repeats are the waste.
            attributable = cost * (repeats - 1) / repeats if repeats else 0.0
            wasted += attributable
            tokens += int(toks * (repeats - 1) / repeats) if repeats else 0

            target = next(
                (o.tool.target_path for o in occurrences if o.tool and o.tool.target_path),
                None,
            )
            if repeats > worst_repeats:
                worst_repeats = repeats
                worst_attempts = _build_attempts(events, occurrences)
            detail = f"Same failure {repeats}× in this session"
            if target:
                detail += f" on {target.rsplit('/', 1)[-1]}"
            detail += (
                ". Resolved eventually, but only after "
                f"{repeats} rounds."
                if converged
                else ". Never resolved in this session."
            )

            evidence.append(
                Evidence(
                    session_id=occurrences[0].session_id,
                    ts=occurrences[0].ts,
                    quote=self.quote(occurrences[0].tool.error_text if occurrences[0].tool else ""),
                    detail=detail,
                    cost_usd=round(attributable, 6),
                    tokens=tokens,
                    full_text=(occurrences[0].tool.error_text if occurrences[0].tool else None),
                    kind="error",
                    target_path=target,
                )
            )

        if not evidence:
            return None

        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "The same error recurred without the model converging on a fix. Each "
                    "attempt re-sends the whole conversation, so the cost compounds while "
                    "the failure stays identical — this is the model not making progress, "
                    "not a badly worded prompt."
                ),
                prescription=(
                    "Stop re-prompting. Give it the exact error and expected output, or "
                    "switch to planning before further edits."
                ),
                escalation=_escalation_ladder(),
                blames_model=True,
                wasted_usd=wasted,
                wasted_tokens=tokens,
                occurrences=worst_repeats,
                evidence=evidence,
                attempts=worst_attempts,
            )
        )

    @staticmethod
    def _converged_after(events: list[Event], last_failure: Event) -> bool:
        """Did anything actually succeed after the final failure?

        A passing test or a clean run on the same target means the loop ended
        in a fix — which is iteration, not pathology.
        """
        target = last_failure.tool.target_path if last_failure.tool else None
        for e in events:
            if e.seq <= last_failure.seq or not e.tool:
                continue
            if e.tool.failed:
                continue
            if e.tool.kind is ToolKind.TEST:
                return True
            if target and e.tool.target_path == target:
                return True
        return False


class SameTargetChurnDetector(Detector):
    """The same file edited over and over with failures in between.

    A weaker, independent signal from `LoopDetector`: it catches churn even
    when the agent's failures don't produce comparable error text (a tool that
    reports failure without a message, for instance).
    """

    name = "target_churn"
    title = "Repeatedly rewriting the same file without it working"
    min_usd = 0.0

    def run(self, events: list[Event]) -> Finding | None:
        edits: dict[str, list[Event]] = defaultdict(list)
        failures = 0
        last_failure: Event | None = None
        for e in events:
            if e.role is Role.TOOL_CALL and e.tool and e.tool.kind is ToolKind.EDIT:
                if e.tool.target_path:
                    edits[e.tool.target_path].append(e)
            if e.tool and e.tool.failed:
                failures += 1
                last_failure = e

        if failures < CHURN_FAILURE_THRESHOLD:
            return None

        # Same suppression as LoopDetector, and for the same reason: a session
        # that ends green is successful debugging, not churn. Without this the
        # detector fires on every productive fix-it session and the user learns
        # to ignore the whole report.
        if last_failure is not None and LoopDetector._converged_after(events, last_failure):
            return None

        evidence: list[Evidence] = []
        wasted = 0.0
        worst = 0
        worst_attempts: list[Attempt] = []
        for path, evs in edits.items():
            if len(evs) < CHURN_EDIT_THRESHOLD:
                continue
            cost, _ = self.cost_between(events, evs[0].seq - 1, evs[-1].seq)
            share = cost * (len(evs) - 1) / len(evs)
            wasted += share
            if len(evs) > worst:
                worst = len(evs)
                worst_attempts = _build_attempts(events, evs)
            evidence.append(
                Evidence(
                    session_id=evs[0].session_id,
                    ts=evs[0].ts,
                    quote=path.rsplit("/", 1)[-1],
                    detail=(
                        f"Rewritten {len(evs)}× with {failures} failure(s) in the same "
                        "session — the fix isn't landing."
                    ),
                    cost_usd=round(share, 6),
                    target_path=path,
                )
            )

        if not evidence:
            return None

        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "Repeated rewrites of one file alongside repeated failures means each "
                    "attempt is a guess. The conversation grows with every round, so the "
                    "last guess costs several times what the first did."
                ),
                prescription=(
                    "Ask for a diagnosis before another edit: what does the model believe "
                    "is wrong, and what evidence supports it?"
                ),
                escalation=_escalation_ladder(),
                blames_model=True,
                wasted_usd=wasted,
                wasted_tokens=0,
                occurrences=worst,
                evidence=evidence,
                attempts=worst_attempts,
            )
        )


class CrossSessionLoopDetector(GlobalDetector):
    """The same failure in more than one session — the strongest signal that
    something should be written down.

    A per-session detector cannot see this by construction: it is handed one
    session at a time. Hitting the same wall in two different sessions is proof
    the knowledge didn't persist, and that is precisely the case where "write a
    skill" is the right advice rather than "write a better prompt".
    """

    name = "cross_session_loop"
    title = "The same failure keeps coming back across sessions"
    min_usd = 0.0

    def run_global(self, sessions: list[list[Event]]) -> Finding | None:
        by_sig: dict[str, list[tuple[str, Event]]] = defaultdict(list)
        # Cost of every session in which a signature reappeared, so the finding
        # can carry a dollar figure like every other finding does. Without one
        # it isn't just less persuasive — it gets filtered out of the report
        # entirely, since low-value findings are suppressed by design.
        session_cost: dict[str, float] = {}
        for events in sessions:
            if not events:
                continue
            sid = events[0].session_id
            session_cost[sid] = sum(e.cost_usd for e in events)
            seen_here: set[str] = set()
            for e in events:
                if not e.tool or not e.tool.failed:
                    continue
                sig = signature(e.tool.error_text)
                if not sig or sig in seen_here:
                    continue
                seen_here.add(sig)
                by_sig[sig].append((e.session_id, e))

        # A session's cost is attributed to cross-session waste **at most
        # once**, no matter how many different recurring signatures it
        # happens to also carry. Without this, a handful of overlapping
        # signatures that each touch mostly the same sessions each re-add
        # that session's full cost — the finding's total can balloon to
        # several multiples of what was actually spent. `wasted` is derived
        # entirely from `charged_sessions`'s cost, which makes "total waste
        # exceeds total spend" structurally impossible rather than merely
        # unlikely.
        charged_sessions: set[str] = set()
        evidence: list[Evidence] = []
        worst = 0
        best_attempts: list[Attempt] = []
        for sig, hits in sorted(by_sig.items(), key=lambda kv: len({s for s, _ in kv[1]}), reverse=True):
            sessions_hit = {sid for sid, _ in hits}
            if len(sessions_hit) < 2:
                continue
            worst = max(worst, len(sessions_hit))
            # Everything after the first session is re-solving something
            # already solved. Attribute those sessions' spend as reclaimable —
            # but only the part not already charged to a different signature.
            repeat_sessions = sorted(sessions_hit)[1:]
            newly_charged = [sid for sid in repeat_sessions if sid not in charged_sessions]
            attributable = sum(session_cost.get(sid, 0.0) for sid in newly_charged)
            charged_sessions.update(newly_charged)
            _, sample = hits[0]
            if attributable > 0:
                cost_note = f"so you paid roughly ${attributable:.2f} to rediscover it"
            else:
                # Every repeat session here was already charged to a bigger/
                # earlier-processed signature that hit the same sessions —
                # still a real recurring pattern worth surfacing, just with
                # nothing new to add to the dollar total (avoids double
                # billing the same session's cost to two different findings).
                cost_note = "already counted above via an overlapping session"
            evidence.append(
                Evidence(
                    session_id=sample.session_id,
                    ts=sample.ts,
                    quote=self.quote(sample.tool.error_text if sample.tool else ""),
                    detail=(
                        f"Hit in {len(sessions_hit)} separate sessions — the fix from last "
                        f"time didn't carry forward, {cost_note}."
                    ),
                    cost_usd=round(attributable, 6),
                    full_text=sample.tool.error_text if sample.tool else None,
                    kind="error",
                    target_path=sample.tool.target_path if sample.tool else None,
                )
            )
            if len(sessions_hit) >= worst:
                # One "attempt" per session that hit this signature — a
                # coarser grain than the per-session detectors above (there is
                # no single event stream spanning sessions to window against),
                # but still one entry per occurrence, in order.
                best_attempts = [
                    Attempt(
                        when=e.ts,
                        prompt_excerpt=self.quote(e.tool.error_text if e.tool else ""),
                        tokens=0,
                        cost_usd=round(session_cost.get(sid, 0.0), 6),
                        calls=1,
                    )
                    for sid, e in sorted(hits, key=lambda h: h[1].ts)
                ]

        if not evidence:
            return None

        # Sum of costs actually charged — by construction, at most the total
        # cost of the sessions involved, never a multiple of it.
        wasted = sum(session_cost.get(sid, 0.0) for sid in charged_sessions)

        return Finding(
            detector=self.name,
            title=self.title,
            mechanism=(
                "You solved this before and paid to solve it again. Nothing in the "
                "project tells the agent what the answer was, so every new session "
                "rediscovers it from scratch."
            ),
            prescription=(
                "Write the resolution into the project — a skill, or a rule in "
                "CLAUDE.md / AGENTS.md / .junie/guidelines.md. Run "
                "`agentlens skills` for a generated starting point."
            ),
            escalation=[
                Escalation(
                    threshold=2,
                    action="skill",
                    advice=(
                        "Capture this as a skill or guideline. It has already cost you "
                        "more than once, and it will keep doing so until it's written "
                        "down where the agent will read it."
                    ),
                )
            ],
            blames_model=False,  # this one really is a missing-documentation problem
            severity="high",
            wasted_usd=wasted,
            occurrences=worst,
            evidence=sorted(evidence, key=lambda e: -e.cost_usd)[:5],
            attempts=best_attempts,
        )

    # `Evidence.quote` lives on Detector; GlobalDetector needs its own copy.
    @staticmethod
    def quote(text: str | None, limit: int = 160) -> str:
        if not text:
            return ""
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"
