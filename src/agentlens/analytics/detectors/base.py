"""Waste detector framework.

**This is the product.** Cost charts are table stakes that a dozen tools already
ship. What nobody ships is: *here is the exact thing you typed, here is what it
cost you, here is what to type instead.*

Three rules every detector must obey, because breaking any one of them turns a
finding into ignorable noise:

1. **Cite the user's own words.** Generic advice is worthless. A verbatim quote
   of a prompt they wrote on a specific date is unforgettable.
2. **Attach a dollar figure.** "$0.69 over 20 calls and 358k tokens" is why
   somebody screenshots this. "You may be using tokens inefficiently" is not.
3. **Give one concrete rewrite.** Not a principle — the actual sentence to type.

A detector that fires on everything is decoration. Thresholds are tuned to stay
quiet unless there is real money on the table.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any

from ...schema import Event


@dataclass(slots=True)
class Evidence:
    """One concrete instance, quoted from the user's own history."""

    session_id: str
    ts: str
    quote: str  # verbatim, truncated for display
    detail: str = ""
    cost_usd: float = 0.0
    tokens: int = 0
    #: The untruncated text behind `quote`.
    #:
    #: `quote` is ellipsised at 160 chars for display, which is right for a
    #: card and useless for anything else — you cannot reload a truncated
    #: prompt into the Prompt Lab and score it. Kept separate rather than
    #: un-truncating `quote`, so no renderer accidentally dumps a 4,000-word
    #: prompt into a table cell.
    full_text: str | None = None
    #: What `full_text` actually is: "prompt" (something the user typed),
    #: "error" (tool output) or "metric" (a computed summary line).
    #:
    #: Without this the UI cannot tell them apart, and offers to "open in
    #: Prompt Lab" on a stack trace — scoring a traceback as if it were a
    #: prompt is meaningless, and offering it makes the feature look broken.
    kind: str = "metric"
    #: The absolute path this evidence is about, when the underlying event
    #: named one (a tool call's `target_path`) — metadata AgentLens already
    #: has, not a new read. Distinct from the opt-in disk-read citation
    #: feature: this field only ever carries a path string, never file
    #: content. The dashboard uses it to know *which* file a "cite the
    #: file" drill-down button should ask `/api/citation` to read; the read
    #: itself only happens if the user clicks that button. See SECURITY.md.
    target_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Attempt:
    """One occurrence in a loop/churn finding — the first ask, or one repeat.

    Derived entirely from events already in the store (timestamps, tool
    calls, token/cost windows) — no new data source. This is what lets the
    dashboard show "first attempt cost $0.04, attempt 5 cost $0.31" instead
    of only a total.
    """

    when: str
    prompt_excerpt: str
    tokens: int
    cost_usd: float
    calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Escalation:
    """Advice that changes with severity.

    Every detector before the loop detector had a single static prescription,
    which is correct when the fix is always the same thing. It is wrong for
    loops: failing twice means "give the model the error text", failing six
    times means "stop prompting and write this down as a skill". Collapsing
    those into one sentence makes the advice wrong at both ends of the range.
    """

    threshold: int  # applies when occurrences >= this
    action: str  # better_prompt | plan_mode | skill | scripted_loop
    advice: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Finding:
    detector: str
    title: str
    #: What to do instead. One sentence, imperative, actionable.
    prescription: str
    #: Why it costs money. One sentence of mechanism.
    mechanism: str
    severity: str = "medium"  # low | medium | high
    wasted_usd: float = 0.0
    wasted_tokens: int = 0
    occurrences: int = 0
    evidence: list[Evidence] = field(default_factory=list)
    #: Optional severity-tiered advice. Empty for every detector whose fix
    #: doesn't change with how often it happened, which is most of them.
    escalation: list[Escalation] = field(default_factory=list)
    #: Set by detectors where the finding is not the user's fault. Every other
    #: detector implicitly blames the user; a tool that can distinguish "you
    #: wrote a vague prompt" from "the model failed six times on the same
    #: error" is more trustworthy in both directions.
    blames_model: bool = False
    #: Per-occurrence breakdown for loop-family findings: the first attempt,
    #: then each repeat, in order — first attempt cost < total finding cost
    #: is the sanity check every consumer of this list can rely on. Empty for
    #: every detector that isn't attempt-shaped (most of them).
    attempts: list[Attempt] = field(default_factory=list)

    def advice_for(self, occurrences: int | None = None) -> str:
        """Highest-threshold escalation that applies, falling back to
        `prescription` so detectors without a ladder behave exactly as before."""
        n = self.occurrences if occurrences is None else occurrences
        applicable = [e for e in self.escalation if n >= e.threshold]
        if not applicable:
            return self.prescription
        return max(applicable, key=lambda e: e.threshold).advice

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = [e.to_dict() for e in self.evidence]
        d["escalation"] = [e.to_dict() for e in self.escalation]
        d["attempts"] = [a.to_dict() for a in self.attempts]
        d["advice"] = self.advice_for()
        d["wasted_usd"] = round(self.wasted_usd, 4)
        return d


class GlobalDetector(ABC):
    """A detector that sees every session at once.

    Per-session detectors are structurally incapable of noticing that you hit
    the same wall in March and again in August — `run_all` feeds them one
    session at a time and merges afterwards. That cross-session view is exactly
    what justifies the strongest advice the tool gives ("this keeps coming
    back, write it down as a skill"), so it needs its own hook rather than a
    workaround inside a normal detector.
    """

    name: str = ""
    title: str = ""
    min_usd: float = 0.01

    @abstractmethod
    def run_global(self, sessions: list[list[Event]]) -> Finding | None:
        """Analyse every session together. Return None when there's nothing
        worth saying."""


class Detector(ABC):
    name: str = ""
    title: str = ""
    #: Findings below this dollar impact are suppressed. A detector that cries
    #: wolf over three cents trains the user to close the tab.
    min_usd: float = 0.01

    @abstractmethod
    def run(self, events: list[Event]) -> Finding | None:
        """Analyse one session. Return None when there is nothing worth saying."""

    # --- helpers -------------------------------------------------------

    @staticmethod
    def quote(text: str | None, limit: int = 160) -> str:
        if not text:
            return ""
        flat = " ".join(text.split())
        return flat if len(flat) <= limit else flat[: limit - 1] + "…"

    @staticmethod
    def cost_between(events: list[Event], lo: int, hi: int) -> tuple[float, int]:
        """Dollars and tokens spent in the event window (lo, hi]."""
        cost = 0.0
        toks = 0
        for e in events:
            if lo < e.seq <= hi:
                cost += e.cost_usd
                toks += e.usage.total if e.usage else 0
        return cost, toks

    def emit(self, finding: Finding) -> Finding | None:
        if finding.wasted_usd < self.min_usd and finding.occurrences < 3:
            return None
        return finding


def merge(findings: list[Finding]) -> list[Finding]:
    """Combine per-session findings of the same detector into one ranked view."""
    by_name: dict[str, Finding] = {}
    for f in findings:
        if not f:
            continue
        cur = by_name.get(f.detector)
        if cur is None:
            by_name[f.detector] = Finding(
                detector=f.detector,
                title=f.title,
                prescription=f.prescription,
                mechanism=f.mechanism,
                severity=f.severity,
                wasted_usd=f.wasted_usd,
                wasted_tokens=f.wasted_tokens,
                occurrences=f.occurrences,
                evidence=list(f.evidence),
                escalation=list(f.escalation),
                blames_model=f.blames_model,
                attempts=list(f.attempts),
            )
        else:
            cur.wasted_usd += f.wasted_usd
            cur.wasted_tokens += f.wasted_tokens
            cur.occurrences += f.occurrences
            cur.evidence.extend(f.evidence)
            cur.attempts.extend(f.attempts)
    out = list(by_name.values())
    for f in out:
        # Lead with the most expensive example — that's the one that lands.
        f.evidence.sort(key=lambda e: -e.cost_usd)
        f.evidence = f.evidence[:5]
        # Same cap and reasoning as evidence — a single session's worth of
        # attempts is the useful drill-down, not every repeat across every
        # session merged into one unreadable list.
        f.attempts.sort(key=lambda a: a.when)
        f.attempts = f.attempts[:20]
        if f.wasted_usd >= 1.0 or f.occurrences >= 10:
            f.severity = "high"
        elif f.wasted_usd >= 0.10 or f.occurrences >= 4:
            f.severity = "medium"
        else:
            f.severity = "low"
    return sorted(out, key=lambda f: (-f.wasted_usd, -f.occurrences))
