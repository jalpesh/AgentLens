"""Confidence framework — every derived number knows where it came from.

The dashboard's entire credibility rests on one promise: **a weak estimate is
never dressed up as a hard fact.** A token count read straight from the agent's
own accounting and a "time saved" figure extrapolated from a configured
assumption are both just "a number on a page", and a tool that renders them
identically is lying by omission. The moment that happens, a sceptical
engineering leader is right to distrust *every* number — including the solid
ones. This module is the vocabulary that keeps them distinguishable.

It is adapter-agnostic by construction, and that is the whole point of putting
it here rather than in any one adapter. Provenance is a property of **how a
number was derived, not of which agent produced the events.** A token total is
``DIRECT`` whether it came from Claude Code, Codex, Gemini or a manual import;
a loop cost is ``HEURISTIC`` regardless of source. So one classifier
(:func:`data_quality`) speaks for all seven adapters at once, and a new adapter
inherits the honesty for free.

There is no I/O here and nothing provider-specific — the same deliberate
discipline as ``schema.py``. This is a vocabulary plus one pure classifier over
numbers other modules already computed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class Source(str, Enum):
    """Where a number actually came from. Ordered strongest → weakest."""

    #: Read verbatim from the provider's own records. A token count the agent
    #: itself reported. Nothing was assumed to produce it.
    DIRECT = "direct"
    #: Arithmetic over direct values — a sum, a ratio, a count. As trustworthy
    #: as its inputs and the pricing table it leans on, no more.
    DERIVED = "derived"
    #: A rule-of-thumb detector fired. A behavioural *signal*, deliberately
    #: tuned, but a signal — not a billed line item.
    HEURISTIC = "heuristic"
    #: Depends on an assumption the user set (an hourly engineering value, a
    #: baseline task duration). Only as good as that assumption.
    CONFIGURED = "configured"
    #: Guessed from indirect evidence where nothing direct exists.
    INFERRED = "inferred"
    #: The data needed to compute this is not present. The number is *not*
    #: zero — it is unknown, and the two must never render the same.
    UNAVAILABLE = "unavailable"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    #: Reserved for :attr:`Source.UNAVAILABLE` — there is no estimate to rate.
    NONE = "none"


#: The order used when tempering a rating down by sample size.
_LADDER = (Confidence.HIGH, Confidence.MEDIUM, Confidence.LOW)


def temper(confidence: Confidence, sample_size: int | None, minimum: int = 5) -> Confidence:
    """Drop a rating one rung when the sample is too thin to support it.

    A ratio computed over four tool calls is arithmetically exact and
    statistically meaningless. The arithmetic earns ``HIGH``; the sample size
    takes it back down. ``NONE`` is never tempered — there is nothing to lower.
    """
    if confidence is Confidence.NONE or sample_size is None:
        return confidence
    if sample_size >= minimum:
        return confidence
    try:
        i = _LADDER.index(confidence)
    except ValueError:
        return confidence
    return _LADDER[min(i + 1, len(_LADDER) - 1)]


@dataclass(slots=True)
class Measure:
    """A number plus the full story of how much to trust it.

    Carried alongside the value everywhere a headline metric is surfaced, so no
    renderer can show the figure without also having the provenance in hand.
    ``value is None`` iff ``source is Source.UNAVAILABLE``.
    """

    label: str
    value: Any
    source: Source
    confidence: Confidence
    #: One line of plain English a non-engineer can read: *why* this number is
    #: as trustworthy as it is, and what it leans on.
    method: str
    unit: str = ""  # "usd" | "tokens" | "pct" | "count" | ""
    sample_size: int | None = None

    def __post_init__(self) -> None:
        if self.source is Source.UNAVAILABLE:
            # An unavailable number cannot carry a real estimate or a rating.
            self.value = None
            self.confidence = Confidence.NONE
        elif self.confidence is Confidence.NONE:
            # NONE is reserved for UNAVAILABLE; a present value must own a rating.
            raise ValueError(
                f"{self.label!r}: Confidence.NONE is only valid for an "
                "UNAVAILABLE measure"
            )

    @property
    def available(self) -> bool:
        return self.source is not Source.UNAVAILABLE

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        d["confidence"] = self.confidence.value
        return d


# --- constructors -----------------------------------------------------------
# Thin, so a caller states intent at the call site ("this is heuristic") rather
# than passing enums around, and so the common availability logic lives once.


def direct(label: str, value: Any, method: str, unit: str = "",
           sample_size: int | None = None) -> Measure:
    return Measure(label, value, Source.DIRECT, Confidence.HIGH, method, unit, sample_size)


def derived(label: str, value: Any, method: str, confidence: Confidence = Confidence.HIGH,
            unit: str = "", sample_size: int | None = None) -> Measure:
    return Measure(label, value, Source.DERIVED, confidence, method, unit, sample_size)


def heuristic(label: str, value: Any, method: str, confidence: Confidence = Confidence.MEDIUM,
              unit: str = "", sample_size: int | None = None) -> Measure:
    return Measure(label, value, Source.HEURISTIC, confidence, method, unit, sample_size)


def configured(label: str, value: Any, method: str, confidence: Confidence = Confidence.LOW,
               unit: str = "", sample_size: int | None = None) -> Measure:
    return Measure(label, value, Source.CONFIGURED, confidence, method, unit, sample_size)


def unavailable(label: str, method: str, unit: str = "") -> Measure:
    return Measure(label, None, Source.UNAVAILABLE, Confidence.NONE, method, unit)


# --- the classifier ---------------------------------------------------------


def _priced_token_share(sessions_events: list) -> tuple[float, int]:
    """Share of analysed tokens that ran on a model the pricing table knows.

    This is the honest hinge the spend figure's confidence turns on. An unknown
    model costs ``$0`` everywhere in the report (we never guess a price), so a
    run with heavy usage on an unpriced model has a *real* bill the dashboard
    cannot see — and the spend number must say so rather than imply the agent
    was free. Returns ``(share, total_tokens)``; share is ``1.0`` when there is
    nothing to price, so an empty scope reads as "nothing missing".
    """
    from ..pricing import is_known

    priced = 0
    total = 0
    for evs in sessions_events:
        for e in evs:
            if not e.usage:
                continue
            t = e.usage.total
            total += t
            if e.model and is_known(e.model):
                priced += t
    if total == 0:
        return 1.0, 0
    return priced / total, total


def data_quality(sessions_events: list, findings: list | None = None) -> list[Measure]:
    """Provenance-annotated headline metrics for the whole analysed scope.

    One ``Measure`` per number the dashboard leads with, each carrying where it
    came from and how far to trust it. Deliberately **parallel** to
    :mod:`agentlens.analytics.rollups` — it does not mutate or replace those
    dicts, so every existing consumer and its exact-shape tests are untouched;
    this is the credibility layer laid *over* the numbers, not a rewrite of
    them.

    Classification is by metric nature, so it holds for every adapter without a
    single provider branch.
    """
    from . import rollups

    findings = findings or []
    tot = rollups.totals(sessions_events)
    eng = rollups.engineering_metrics(sessions_events, findings)

    out: list[Measure] = []

    # Tokens and session counts are read straight from the agents' own records.
    out.append(direct(
        "Tokens analysed", tot["tokens"], unit="tokens",
        method="Summed from each turn's usage record as the agent reported it.",
    ))
    out.append(direct(
        "Sessions analysed", tot["sessions"], unit="count",
        method="A direct count of session files parsed from local history.",
    ))

    # Spend is arithmetic over direct token counts and the pricing table — only
    # as sound as that table's coverage, which this makes explicit.
    share, total_tokens = _priced_token_share(sessions_events)
    pct_priced = round(100.0 * share, 1)
    if share >= 0.99:
        spend_conf = Confidence.HIGH
        note = "Every model in scope has a known price."
    elif share >= 0.80:
        spend_conf = Confidence.MEDIUM
        note = f"{pct_priced}% of tokens ran on priced models; the rest cost $0 here."
    else:
        spend_conf = Confidence.LOW
        note = (
            f"Only {pct_priced}% of tokens ran on priced models — real spend on "
            "the unpriced remainder is invisible to this figure."
        )
    out.append(derived(
        "Total spend", tot["cost_usd"], unit="usd", confidence=spend_conf,
        method="Token counts priced per-model and summed. " + note,
    ))

    # Cache hit rate is a direct ratio of reported token kinds.
    billed = tot["input"] + tot["cache_read"]
    if billed:
        out.append(direct(
            "Cache hit rate", tot["cache_hit_pct"], unit="pct",
            method="Cache-read tokens over billable input, both reported by the agent.",
        ))
    else:
        out.append(unavailable(
            "Cache hit rate", unit="pct",
            method="No billable input tokens were recorded in this scope.",
        ))

    # Discovery share is an exact ratio, but a ratio over a handful of steps is
    # noise — the sample size tempers it.
    steps = eng["total_agent_steps"]
    out.append(derived(
        "Discovery share", eng["discovery_share_pct"], unit="pct",
        confidence=temper(Confidence.HIGH, steps, minimum=30),
        method=f"Search/grep calls over {steps} agent step(s). Exact ratio; "
               "thin on small samples.",
        sample_size=steps,
    ))

    # Loop cost is the sum of behavioural detectors — a tuned signal, not a bill.
    out.append(heuristic(
        "Loop / rework cost", eng["loop_cost_usd"], unit="usd",
        method="Summed from loop-family detector findings, capped at real spend. "
               "A behavioural signal, not a billed line item.",
        sample_size=sum(1 for _ in findings) or None,
    ))

    # Time saved is honestly not computed yet — it needs a configured baseline.
    # Saying "unavailable" out loud is the whole point; a fabricated hour here
    # would be exactly the lie this module exists to prevent.
    out.append(unavailable(
        "Estimated time saved", unit="hours",
        method="Requires a configured engineering baseline; none is set, so no "
               "estimate is shown rather than an invented one.",
    ))

    return out


def summary(measures: list[Measure]) -> dict[str, int]:
    """Count measures by confidence rung — the one-glance 'how solid is this
    page' figure for the top of a data-quality panel."""
    counts = {c.value: 0 for c in Confidence}
    for m in measures:
        counts[m.confidence.value] += 1
    return counts
