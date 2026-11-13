"""Detector registry — the differentiated layer.

Every finding produced here must carry: the user's own words, a dollar figure,
and one concrete rewrite. See ``base.py`` for why.

Two kinds of detector live here:

* `Detector` — sees one session at a time. Almost everything is one of these.
* `GlobalDetector` — sees every session at once, for anything that is only
  visible across sessions (the same failure recurring in March and August).
"""

from __future__ import annotations

from ...schema import Event
from .base import Attempt, Detector, Escalation, Evidence, Finding, GlobalDetector, merge
from .blind_retry import BlindRetryDetector
from .context import (
    CompactionThrashDetector,
    ModelMismatchDetector,
    VagueInstructionDetector,
)
from .loops import (
    CrossSessionLoopDetector,
    LoopDetector,
    SameTargetChurnDetector,
    signature,
)
from .output_flood import OutputFloodDetector
from .workflow import (
    EditWithoutTestDetector,
    UncachedDocsDetector,
    WholeFileRereadDetector,
)

DETECTORS: list[Detector] = [
    LoopDetector(),
    SameTargetChurnDetector(),
    BlindRetryDetector(),
    OutputFloodDetector(),
    VagueInstructionDetector(),
    EditWithoutTestDetector(),
    WholeFileRereadDetector(),
    CompactionThrashDetector(),
    UncachedDocsDetector(),
    ModelMismatchDetector(),
]

GLOBAL_DETECTORS: list[GlobalDetector] = [
    CrossSessionLoopDetector(),
]


def run_session(events: list[Event]) -> list[Finding]:
    out = []
    for d in DETECTORS:
        try:
            f = d.run(events)
        except Exception:  # a broken detector must never break the report
            continue
        if f:
            out.append(f)
    return out


def run_global(sessions: list[list[Event]]) -> list[Finding]:
    out = []
    for d in GLOBAL_DETECTORS:
        try:
            f = d.run_global(sessions)
        except Exception:
            continue
        if f:
            out.append(f)
    return out


def run_all(sessions: list[list[Event]]) -> list[Finding]:
    return run_all_scored(sessions)[0]


def run_all_scored(
    sessions: list[list[Event]], dismissed: frozenset | set = frozenset()
) -> tuple[list[Finding], float]:
    """Per-session detectors, merged, plus cross-session ones — and the
    reclaimable dollar figure the headline tile shows.

    Global findings are appended *after* the merge and then re-sorted, because
    they are already whole-corpus results — passing them through `merge` would
    have nothing to merge them with and would recompute a severity they set
    deliberately.

    Detectors overlap: a loop, its target churn and the flooded output re-sent
    on every looped turn are often the same dollars. Summing findings and
    capping at total spend therefore showed "100% reclaimable" on real data.
    Instead, each session contributes only its single largest finding, capped
    at that session's spend — a floor that never counts a dollar twice.
    `dismissed` is (detector, session_id, ts) evidence the user said wasn't
    waste; it comes off that session's figure too.
    """
    findings: list[Finding] = []
    reclaimable = 0.0
    for evs in sessions:
        spend = sum(e.cost_usd for e in evs)
        best = 0.0
        for f in run_session(evs):
            # A per-session finding can never cost more than its session did.
            f.wasted_usd = min(f.wasted_usd, spend)
            kept = f.wasted_usd - sum(
                ev.cost_usd for ev in f.evidence
                if (f.detector, ev.session_id, ev.ts) in dismissed
            )
            best = max(best, kept)
            findings.append(f)
        reclaimable += min(best, spend)
    merged = merge(findings)
    merged.extend(run_global(sessions))

    # Backstop: no single finding may exceed everything analysed (global
    # detectors aren't covered by the per-session cap above).
    total_spend = sum(e.cost_usd for evs in sessions for e in evs)
    for f in merged:
        if f.wasted_usd > total_spend:
            f.wasted_usd = total_spend

    return sorted(merged, key=lambda f: (-f.wasted_usd, -f.occurrences)), reclaimable


__all__ = [
    "Detector", "GlobalDetector", "Escalation", "Evidence", "Finding", "Attempt", "merge",
    "DETECTORS", "GLOBAL_DETECTORS", "run_session", "run_global", "run_all", "run_all_scored",
    "signature",
    "LoopDetector", "SameTargetChurnDetector", "CrossSessionLoopDetector",
    "BlindRetryDetector", "OutputFloodDetector", "VagueInstructionDetector",
    "EditWithoutTestDetector", "WholeFileRereadDetector",
    "CompactionThrashDetector", "UncachedDocsDetector", "ModelMismatchDetector",
]
