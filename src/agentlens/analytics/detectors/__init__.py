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
    """Per-session detectors, merged, plus cross-session ones.

    Global findings are appended *after* the merge and then re-sorted, because
    they are already whole-corpus results — passing them through `merge` would
    have nothing to merge them with and would recompute a severity they set
    deliberately.
    """
    findings: list[Finding] = []
    for evs in sessions:
        findings.extend(run_session(evs))
    merged = merge(findings)
    merged.extend(run_global(sessions))

    # Backstop, not the primary fix: `merge()` sums each detector's per-session
    # `wasted_usd` with no ceiling, so any detector with an overlap or
    # double-count bug we haven't (yet) found and fixed at the source can still
    # report a finding costing more than every session it drew from combined.
    # Two prior spot-fixes (`rollups.engineering_metrics`'s loop_cost_usd and
    # `server.py`'s aggregate `reclaimable`) capped the *sum across findings*
    # downstream, but never touched an individual finding's own `wasted_usd` —
    # so a single finding could still show a number bigger than the user's
    # entire analyzed spend even after those fixes, which is exactly what real
    # usage data surfaced (target_churn and loop findings each individually
    # exceeding total spend, 2026-08-16). Capping every finding here, at the
    # one place all of them pass through, makes "a finding costs more than you
    # spent" structurally impossible instead of "impossible until the next
    # detector ships with the same bug."
    total_spend = sum(e.cost_usd for evs in sessions for e in evs)
    for f in merged:
        if f.wasted_usd > total_spend:
            f.wasted_usd = total_spend

    return sorted(merged, key=lambda f: (-f.wasted_usd, -f.occurrences))


__all__ = [
    "Detector", "GlobalDetector", "Escalation", "Evidence", "Finding", "Attempt", "merge",
    "DETECTORS", "GLOBAL_DETECTORS", "run_session", "run_global", "run_all",
    "signature",
    "LoopDetector", "SameTargetChurnDetector", "CrossSessionLoopDetector",
    "BlindRetryDetector", "OutputFloodDetector", "VagueInstructionDetector",
    "EditWithoutTestDetector", "WholeFileRereadDetector",
    "CompactionThrashDetector", "UncachedDocsDetector", "ModelMismatchDetector",
]
