"""Session drill-down — the turn-by-turn story behind one session's total.

`Store`/`rollups` already give you the aggregate ("$4.12, 38k tokens"). This
module answers the next question: *where in the session did that go?* It is
pure reuse — the cost/context curve comes from `rollups.context_curve`
(already clamps context fill the same way `detectors/context.py` does for its
peak reading), the per-session model table comes from `rollups.by_model`
scoped to one session, and prompt flagging comes from the same `Evidence.kind
== "prompt"` contract every other Prompt-Lab round-trip already relies on.
No new charting, tokenizing or rollup logic is introduced here.
"""

from __future__ import annotations

from ..schema import Event, Role
from . import rollups
from .detectors import run_session
from .prompt_score import count_tokens


def build_session_detail(events: list[Event]) -> dict:
    """Turn one session's events into the drill-down payload.

    Kept deliberately dumb: no filesystem access, no network, nothing that
    isn't already derivable from events already sitting in the store.
    """
    if not events:
        return {
            "cost_curve": [],
            "context_fill_curve": [],
            "models_used": [],
            "prompts": [],
        }

    curve = rollups.context_curve(events)
    cost_curve = [
        {"seq": row["turn"], "cumulative_usd": row["cost_usd"]} for row in curve
    ]
    # Same clamp `summarize_session`'s peak_context_pct uses (see
    # detectors/context.py's compaction-thrash peak calc for the sibling
    # copy of this reasoning) — a >100% reading destroys trust in every other
    # number on the page, so it's clamped here too rather than re-derived.
    context_fill_curve = [
        {"seq": row["turn"], "pct": min(100.0, row["context_pct"])}
        for row in curve
        if row["context_pct"] is not None
    ]

    models_used = rollups.by_model([events])

    # A prompt "has Prompt-Lab round-trip evidence" when some detector's
    # finding quoted it verbatim as `kind == "prompt"` evidence — that's
    # exactly the set of prompts the Waste tab already offers an "Open in
    # Prompt Lab" button for, so the session view can offer the same button
    # without inventing a second definition of "flagged".
    findings = run_session(events)
    flagged_texts = {
        e.full_text
        for f in findings
        for e in f.evidence
        if e.kind == "prompt" and e.full_text
    }

    prompts = []
    for e in events:
        if e.role is not Role.USER or not e.text or not e.text.strip():
            continue
        prompts.append(
            {
                "seq": e.seq,
                "ts": e.ts,
                "text": e.text,
                "est_tokens": count_tokens(e.text),
                "has_lab_evidence": e.text in flagged_texts,
            }
        )

    return {
        "cost_curve": cost_curve,
        "context_fill_curve": context_fill_curve,
        "models_used": models_used,
        "prompts": prompts,
    }
