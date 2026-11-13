"""Optimization Recommendation Engine — one ranked to-do list.

Waste detectors, Context Efficiency and Prompt Patterns each find problems in
their own vocabulary. This module puts them in one shape (problem, evidence,
fix, expected impact, confidence, affected sessions) and ranks by money, then
tokens, so the dashboard can answer "what should I change first?".

It adds no new detection: every recommendation is backed by a finding the
other modules already computed. Savings come from different lenses on the same
sessions and can overlap, so they are never summed into one headline.
"""

from __future__ import annotations

from .confidence import Confidence, temper

DEFAULT_LIMIT = 10


def _impact(cost: float | None, tokens: int, sessions: int) -> str:
    if not tokens and not cost:
        return "Insufficient data to estimate a saving."
    money = ("cost unknown: unpriced model" if not cost
             else "<$0.01" if cost < 0.01 else f"~${cost:,.2f}")
    what = f"{tokens:,} tokens ({money})" if tokens else money
    return f"Up to {what} over the analysed period, across {sessions} session(s)."


def _rec(source, kind, title, problem, recommendation, confidence, sessions,
         tokens, cost, evidence) -> dict:
    cost = round(cost, 4) if cost else None
    return {
        "source": source, "kind": kind, "title": title, "problem": problem,
        "recommendation": recommendation, "confidence": confidence,
        "sessions": sessions, "tokens": int(tokens), "cost_usd": cost,
        "expected_impact": _impact(cost, int(tokens), sessions),
        "evidence": [e for e in evidence if e][:3],
    }


def recommend(findings: list[dict], context_opps: list[dict], patterns: list[dict],
              limit: int = DEFAULT_LIMIT) -> list[dict]:
    """All inputs are the `to_dict()` forms already in the dashboard payload."""
    recs = []
    for f in findings:
        n = f.get("sessions") or len({e["session_id"] for e in f.get("evidence", [])})
        # Detectors are tuned rules of thumb: MEDIUM at best, lower when thin.
        conf = temper(Confidence.MEDIUM, n or f.get("occurrences", 0))
        problem = f["mechanism"] + (" The model, not the prompt, looks at fault here."
                                    if f.get("blames_model") else "")
        recs.append(_rec("waste", f["detector"], f["title"], problem, f["advice"],
                         conf.value, n, f.get("wasted_tokens", 0),
                         f.get("wasted_usd", 0.0),
                         [e.get("quote", "") for e in f.get("evidence", [])]))
    for o in context_opps:
        conf = temper(Confidence.MEDIUM, o["sessions"])
        recs.append(_rec("context", o["kind"], o["title"],
                         f"{o['occurrences']} occurrence(s) put "
                         f"{o['tokens']:,} tokens of likely-unneeded context in front of the model.",
                         o["recommendation"], conf.value, o["sessions"], o["tokens"],
                         o["cost_usd"], o.get("examples", [])))
    for p in patterns:
        recs.append(_rec("prompt", p["kind"], p["title"], p["problem"], p["recommendation"],
                         p["confidence"], p["sessions"], p["tokens"], p["cost_usd"],
                         p.get("examples", [])))
    # ponytail: fixed order by $ then tokens; add effort/ease weighting if users ask
    recs.sort(key=lambda r: (-(r["cost_usd"] or 0), -r["tokens"], -r["sessions"]))
    return recs[:limit]
