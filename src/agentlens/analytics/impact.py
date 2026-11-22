"""Skill / guideline before-after measurement.

You added a skill, a CLAUDE.md / AGENTS.md rule or a hook on some date. Did
sessions get cheaper or smoother afterwards? This splits the sessions in scope
at that date and compares the two sides on the same metrics.

It works for every agent because it needs nothing but the change date: agents
don't reliably record when an instruction file changed or which skill ran.

Honesty rules:
- Fewer than ``MIN_SIDE`` sessions on either side → "Insufficient data", no number.
- Per-session cost/tokens/prompts use the **median**: one marathon session
  would otherwise decide the verdict.
- Before/after is not a controlled test. Confidence never exceeds MEDIUM, and a
  change of main model between the sides is called out, since a cheaper model
  alone moves cost.
"""

from __future__ import annotations

from collections import Counter
from statistics import median

from ..schema import Event, Provider, Role
from .confidence import Confidence
from .detectors import run_all_scored
from .prompt_patterns import _CORRECTION, _prompts

MIN_SIDE = 5
#: Both sides this big earns MEDIUM; below it the comparison is LOW.
SOLID_SIDE = 20


def _side(sessions: list[list[Event]]) -> dict:
    prompts = corr = results = failed = 0
    per_cost, per_tokens, per_prompts = [], [], []
    models: Counter = Counter()
    for evs in sessions:
        ps = {id(p) for p in _prompts(evs)}
        acted, n_p = False, 0
        for e in evs:
            if e.role is Role.TOOL_CALL:
                acted = True
            elif e.role is Role.TOOL_RESULT and e.tool:
                results += 1
                failed += e.tool.failed
            elif id(e) in ps:
                n_p += 1
                corr += acted and bool(_CORRECTION.match(e.text))
                acted = False
            if e.model and e.cost_usd:
                models[e.model] += e.cost_usd
        prompts += n_p
        per_cost.append(sum(e.cost_usd for e in evs))
        per_tokens.append(sum(e.usage.total for e in evs if e.usage))
        per_prompts.append(n_p)
    spend = sum(per_cost)
    _, reclaimable = run_all_scored(sessions)
    return {
        "sessions": len(sessions),
        "Cost / session (median)": median(per_cost),
        "Tokens / session (median)": median(per_tokens),
        "Prompts / session (median)": median(per_prompts),
        "Correction rate %": 100.0 * corr / prompts if prompts else None,
        "Failed tool calls %": 100.0 * failed / results if results else None,
        "Waste share %": 100.0 * min(reclaimable, spend) / spend if spend else None,
        "main_model": models.most_common(1)[0][0] if models else None,
    }


#: Every metric here is better when it goes down.
METRICS = ("Cost / session (median)", "Tokens / session (median)",
           "Prompts / session (median)", "Correction rate %", "Failed tool calls %",
           "Waste share %")


def before_after(sessions_events: list[list[Event]], change_date: str, label: str = "") -> dict:
    """`change_date` is YYYY-MM-DD; events on that day count as after."""
    real = [evs for evs in sessions_events if evs and evs[0].provider is not Provider.MANUAL]
    # Split events, not sessions: a resumed session can run for months, and its
    # work after the change belongs on the "after" side.
    # A segment with no typed prompt (idle tail, harness-only resume) is not work.
    def split(keep):
        segs = ([e for e in evs if keep((e.ts or "")[:10])] for evs in real)
        return [s for s in segs if next(_prompts(s), None)]

    before = split(lambda d: d < change_date)
    after = split(lambda d: d >= change_date)
    out = {"change_date": change_date, "label": label,
           "before_sessions": len(before), "after_sessions": len(after),
           "rows": [], "caveats": [], "available": False}
    if len(before) < MIN_SIDE or len(after) < MIN_SIDE:
        out["caveats"].append(
            f"Insufficient data: need {MIN_SIDE}+ sessions on each side of {change_date} "
            f"(have {len(before)} before, {len(after)} after).")
        return out

    b, a = _side(before), _side(after)
    conf = (Confidence.MEDIUM if min(len(before), len(after)) >= SOLID_SIDE
            and b["main_model"] == a["main_model"] else Confidence.LOW)
    for m in METRICS:
        bv, av = b[m], a[m]
        change = (100.0 * (av - bv) / bv) if bv not in (None, 0) and av is not None else None
        out["rows"].append({
            "metric": m,
            "before": None if bv is None else round(bv, 4),
            "after": None if av is None else round(av, 4),
            "change_pct": None if change is None else round(change, 1),
            # ponytail: ±5% is noise for a before/after split; a real test
            # (bootstrap CI) would replace this if the verdicts get relied on.
            "verdict": ("insufficient data" if change is None
                        else "better" if change <= -5 else "worse" if change >= 5
                        else "no clear change"),
            "confidence": conf.value,
        })
    out["available"] = True
    out["caveats"].append("Before/after comparison, not a controlled test: task mix, "
                          "model and project changes over time also move these numbers.")
    if b["main_model"] != a["main_model"]:
        out["caveats"].append(
            f"Main model changed ({b['main_model']} → {a['main_model']}); cost and token "
            "differences may come from the model, not from this change.")
    return out
