"""Aggregations: the table-stakes numbers.

This is the part every competing tool already has. It exists so AgentLens is
taken seriously, not because it is the reason to use AgentLens. The reason is in
``detectors/``.
"""

from __future__ import annotations

from collections import defaultdict

from ..schema import Event, Provider, Role, Session, Usage


def _day(ts: str) -> str:
    return (ts or "")[:10]


def summarize_session(events: list[Event]) -> Session:
    if not events:
        return Session(session_id="", provider=Provider.CLAUDE_CODE)

    first = events[0]
    total_usage = Usage()
    cost = 0.0
    peak_pct = 0.0
    compactions = 0
    models: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0.0, "tokens": 0.0, "cost_usd": 0.0}
    )
    title = None
    stamps = [e.ts for e in events if e.ts]

    for e in events:
        if e.usage:
            total_usage = total_usage + e.usage
        cost += e.cost_usd
        if e.role is Role.COMPACTION:
            compactions += 1
        if e.context_tokens and e.context_limit:
            # Clamped: providers occasionally report cumulative counters that
            # exceed the window, and a "757% full" reading destroys trust in
            # every other number on the page.
            peak_pct = max(peak_pct, min(100.0, 100.0 * e.context_tokens / e.context_limit))
        if e.model and e.usage:
            m = models[e.model]
            m["calls"] += 1
            m["tokens"] += e.usage.total
            m["cost_usd"] += e.cost_usd
        if title is None and e.role in (Role.SYSTEM, Role.USER) and e.text:
            title = e.text.strip().splitlines()[0][:80]

    return Session(
        session_id=first.session_id,
        provider=first.provider,
        title=title,
        started_at=min(stamps) if stamps else "",
        ended_at=max(stamps) if stamps else "",
        total_cost_usd=round(cost, 6),
        usage=total_usage,
        event_count=len(events),
        peak_context_pct=round(peak_pct, 2),
        compaction_count=compactions,
        models={k: {kk: round(vv, 6) for kk, vv in v.items()} for k, v in models.items()},
        repo_id=first.repo_id,
        cwd=first.cwd,
    )


def daily(sessions_events: list[list[Event]]) -> list[dict]:
    buckets: dict[str, dict] = defaultdict(
        lambda: {"date": "", "cost_usd": 0.0, "tokens": 0, "sessions": set()}
    )
    for evs in sessions_events:
        for e in evs:
            d = _day(e.ts)
            if not d:
                continue
            b = buckets[d]
            b["date"] = d
            b["cost_usd"] += e.cost_usd
            b["tokens"] += e.usage.total if e.usage else 0
            b["sessions"].add(e.session_id)
    out = []
    for d in sorted(buckets):
        b = buckets[d]
        out.append(
            {
                "date": d,
                "cost_usd": round(b["cost_usd"], 6),
                "tokens": b["tokens"],
                "sessions": len(b["sessions"]),
            }
        )
    return out


def by_model(sessions_events: list[list[Event]]) -> list[dict]:
    buckets: dict[str, dict] = defaultdict(
        lambda: {"model": "", "calls": 0, "tokens": 0, "cost_usd": 0.0,
                 "cache_read": 0, "input": 0, "output": 0}
    )
    for evs in sessions_events:
        for e in evs:
            if not e.model or not e.usage:
                continue
            b = buckets[e.model]
            b["model"] = e.model
            b["calls"] += 1
            b["tokens"] += e.usage.total
            b["cost_usd"] += e.cost_usd
            b["cache_read"] += e.usage.cache_read
            b["input"] += e.usage.input
            b["output"] += e.usage.output
    rows = sorted(buckets.values(), key=lambda r: -r["cost_usd"])
    for r in rows:
        r["cost_usd"] = round(r["cost_usd"], 6)
        billed = r["input"] + r["cache_read"]
        r["cache_hit_pct"] = round(100.0 * r["cache_read"] / billed, 1) if billed else 0.0
    return rows


def by_provider(sessions_events: list[list[Event]]) -> list[dict]:
    buckets: dict[str, dict] = defaultdict(
        lambda: {"provider": "", "sessions": 0, "tokens": 0, "cost_usd": 0.0}
    )
    for evs in sessions_events:
        if not evs:
            continue
        p = evs[0].provider.value
        b = buckets[p]
        b["provider"] = p
        b["sessions"] += 1
        for e in evs:
            b["tokens"] += e.usage.total if e.usage else 0
            b["cost_usd"] += e.cost_usd
    rows = sorted(buckets.values(), key=lambda r: -r["cost_usd"])
    for r in rows:
        r["cost_usd"] = round(r["cost_usd"], 6)
    return rows


def tool_mix(sessions_events: list[list[Event]]) -> list[dict]:
    calls: dict[str, int] = defaultdict(int)
    out_bytes: dict[str, int] = defaultdict(int)
    for evs in sessions_events:
        for e in evs:
            if not e.tool:
                continue
            k = e.tool.kind.value
            if e.role is Role.TOOL_CALL:
                calls[k] += 1
            if e.role is Role.TOOL_RESULT:
                out_bytes[k] += e.tool.output_bytes
    keys = set(calls) | set(out_bytes)
    return sorted(
        ({"tool": k, "calls": calls[k], "output_bytes": out_bytes[k]} for k in keys),
        key=lambda r: -r["calls"],
    )


def context_curve(events: list[Event]) -> list[dict]:
    out = []
    n = 0
    running = 0.0
    for e in events:
        if e.usage or e.context_tokens:
            n += 1
            running += e.cost_usd
            pct = (
                round(100.0 * e.context_tokens / e.context_limit, 2)
                if e.context_tokens and e.context_limit
                else None
            )
            out.append({"turn": n, "cost_usd": round(running, 6), "context_pct": pct})
    return out


#: `_LOOP_DETECTORS` names the loop-family findings the Engineering tab's
#: "Loop Cost" tile sums. Kept as a literal set here rather than importing
#: the detector classes, so this module — the "table stakes" layer — doesn't
#: reach up into `detectors/`, which is the differentiated layer.
_LOOP_DETECTORS = {"loop", "target_churn", "cross_session_loop"}


def engineering_metrics(sessions_events: list[list[Event]], findings: list | None = None) -> dict:
    """Engineering tab top tiles: discovery share, searches/greps, files
    opened, loop cost. Pure rollup over events + findings already computed
    elsewhere — no new detector.
    """
    from ..schema import ToolKind

    total_steps = 0
    search_calls = 0
    project_search = 0
    shell_grep = 0
    opened: dict[str, int] = defaultdict(int)

    for evs in sessions_events:
        for e in evs:
            if e.role is not Role.TOOL_CALL or not e.tool:
                continue
            total_steps += 1
            if e.tool.kind is ToolKind.SEARCH:
                search_calls += 1
                project_search += 1
            elif e.tool.kind is ToolKind.BASH and e.text and "grep" in e.text.lower():
                search_calls += 1
                shell_grep += 1
            if e.tool.kind is ToolKind.READ and e.tool.target_path:
                opened[e.tool.target_path] += 1

    discovery_share_pct = round(100.0 * search_calls / total_steps, 1) if total_steps else 0.0
    files_opened = len(opened)
    repeat_opens = sum(1 for n in opened.values() if n >= 2)
    raw_loop_cost = sum(
        f.wasted_usd for f in (findings or []) if f.detector in _LOOP_DETECTORS
    )
    # Belt-and-braces sanity cap: the three loop-family detectors are
    # independent and can each attribute cost from the same session (a
    # session can be both a repeated-failure loop *and* part of a
    # cross-session recurrence), so their sum is not guaranteed distinct even
    # after each detector's own internal dedup. "Waste" can never legitimately
    # exceed what was actually spent analysing these sessions — capping here
    # is the last line of defense against a detector bug quietly inflating
    # this tile into something the user's real bill can't back up.
    total_spend = sum(e.cost_usd for evs in sessions_events for e in evs)
    loop_cost_usd = round(min(raw_loop_cost, total_spend), 4)

    return {
        "discovery_share_pct": discovery_share_pct,
        "total_agent_steps": total_steps,
        "searches": {
            "total": search_calls,
            "project_search": project_search,
            "shell_grep": shell_grep,
        },
        "files_opened": files_opened,
        "files_opened_repeat": repeat_opens,
        "loop_cost_usd": loop_cost_usd,
    }


def totals(sessions_events: list[list[Event]]) -> dict:
    u = Usage()
    cost = 0.0
    compactions = 0
    for evs in sessions_events:
        for e in evs:
            if e.usage:
                u = u + e.usage
            cost += e.cost_usd
            if e.role is Role.COMPACTION:
                compactions += 1
    billed = u.input + u.cache_read
    return {
        "sessions": len(sessions_events),
        "cost_usd": round(cost, 4),
        "tokens": u.total,
        "input": u.input,
        "output": u.output,
        "cache_read": u.cache_read,
        "cache_write": u.cache_write,
        "cache_hit_pct": round(100.0 * u.cache_read / billed, 1) if billed else 0.0,
        "compactions": compactions,
    }
