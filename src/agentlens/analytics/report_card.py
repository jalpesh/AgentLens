"""Shareable AI-coding report card — one 1200×675 image (the size LinkedIn and
X preview at) built only from aggregates the dashboard already computed.

Privacy rule: nothing on the card can identify code or projects. It never
shows prompts, file paths, project names or evidence quotes. The "biggest
habit" comes only from waste/context detectors, whose titles are fixed text;
prompt-pattern titles are the user's own sentences, so they are excluded.
``hide_money`` swaps every dollar figure for a share, for people who want to
post their habits but not their bill.
"""

from __future__ import annotations

from datetime import date
from xml.sax.saxutils import escape

W, H = 1200, 675

AGENT_NAMES = {
    "claude-code": "Claude Code", "codex": "Codex", "junie": "Junie", "opencode": "OpenCode",
    "gemini-cli": "Gemini CLI", "copilot-cli": "Copilot CLI", "goose": "Goose", "aider": "Aider",
}
#: Grade from the avoidable share of spend. A rule of thumb, stated on the card.
GRADES = ((5, "A"), (10, "B"), (15, "C"), (25, "D"))


def grade(avoidable_pct: float) -> str:
    return next((g for limit, g in GRADES if avoidable_pct < limit), "F")


def _period(daily: list[dict]) -> str:
    days = sorted(d["date"] for d in daily if d.get("date"))
    if not days:
        return ""
    a, b = (date.fromisoformat(x[:10]) for x in (days[0], days[-1]))
    if (a.year, a.month) == (b.year, b.month):
        return a.strftime("%b %Y")
    return f"{a:%b} – {b:%b %Y}" if a.year == b.year else f"{a:%b %Y} – {b:%b %Y}"


def card_data(totals: dict, reclaimable: float, recommendations: list[dict],
              task_types: list[dict], tasks: list[dict], by_provider: list[dict],
              daily: list[dict]) -> dict:
    spend = float(totals.get("cost_usd") or 0)
    costs = sorted(t["cost_usd"] for t in tasks)
    habit = next((r for r in recommendations
                  if r.get("source") in ("waste", "context") and r.get("cost_usd")), None)
    pct = 100 * reclaimable / spend if spend else 0.0
    return {
        "period": _period(daily),
        "agents": [AGENT_NAMES.get(p["provider"], p["provider"])
                   for p in sorted(by_provider, key=lambda p: -p.get("cost_usd", 0))
                   # a couple of $0 trial runs isn't "an agent I use"
                   if p.get("cost_usd") or p.get("sessions", 0) >= 3][:4],
        "spend": round(spend, 2),
        "sessions": int(totals.get("sessions") or 0),
        "avoidable_usd": round(reclaimable, 2),
        "avoidable_pct": round(pct, 1),
        "grade": grade(pct),
        "tasks": len(costs),
        "median_task": round(costs[len(costs) // 2], 2) if costs else None,
        "habit": None if not habit else {
            "title": habit["title"], "cost_usd": habit["cost_usd"],
            "pct": round(100 * habit["cost_usd"] / spend, 1) if spend else 0.0,
            "sessions": habit["sessions"], "fix": habit["recommendation"]},
        "types": [{"type": t["task_type"], "pct": t["pct"], "per_task": t["cost_per_task"]}
                  for t in task_types[:5]],
    }


def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return escape(text if len(text) <= n else text[: n - 1].rstrip() + "…")


def _money(v: float) -> str:
    return f"${v:,.0f}" if v >= 100 else f"${v:,.2f}"


def render_svg(d: dict, hide_money: bool = False) -> str:
    """The card as a standalone SVG string (fonts are system fonts, no fetches)."""
    bg, panel, line, text, dim, accent, warn = (
        "#0f1115", "#171a21", "#262a33", "#f2f4f8", "#8b93a1", "#34d399", "#f87171")
    font = "-apple-system,BlinkMacSystemFont,'Segoe UI',Inter,Helvetica,Arial,sans-serif"
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'viewBox="0 0 {W} {H}" font-family="{font}">',
         f'<rect width="{W}" height="{H}" fill="{bg}"/>']
    t = o.append

    # Header
    t(f'<circle cx="64" cy="62" r="7" fill="{accent}"/>')
    t(f'<text x="80" y="68" fill="{text}" font-size="20" font-weight="700">AgentLens</text>')
    t(f'<text x="56" y="112" fill="{text}" font-size="34" font-weight="700">'
      'My AI coding report card</text>')
    sub = " · ".join(x for x in (d["period"], " · ".join(d["agents"])) if x)
    t(f'<text x="56" y="146" fill="{dim}" font-size="18">{_clip(sub, 80)}</text>')

    # Grade
    gc = accent if d["grade"] in "AB" else ("#fbbf24" if d["grade"] == "C" else warn)
    t(f'<rect x="932" y="40" width="212" height="128" rx="16" fill="{panel}" stroke="{line}"/>')
    t(f'<text x="990" y="134" fill="{gc}" font-size="84" font-weight="800" '
      f'text-anchor="middle">{d["grade"]}</text>')
    t(f'<text x="1040" y="92" fill="{dim}" font-size="14">efficiency</text>')
    t(f'<text x="1040" y="114" fill="{text}" font-size="22" font-weight="700">'
      f'{d["avoidable_pct"]:g}%</text>')
    t(f'<text x="1040" y="136" fill="{dim}" font-size="14">avoidable</text>')

    # Stat tiles
    tiles = ([("Sessions", f'{d["sessions"]:,}'), ("Tasks", f'{d["tasks"]:,}'),
              ("Agents", f'{len(d["agents"])}')] if hide_money else
             [("AI spend", _money(d["spend"])), ("Tasks", f'{d["tasks"]:,}'),
              ("Typical task", _money(d["median_task"]) if d["median_task"] is not None
               else "—")])
    for i, (label, value) in enumerate(tiles):
        x = 56 + i * 296
        t(f'<rect x="{x}" y="188" width="280" height="104" rx="14" fill="{panel}" '
          f'stroke="{line}"/>')
        t(f'<text x="{x + 22}" y="222" fill="{dim}" font-size="15" letter-spacing="1">'
          f'{escape(label.upper())}</text>')
        t(f'<text x="{x + 22}" y="270" fill="{text}" font-size="38" font-weight="700">'
          f'{escape(value)}</text>')
    t(f'<rect x="944" y="188" width="200" height="104" rx="14" fill="{panel}" stroke="{line}"/>')
    t(f'<text x="966" y="222" fill="{dim}" font-size="15" letter-spacing="1">AVOIDABLE</text>')
    avoid = (f'{d["avoidable_pct"]:g}%' if hide_money else _money(d["avoidable_usd"]))
    t(f'<text x="966" y="270" fill="{warn}" font-size="38" font-weight="700">'
      f'{escape(avoid)}</text>')

    # Biggest habit
    t(f'<rect x="56" y="312" width="560" height="236" rx="14" fill="{panel}" stroke="{line}"/>')
    t(f'<text x="78" y="346" fill="{dim}" font-size="15" letter-spacing="1">'
      'MY MOST EXPENSIVE HABIT</text>')
    h = d["habit"]
    if h:
        t(f'<text x="78" y="384" fill="{text}" font-size="19" font-weight="700">'
          f'{_clip(h["title"], 52)}</text>')
        cost = (f'{h["pct"]:g}% of spend' if hide_money else
                f'{_money(h["cost_usd"])} · {h["pct"]:g}% of spend')
        t(f'<text x="78" y="416" fill="{warn}" font-size="18" font-weight="600">'
          f'{escape(cost)} · {h["sessions"]} sessions</text>')
        words, lines = h["fix"].replace("`", "").split(), [""]
        for w in words:  # wrap the fix to three short lines
            if len(lines[-1]) + len(w) > 50:
                if len(lines) == 3:
                    lines[-1] = lines[-1].rstrip() + " …"
                    break
                lines.append("")
            lines[-1] += w + " "
        t(f'<text x="78" y="458" fill="{accent}" font-size="16" font-weight="600">FIX</text>')
        for i, ln in enumerate(lines):
            t(f'<text x="78" y="{484 + i * 24}" fill="{text}" font-size="17">'
              f'{escape(ln.strip())}</text>')
    else:
        t(f'<text x="78" y="390" fill="{text}" font-size="20">No costly habit found.</text>')

    # Spend by task type
    t(f'<rect x="632" y="312" width="512" height="236" rx="14" fill="{panel}" stroke="{line}"/>')
    t(f'<text x="654" y="346" fill="{dim}" font-size="15" letter-spacing="1">'
      'WHERE THE MONEY WENT</text>')
    top = max((x["pct"] for x in d["types"]), default=1) or 1
    for i, ty in enumerate(d["types"]):
        y = 378 + i * 34
        t(f'<text x="654" y="{y + 5}" fill="{text}" font-size="16">'
          f'{escape(ty["type"])}</text>')
        t(f'<rect x="790" y="{y - 9}" width="{max(4, 190 * ty["pct"] / top):.0f}" height="16" '
          f'rx="4" fill="{accent}" opacity="0.85"/>')
        right = f'{ty["pct"]:g}%' if hide_money else f'{ty["pct"]:g}% · {_money(ty["per_task"])}/task'
        t(f'<text x="1124" y="{y + 5}" fill="{dim}" font-size="15" text-anchor="end">'
          f'{escape(right)}</text>')

    # Footer
    t(f'<line x1="56" y1="580" x2="1144" y2="580" stroke="{line}"/>')
    t(f'<text x="56" y="618" fill="{dim}" font-size="16">Analysed on my machine — no prompts '
      'or code uploaded. Grade: A &lt;5% avoidable, B &lt;10%, C &lt;15%, D &lt;25%.</text>')
    t(f'<text x="1144" y="618" fill="{text}" font-size="16" font-weight="600" '
      'text-anchor="end">github.com/jalpesh/AgentLens</text>')
    t('</svg>')
    return "\n".join(o)
