"""AgentLens CLI.

    agentlens doctor      which agents are on this machine
    agentlens ingest      parse local history into the local database
    agentlens report      cost + token summary (table stakes)
    agentlens waste       WHY it was expensive, with evidence  ← the point
    agentlens lab         score a prompt before you send it
    agentlens playbook    ranked, paste-ready rules from your own findings
    agentlens sessions    per-session breakdown
    agentlens projects    list projects for --project
    agentlens skills      agent/skill/hook suggestions from your findings
    agentlens serve       local dashboard on 127.0.0.1:7878
    agentlens export      redacted JSON

Everything runs offline. Nothing is uploaded, ever.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .adapters import ADAPTERS, EXPERIMENTAL, get as get_adapter
from .analytics import (
    build_playbook,
    fingerprint_all,
    llm_rewrite,
    render_playbook,
    rollups,
    scaffold,
    score_prompt,
    suggest,
)
from .analytics.skills import generic_suggestions, get as skills_get
from .analytics.detectors import run_all
from .redact import redact_obj
from .store import SCHEMA_VERSION, Store, default_db_path

console = Console()

SEV_COLOR = {"high": "bright_red", "medium": "yellow", "low": "cyan"}


def _money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.01 else f"${x:.4f}"


def _since(days: int | None) -> str | None:
    if not days:
        return None
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")


def _open_store(args) -> Store:
    """Open the database and surface a schema upgrade if one just happened.

    Without this, a user upgrading AgentLens would run `waste`, see an empty
    report, and reasonably conclude the tool is broken — when in fact the
    derived cache was rebuilt and just needs repopulating.
    """
    store = Store(args.db)
    if store.migrated_from is not None:
        console.print(
            f"[yellow]Database upgraded[/yellow] (schema v{store.migrated_from} → "
            f"v{SCHEMA_VERSION}). The local cache was rebuilt — your agent history "
            "files were not touched.\nRun [bold]agentlens ingest[/bold] to repopulate."
        )
    return store


def _scope(args, store: Store) -> dict:
    """Resolve the filters every reporting command shares."""
    return {
        "provider": getattr(args, "provider", None),
        "since": _since(getattr(args, "days", None)),
        "repo_id": store.resolve_project(getattr(args, "project", None)),
        "session_id": getattr(args, "session", None),
    }


def _scoped_sessions(args, store: Store) -> list:
    s = _scope(args, store)
    return [
        evs
        for _, evs in store.iter_sessions(
            s["provider"], s["since"], s["repo_id"], s["session_id"]
        )
    ]


# --- commands -----------------------------------------------------------


def cmd_doctor(args) -> int:
    t = Table(title="Agents on this machine", header_style="bold")
    t.add_column("Agent")
    t.add_column("Status")
    t.add_column("History location")
    t.add_column("Files", justify="right")

    any_found = False
    hints: list[str] = []
    for a in ADAPTERS:
        found = a.detect()
        files = len(a.discover()) if found else 0
        any_found = any_found or files > 0
        label = a.display_name
        if a.provider in EXPERIMENTAL:
            label += " [dim]⚠ experimental[/dim]"
        status = Text("found", style="green") if found else Text("not installed", style="dim")
        if found and files == 0:
            status = Text("found, 0 readable", style="yellow")
        t.add_row(label, status, str(a.roots()[0]), str(files) if found else "—")
        if a.last_permission_error and (hint := a.permission_hint()):
            hints.append(f"  {a.display_name}: {hint}")

    console.print(t)
    console.print(f"\n[dim]Database:[/dim] {default_db_path()}")
    if hints:
        console.print("\n[yellow]Permission issues:[/yellow]")
        for h in hints:
            console.print(f"[dim]{h}[/dim]")
    if EXPERIMENTAL & {a.provider for a in ADAPTERS if a.detect()}:
        console.print(
            "\n[dim]⚠ experimental adapters read undocumented or unconfirmed formats — "
            "verify their numbers look right before trusting them.[/dim]"
        )
    if not any_found:
        console.print(
            "\n[yellow]No agent history found.[/yellow] Try "
            "[bold]python fixtures/generate.py --demo ~/agentlens-demo[/bold] "
            "to generate a synthetic dataset and see what AgentLens does."
        )
    return 0


def cmd_ingest(args) -> int:
    adapters = [get_adapter(args.provider)] if args.provider else ADAPTERS
    adapters = [a for a in adapters if a]
    total_files = 0
    total_events = 0

    with _open_store(args) as store:
        if args.reset:
            store.reset()
            console.print("[dim]Cleared existing data.[/dim]")

        for a in adapters:
            sources = a.discover()
            if not sources:
                continue
            fresh = 0
            for s in sources:
                key = str(s.path)
                if not args.force and store.source_is_current(
                    key, s.size_bytes, s.mtime, a.schema_version
                ):
                    continue
                try:
                    n = store.add_events(a.parse(s))
                except Exception as exc:  # one bad file must not abort the run
                    if args.verbose:
                        console.print(f"[red]skip[/red] {s.path}: {exc}")
                    continue
                store.mark_source(key, a.provider.value, s.size_bytes, s.mtime, a.schema_version)
                total_events += n
                fresh += 1
            total_files += fresh
            console.print(
                f"  {a.display_name:<14} {len(sources):>4} file(s), {fresh:>4} new/changed"
            )

        c = store.counts()

    console.print(
        f"\n[green]Ingested[/green] {total_events:,} new events from {total_files} file(s). "
        f"Database now holds {c['events']:,} events across {c['sessions']:,} sessions "
        f"({_money(c['cost_usd'])})."
    )
    if total_events == 0 and total_files == 0:
        console.print("[dim]Nothing new — everything already up to date.[/dim]")
    return 0


def cmd_report(args) -> int:
    if args.html:
        from .web.server import export_html

        with Store(args.db) as store:
            repo_id = store.resolve_project(args.project)
        out = export_html(
            args.html, db=args.db, provider=args.provider, days=args.days,
            repo_id=repo_id, session_id=args.session, redact=not args.raw,
        )
        console.print(
            f"[green]Wrote[/green] {out} — open it directly, no server needed"
            f"{'' if args.raw else ' [dim](redacted)[/dim]'}"
        )
        return 0

    with _open_store(args) as store:
        sessions = _scoped_sessions(args, store)

    if not sessions:
        console.print("[yellow]No data.[/yellow] Run [bold]agentlens ingest[/bold] first.")
        return 1

    tot = rollups.totals(sessions)
    console.print(
        Panel.fit(
            f"[bold]{_money(tot['cost_usd'])}[/bold] across {tot['sessions']:,} sessions\n"
            f"{tot['tokens']:,} tokens · {tot['cache_hit_pct']}% served from cache · "
            f"{tot['compactions']} compaction(s)",
            title="Total",
            border_style="cyan",
        )
    )

    t = Table(title="By model", header_style="bold")
    for c, j in (("Model", "left"), ("Calls", "right"), ("Tokens", "right"),
                 ("Cache hit", "right"), ("Cost", "right")):
        t.add_column(c, justify=j)
    for r in rollups.by_model(sessions)[:15]:
        t.add_row(r["model"], f"{r['calls']:,}", f"{r['tokens']:,}",
                  f"{r['cache_hit_pct']}%", _money(r["cost_usd"]))
    console.print(t)

    p = Table(title="By agent", header_style="bold")
    for c, j in (("Agent", "left"), ("Sessions", "right"), ("Tokens", "right"), ("Cost", "right")):
        p.add_column(c, justify=j)
    for r in rollups.by_provider(sessions):
        p.add_row(r["provider"], f"{r['sessions']:,}", f"{r['tokens']:,}", _money(r["cost_usd"]))
    console.print(p)

    days = rollups.daily(sessions)[-14:]
    if days:
        d = Table(title="Recent days", header_style="bold")
        for c, j in (("Date", "left"), ("Sessions", "right"), ("Tokens", "right"), ("Cost", "right")):
            d.add_column(c, justify=j)
        for r in days:
            d.add_row(r["date"], str(r["sessions"]), f"{r['tokens']:,}", _money(r["cost_usd"]))
        console.print(d)
    return 0


def cmd_waste(args) -> int:
    """The reason this tool exists."""
    with _open_store(args) as store:
        sessions = _scoped_sessions(args, store)

    if not sessions:
        console.print("[yellow]No data.[/yellow] Run [bold]agentlens ingest[/bold] first.")
        return 1

    findings = run_all(sessions)
    spend = rollups.totals(sessions)["cost_usd"]

    if not findings:
        console.print(
            Panel.fit(
                "No recurring waste patterns found across "
                f"{len(sessions)} session(s). Your prompts are carrying their weight.",
                border_style="green",
            )
        )
        return 0

    reclaimable = sum(f.wasted_usd for f in findings)
    pct = (100.0 * reclaimable / spend) if spend else 0.0
    console.print(
        Panel.fit(
            f"[bold]{_money(reclaimable)}[/bold] of {_money(spend)} "
            f"([bold]{pct:.0f}%[/bold]) traced to habits you repeat.\n"
            f"[dim]Across {len(sessions)} session(s). Ranked by what it costs you.[/dim]",
            title="Change this",
            border_style="bright_red" if pct > 15 else "yellow",
        )
    )

    for i, f in enumerate(findings, 1):
        if f.wasted_usd < args.min_usd and f.occurrences < 3:
            continue
        color = SEV_COLOR.get(f.severity, "white")
        head = Text()
        head.append(f"{i}. {f.title}  ", style=f"bold {color}")
        head.append(f"{_money(f.wasted_usd)}", style="bold")
        head.append(f"  ·  {f.occurrences}×", style="dim")
        if f.wasted_tokens:
            head.append(f"  ·  {f.wasted_tokens:,} tokens", style="dim")
        if f.blames_model:
            # Worth calling out explicitly: every other finding is something
            # the user did. This one isn't, and saying so is what makes the
            # rest of the report credible.
            head.append("  ·  not your prompt", style="italic cyan")
        console.print()
        console.print(head)
        console.print(Text(f"   Why it costs: {f.mechanism}", style="dim"))
        # `advice_for` returns the escalation rung matching how often this
        # happened, falling back to the static prescription. For most detectors
        # these are the same string.
        console.print(Text(f"   → {f.advice_for()}", style="green"))
        if f.escalation and len(f.escalation) > 1:
            console.print(Text("     If it keeps happening:", style="dim"))
            for step in sorted(f.escalation, key=lambda e: e.threshold):
                marker = "▸" if f.occurrences >= step.threshold else "·"
                style = "cyan" if f.occurrences >= step.threshold else "dim"
                console.print(
                    Text(f"       {marker} {step.threshold}+ → {step.advice}", style=style)
                )

        for ev in f.evidence[: args.evidence]:
            console.print()
            when = (ev.ts or "")[:16].replace("T", " ")
            console.print(Text(f'     "{ev.quote}"', style="italic"))
            meta = f"       {when}"
            if ev.cost_usd:
                meta += f" · {_money(ev.cost_usd)}"
            if ev.detail:
                meta += f" · {ev.detail}"
            console.print(Text(meta, style="dim"))

    console.print()
    return 0


def cmd_lab(args) -> int:
    text = args.text
    if not text:
        if sys.stdin.isatty():
            console.print("[dim]Paste your prompt, then Ctrl-D:[/dim]")
        text = sys.stdin.read()

    r = score_prompt(text, args.model)

    # Pull real context from the named session so the scaffold isn't a generic
    # form — it should already know the file, the test command and the error.
    events: list = []
    if getattr(args, "session", None):
        with _open_store(args) as store:
            events = store.events_for(args.session)
    sc = scaffold(text, events, None, [i.code for i in r.issues])

    if args.json:
        payload = r.to_dict()
        payload["scaffold"] = sc.to_dict()
        print(json.dumps(payload, indent=2))
        return 0

    color = "green" if r.score >= 85 else "yellow" if r.score >= 55 else "bright_red"
    console.print(
        Panel.fit(
            f"[bold {color}]{r.score}[/bold {color}]/100   {r.verdict}\n"
            f"[dim]{r.tokens:,} tokens · {r.words} words · "
            f"{r.context_pct:.2f}% of {r.context_limit:,} window · "
            f"{_money(r.cost_usd)} to send on {r.model}[/dim]",
            title="Prompt report",
            border_style=color,
        )
    )

    if r.saved_tokens:
        console.print(
            f"[dim]Removing filler saves {r.saved_tokens} token(s) → "
            f"{_money(r.cleaned_cost_usd)}. Meaning unchanged.[/dim]\n"
        )

    if not r.issues:
        console.print("[green]Nothing wasteful found.[/green]")
        return 0

    for iss in sorted(r.issues, key=lambda i: -i.penalty):
        console.print(Text(f"  −{iss.penalty:<3} {iss.label}", style="bold yellow"))
        if iss.span:
            console.print(Text(f'       found: "{iss.span}"', style="dim italic"))
        console.print(Text(f"       → {iss.fix}", style="green"))
        console.print()

    rewritten = sc.text
    if getattr(args, "llm", False):
        try:
            rewritten = llm_rewrite(text, sc.text, args.model)
        except Exception as exc:
            console.print(f"[yellow]LLM rewrite unavailable:[/yellow] {exc}")
            console.print("[dim]Falling back to the offline scaffold.[/dim]\n")

    console.print(
        Panel(
            rewritten,
            title="Rewrite" + (" [dim](via LLM)[/dim]" if args.llm else " [dim](offline)[/dim]"),
            border_style="cyan",
        )
    )
    if sc.filled:
        console.print(
            "[dim]Filled from this session: "
            + ", ".join(f"{k}={v[:48]}" for k, v in sc.filled.items())
            + "[/dim]"
        )
    if sc.placeholders and not args.llm:
        console.print(f"[dim]Still needs: {', '.join(sc.placeholders)}[/dim]")
    return 0


def cmd_sessions(args) -> int:
    with _open_store(args) as store:
        sc = _scope(args, store)
        sessions = store.sessions(sc["provider"], sc["since"], sc["repo_id"], sc["session_id"])
    if not sessions:
        console.print("[yellow]No data.[/yellow] Run [bold]agentlens ingest[/bold] first.")
        return 1

    sessions.sort(key=lambda s: -s.total_cost_usd)
    t = Table(title="Sessions by cost", header_style="bold")
    for c, j in (("Started", "left"), ("Agent", "left"), ("Title", "left"),
                 ("Tokens", "right"), ("Peak ctx", "right"), ("Cost", "right")):
        t.add_column(c, justify=j, overflow="ellipsis")
    for s in sessions[: args.limit]:
        t.add_row(
            (s.started_at or "")[:16].replace("T", " "),
            s.provider.value,
            (s.title or s.session_id)[:52],
            f"{s.usage.total:,}",
            f"{s.peak_context_pct:.0f}%",
            _money(s.total_cost_usd),
        )
    console.print(t)
    return 0


def cmd_skills(args) -> int:
    with _open_store(args) as store:
        sessions = _scoped_sessions(args, store)
    if not sessions:
        console.print("[yellow]No data.[/yellow] Run [bold]agentlens ingest[/bold] first.")
        return 1

    findings = run_all(sessions)
    profile = fingerprint_all(sessions)
    metrics = rollups.engineering_metrics(sessions, findings)
    suggestions = suggest(findings, profile, metrics)
    generic = generic_suggestions()

    # --- init: write or preview one artifact -------------------------------
    if args.init:
        s = skills_get(suggestions, args.init) or skills_get(generic, args.init)
        if not s:
            console.print(f"[red]Unknown suggestion:[/red] {args.init}")
            console.print("[dim]Run 'agentlens skills' to list available ids.[/dim]")
            return 1
        if not s.artifact:
            console.print(f"[yellow]{s.title}[/yellow] has no file to generate.")
            return 1

        dest = Path(args.dir or ".") / s.artifact.path
        if not args.write:
            console.print(
                Panel(
                    s.artifact.content.rstrip(),
                    title=f"{s.artifact.path} [dim](preview)[/dim]",
                    border_style="cyan",
                )
            )
            console.print(f"[dim]Write it with:[/dim] agentlens skills --init {s.id} --write")
            return 0

        if dest.exists() and not args.force:
            console.print(
                f"[yellow]{dest} already exists.[/yellow] Re-run with --force to overwrite, "
                "or --dir to write elsewhere."
            )
            return 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(s.artifact.content, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {dest}")
        return 0

    # --- list ---------------------------------------------------------------
    lang = profile.primary_language or "unknown"
    console.print(
        Panel.fit(
            f"[bold]{lang}[/bold] project"
            + (f" · tests: [bold]{profile.test_command}[/bold]" if profile.test_command else "")
            + (f" · {profile.package_manager}" if profile.package_manager else "")
            + f"\n[dim]Inferred from {profile.sample_events:,} events"
            + ("" if profile.confident else " — thin sample, treat as a guess")
            + ". No filesystem scan.[/dim]",
            title="Detected project",
            border_style="cyan" if profile.confident else "yellow",
        )
    )

    def _print_suggestion(s) -> None:
        head = Text()
        head.append(f"{s.id}  ", style="bold cyan")
        head.append(f"[{s.kind}]  ", style="dim")
        head.append(s.title, style="bold")
        if s.evidence_usd:
            head.append(f"  {_money(s.evidence_usd)}", style="bold")
        console.print(head)
        console.print(Text(f"   {s.why}", style="dim"))
        if s.triggered_by:
            console.print(Text(f"   triggered by: {s.triggered_by}", style="dim italic"))
        if s.unverified_category:
            console.print(
                Text(
                    "   ⚠ described as a category — AgentLens does not name packages "
                    "it hasn't verified.",
                    style="yellow",
                )
            )
        if s.artifact:
            console.print(Text(f"   → {s.artifact.path}", style="green"))
        console.print()

    if not suggestions:
        console.print(
            "\n[green]Nothing to suggest from your data.[/green] Suggestions are generated "
            "from findings, and there aren't any worth acting on."
        )
    else:
        console.print()
        for s in suggestions:
            _print_suggestion(s)

    console.print(
        Text("Generic best practices — not derived from your data", style="bold")
    )
    console.print(
        Text(
            "Useful for most projects regardless of what we found.",
            style="dim",
        )
    )
    console.print()
    for s in generic:
        _print_suggestion(s)

    console.print(
        "[dim]Preview:[/dim] agentlens skills --init <id>          "
        "[dim]Write:[/dim] agentlens skills --init <id> --write"
    )
    return 0


def cmd_projects(args) -> int:
    with _open_store(args) as store:
        projects = store.projects()
    if not projects:
        console.print("[yellow]No data.[/yellow] Run [bold]agentlens ingest[/bold] first.")
        return 1
    t = Table(title="Projects", header_style="bold")
    for c, j in (("Project", "left"), ("Sessions", "right"), ("Cost", "right"),
                 ("Last seen", "left"), ("Id", "left")):
        t.add_column(c, justify=j, overflow="ellipsis")
    for p in projects:
        t.add_row(
            p["label"], str(p["sessions"]), _money(p["cost_usd"]),
            (p["last_seen"] or "")[:10], p["repo_id"][:8],
        )
    console.print(t)
    console.print("\n[dim]Use with:[/dim] agentlens waste --project <name>")
    return 0


def cmd_playbook(args) -> int:
    with _open_store(args) as store:
        sessions = _scoped_sessions(args, store)

    if not sessions:
        console.print("[yellow]No data.[/yellow] Run [bold]agentlens ingest[/bold] first.")
        return 1

    findings = run_all(sessions)
    rules = build_playbook(findings, limit=args.limit)
    text = render_playbook(rules, fmt=args.format, include_reference=not args.no_reference)

    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {args.out} — {len(rules)} personalised rule(s)")
    else:
        print(text)
    return 0


def cmd_export(args) -> int:
    with _open_store(args) as store:
        sessions = _scoped_sessions(args, store)
        summaries = [rollups.summarize_session(e) for e in sessions]

    payload = {
        "agentlens_version": __version__,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "redacted": not args.raw,
        "totals": rollups.totals(sessions),
        "by_model": rollups.by_model(sessions),
        "by_provider": rollups.by_provider(sessions),
        "daily": rollups.daily(sessions),
        "tool_mix": rollups.tool_mix(sessions),
        "findings": [f.to_dict() for f in run_all(sessions)],
        "playbook": [r.to_dict() for r in build_playbook(run_all(sessions))],
        "sessions": [
            {
                "session_id": s.session_id, "provider": s.provider.value, "title": s.title,
                "started_at": s.started_at, "cost_usd": s.total_cost_usd,
                "tokens": s.usage.total, "peak_context_pct": s.peak_context_pct,
                "compactions": s.compaction_count, "repo_id": s.repo_id,
            }
            for s in summaries
        ],
    }
    if not args.raw:
        payload = redact_obj(payload, full=True)

    out = json.dumps(payload, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        console.print(f"[green]Wrote[/green] {args.out}"
                      f"{'' if args.raw else ' [dim](redacted)[/dim]'}")
    else:
        print(out)
    return 0


def cmd_serve(args) -> int:
    from .web.server import serve

    serve(host=args.host, port=args.port, db=args.db, open_browser=not args.no_open)
    return 0


# --- parser -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agentlens",
        description="Find out why your coding agent is expensive. Entirely offline.",
    )
    p.add_argument("--version", action="version", version=f"agentlens {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--db", default=None, help="database path")
        sp.add_argument("--provider", default=None,
                        choices=[a.provider.value for a in ADAPTERS])
        sp.add_argument("--days", type=int, default=None, help="only the last N days")
        sp.add_argument("--project", default=None, metavar="NAME",
                        help="limit to one project (name or id; see 'agentlens projects')")
        sp.add_argument("--session", default=None, metavar="ID",
                        help="limit to a single session id")
        return sp

    d = sub.add_parser("doctor", help="show which agents are installed")
    d.set_defaults(func=cmd_doctor)

    i = common(sub.add_parser("ingest", help="parse local agent history"))
    i.add_argument("--force", action="store_true", help="re-parse unchanged files")
    i.add_argument("--reset", action="store_true", help="wipe the database first")
    i.add_argument("-v", "--verbose", action="store_true")
    i.set_defaults(func=cmd_ingest)

    r = common(sub.add_parser("report", help="cost and token summary"))
    r.add_argument("--html", metavar="PATH", default=None,
                    help="write a self-contained static dashboard instead (no server needed)")
    r.add_argument("--raw", action="store_true",
                    help="with --html: skip redaction (secrets are still stripped)")
    r.set_defaults(func=cmd_report)

    w = common(sub.add_parser("waste", help="why it was expensive, with evidence"))
    w.add_argument("--min-usd", type=float, default=0.005)
    w.add_argument("--evidence", type=int, default=2, help="examples shown per finding")
    w.set_defaults(func=cmd_waste)

    s = common(sub.add_parser("sessions", help="per-session breakdown"))
    s.add_argument("--limit", type=int, default=25)
    s.set_defaults(func=cmd_sessions)

    sk = common(sub.add_parser(
        "skills", help="agent/skill/hook suggestions derived from your findings"
    ))
    sk.add_argument("--init", metavar="ID", default=None,
                    help="preview one suggestion's generated file")
    sk.add_argument("--write", action="store_true",
                    help="with --init: actually write the file to disk")
    sk.add_argument("--dir", default=None, metavar="PATH",
                    help="project root to write into (default: current directory)")
    sk.add_argument("--force", action="store_true", help="overwrite an existing file")
    sk.set_defaults(func=cmd_skills)

    pj = sub.add_parser("projects", help="list projects available to --project")
    pj.add_argument("--db", default=None)
    pj.set_defaults(func=cmd_projects)

    pb = common(sub.add_parser(
        "playbook", help="ranked, paste-ready rules generated from your own findings"
    ))
    pb.add_argument("--format", choices=["claude", "agents", "junie", "markdown"],
                     default="markdown")
    pb.add_argument("--limit", type=int, default=8,
                     help="max personalised rules (default 8 — more is context bloat)")
    pb.add_argument("--no-reference", action="store_true",
                     help="omit the static curated-reference section")
    pb.add_argument("--out", default=None)
    pb.set_defaults(func=cmd_playbook)

    lab = sub.add_parser("lab", help="score a prompt before you send it")
    lab.add_argument("text", nargs="?", help="prompt text (or pipe on stdin)")
    lab.add_argument("--model", default="claude-sonnet-4-5")
    lab.add_argument("--json", action="store_true")
    lab.add_argument("--db", default=None)
    lab.add_argument("--session", default=None, metavar="ID",
                     help="fill the rewrite from this session's real file/test/error")
    lab.add_argument("--llm", action="store_true",
                     help="rewrite via an LLM (needs ANTHROPIC_API_KEY or OPENAI_API_KEY; "
                          "the default scaffold is fully offline)")
    lab.set_defaults(func=cmd_lab)

    e = common(sub.add_parser("export", help="export analysis as JSON"))
    e.add_argument("--out", default=None)
    e.add_argument("--raw", action="store_true",
                   help="skip redaction (secrets are still stripped)")
    e.set_defaults(func=cmd_export)

    v = sub.add_parser("serve", help="local dashboard")
    v.add_argument("--db", default=None)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=7878)
    v.add_argument("--no-open", action="store_true")
    v.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
