"""Local dashboard server.

Stdlib only — no FastAPI, no uvicorn, no Node. That is a deliberate deployment
choice: ``uvx agentlens-cli serve`` must work identically on macOS, Linux and
Windows with nothing else installed. Every extra runtime dependency is a person
who never gets the tool running.

Binds to 127.0.0.1 by default. Nothing is exposed to the network and nothing is
uploaded anywhere.
"""

from __future__ import annotations

import dataclasses
import json
import re
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from ..analytics import (
    CURATED_REFERENCE,
    build_playbook,
    build_session_detail,
    content_type_budget,
    fingerprint_all,
    generic_suggestions,
    prompt_trends,
    rollups,
    scaffold,
    score_across_models,
    score_prompt,
    suggest,
)
from ..analytics.detectors import run_all
from ..redact import redact_obj
from ..schema import Provider, Usage
from ..store import Store

STATIC = Path(__file__).parent / "static"
_MARKER = "const AGENTLENS_STATIC_DATA = null;"

#: Cap on the citation drill-down's disk read (see `/api/citation` below and
#: SECURITY.md's "Loop citations" section). Large enough to show real
#: context around a cited failure, small enough that pointing this at an
#: enormous file doesn't turn a UI click into reading gigabytes into memory.
_CITATION_MAX_LINES = 400
_CITATION_MAX_BYTES = 512_000

#: Matches a POSIX-absolute path ("/home/dev/x.py"), a Windows drive path
#: ("C:\x.py" or "C:/x.py"), or a Windows UNC path ("\\host\share\x.py").
#:
#: `Path(path).is_absolute()` is NOT good enough here: it's answered by
#: whichever `Path` class the *current* OS instantiates, not by the shape of
#: the string. The whole point of citation is showing a file from a session
#: that may have run on a *different* machine — a Claude Code session logged
#: on Linux/macOS records POSIX paths like "/home/dev/project/file.py". Open
#: that same dashboard on Windows and `Path("/home/dev/...").is_absolute()`
#: is `False` (Windows absolute paths need a drive letter), so the endpoint
#: rejected a perfectly well-formed absolute path as malformed input instead
#: of correctly reporting "file not available on this machine". Caught by CI
#: failing on windows-latest only, 2026-08-15 — ubuntu/macOS runners never
#: exercise the branch where the string and the OS disagree.
_ABS_PATH_RE = re.compile(r"^(/|[A-Za-z]:[\\/]|\\\\)")


def _apply_dismissals(findings: list, dismissed: set[tuple[str, str, str]]) -> list:
    """Filter dismissed evidence out of findings and recompute the numbers
    that describe what's left, rather than leaving a finding's headline
    dollar figure counting evidence the person already said wasn't waste.

    A finding whose *every* piece of evidence gets dismissed is dropped
    entirely — "0 occurrences, $0.00 wasted" is not a habit worth a card.
    `wasted_usd`/`wasted_tokens` are recomputed as the sum over the evidence
    that survives, which is an approximation for detectors whose original
    figure came from a window computation rather than a literal sum of
    per-evidence costs (see loops.py) — the alternative, leaving the old
    total in place after evidence was removed from under it, would be a
    number the remaining evidence can't account for, which is worse.
    """
    import dataclasses

    kept: list = []
    for f in findings:
        surviving = [
            e for e in f.evidence if (f.detector, e.session_id, e.ts) not in dismissed
        ]
        if not surviving:
            continue
        if len(surviving) != len(f.evidence):
            f = dataclasses.replace(
                f,
                evidence=surviving,
                occurrences=len(surviving),
                wasted_usd=round(sum(e.cost_usd for e in surviving), 6),
                wasted_tokens=sum(e.tokens for e in surviving),
            )
        kept.append(f)
    return kept


def _exclude_manual(sessions: list[list]) -> list[list]:
    """Drop manually-entered sessions (see `Provider.MANUAL`) before handing
    events to anything that reasons over turn-by-turn behaviour — detectors,
    the project fingerprint. A manual session is one synthetic aggregate
    event with no prompts and no tool calls, so in practice no detector
    would ever fire on it regardless of this filter; this exists as an
    explicit, testable guarantee rather than relying on that being true
    forever as detectors change, and the filter itself carries no privacy
    weight either way — it just decides what a manual entry is allowed to
    influence.
    """
    return [evs for evs in sessions if evs and evs[0].provider != Provider.MANUAL]


def build_payload(
    db: str | None,
    provider: str | None = None,
    days: int | None = None,
    repo_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """The dashboard's full data payload — same function whether it's served
    live over `/api/data` or baked into a static export, so the two can never
    disagree about what a session or a finding looks like."""
    from datetime import datetime, timedelta

    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d") if days else None
    with Store(db) as store:
        pairs = list(store.iter_sessions(provider, since, repo_id, session_id))
        sessions = [evs for _, evs in pairs]
        summaries = [rollups.summarize_session(e) for e in sessions]
        # Scope options are deliberately computed UNFILTERED — a dropdown that
        # only lists what the current filter already matches is a one-way trip
        # the user can't navigate back out of.
        all_projects = store.projects()
        all_sessions = store.sessions_index(limit=300)
        dismissed = store.dismissed_keys()
        dismissed_count = store.dismissed_count()

    # Manually-entered sessions belong in totals and charts (that's the whole
    # point of adding them) but not in anything that reasons over
    # turn-by-turn behaviour — see `_exclude_manual`.
    real_sessions = _exclude_manual(sessions)

    findings = run_all(real_sessions)
    if dismissed:
        findings = _apply_dismissals(findings, dismissed)
    profile = fingerprint_all(real_sessions)
    totals = rollups.totals(sessions)
    # Different detectors can legitimately point at overlapping evidence in
    # the same session (a session can be flagged by more than one detector),
    # so the sum across all findings is not guaranteed to be a distinct
    # dollar figure. It can never legitimately exceed what was actually spent
    # analysing these sessions — cap it here rather than let a detector's
    # internal double-count (or two detectors' honest-but-overlapping totals)
    # show the user a bill bigger than their real one. See
    # `rollups.engineering_metrics` for the same cap applied to the
    # loop-family subset specifically.
    reclaimable = min(sum(f.wasted_usd for f in findings), totals["cost_usd"])
    engineering = rollups.engineering_metrics(sessions, findings)

    return {
        "totals": totals,
        "reclaimable_usd": round(reclaimable, 4),
        "reclaimable_pct": round(
            100.0 * reclaimable / totals["cost_usd"], 1
        ) if totals["cost_usd"] else 0.0,
        "daily": rollups.daily(sessions),
        "by_model": rollups.by_model(sessions),
        "by_provider": rollups.by_provider(sessions),
        "tool_mix": rollups.tool_mix(sessions),
        "findings": [f.to_dict() for f in findings],
        "dismissed_count": dismissed_count,
        "playbook": [r.to_dict() for r in build_playbook(findings)],
        "curated_reference": CURATED_REFERENCE,
        # Suggestions and the project profile ride along in the same payload,
        # so the static export gets them too — the dashboard never writes
        # files, it only previews and downloads, so there is nothing here that
        # needs a live server.
        "suggestions": [s.to_dict() for s in suggest(findings, profile, engineering)],
        # Generic, org-wide bucket — never derived from this user's data, kept
        # as a visibly separate list so it can never be mistaken for a
        # triggered suggestion. See analytics/skills.py's module docstring.
        "generic_suggestions": [s.to_dict() for s in generic_suggestions()],
        "profile": profile.to_dict(),
        "engineering": engineering,
        "history": prompt_trends(sessions),
        "sessions": sorted(
            (
                {
                    "session_id": s.session_id,
                    "provider": s.provider.value,
                    "title": s.title or s.session_id,
                    "started_at": s.started_at,
                    "cost_usd": s.total_cost_usd,
                    "tokens": s.usage.total,
                    "peak_context_pct": s.peak_context_pct,
                    "compactions": s.compaction_count,
                }
                for s in summaries
            ),
            key=lambda r: -r["cost_usd"],
        ),
        "providers": [p["provider"] for p in rollups.by_provider(sessions)],
        "scopes": {
            "projects": all_projects,
            "sessions": all_sessions,
            "providers": sorted({s.provider.value for s in summaries}),
        },
        "active_scope": {
            "provider": provider,
            "days": days,
            "repo_id": repo_id,
            "session_id": session_id,
            "project_label": next(
                (p["label"] for p in all_projects if p["repo_id"] == repo_id), None
            ),
        },
    }


def export_html(
    out_path: str,
    db: str | None = None,
    provider: str | None = None,
    days: int | None = None,
    repo_id: str | None = None,
    session_id: str | None = None,
    redact: bool = True,
) -> Path:
    """Bake the dashboard's data into ``web/static/index.html`` and write one
    self-contained file: no server, no network, opens with a double-click.

    This is a *substitution*, not a fork — the exact same template that
    `serve()` sends over HTTP is reused verbatim, with only the marker line
    replaced by a JSON literal. If the two ever needed separate copies of the
    HTML they would inevitably drift; this way there is exactly one dashboard
    implementation, served two ways.

    Redacted by default, since a static file is explicitly meant to be shared
    — pass ``redact=False`` to keep raw prompt text (secrets are still
    stripped regardless; see `redact.py`).
    """
    from datetime import datetime

    payload = build_payload(db, provider, days, repo_id, session_id)
    payload["generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["redacted"] = redact
    if redact:
        payload = redact_obj(payload, full=True)

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    if _MARKER not in html:
        raise RuntimeError(
            "static export marker not found in index.html — template and "
            "exporter have drifted out of sync"
        )
    literal = "const AGENTLENS_STATIC_DATA = " + json.dumps(payload, default=str) + ";"
    html = html.replace(_MARKER, literal, 1)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def make_handler(db: str | None):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # keep the console clean
            pass

        def _send(self, body: bytes, ctype: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, status: int = 200) -> None:
            self._send(json.dumps(obj, default=str).encode(), "application/json", status)

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = parse_qs(u.query)
            provider = (q.get("provider") or [None])[0] or None
            days = int(q["days"][0]) if q.get("days") and q["days"][0].isdigit() else None
            repo_id = (q.get("repo_id") or [None])[0] or None
            session_id = (q.get("session_id") or [None])[0] or None

            if u.path in ("/", "/index.html"):
                html = (STATIC / "index.html").read_bytes()
                self._send(html, "text/html; charset=utf-8")
            elif u.path == "/api/data":
                try:
                    self._json(build_payload(db, provider, days, repo_id, session_id))
                except Exception as exc:
                    self._json({"error": str(exc)}, 500)
            elif u.path == "/api/session":
                sid = (q.get("id") or [""])[0]
                with Store(db) as store:
                    evs = store.events_for(sid)
                summary = rollups.summarize_session(evs) if evs else None
                self._json(
                    {
                        "session_id": sid,
                        "curve": rollups.context_curve(evs),
                        # `Session` is a `slots=True` dataclass — it has no
                        # `__dict__` — so this walks its declared fields
                        # instead of a magic attribute that dataclasses with
                        # slots never populate.
                        "summary": {
                            f.name: getattr(summary, f.name)
                            for f in dataclasses.fields(summary)
                            if f.name != "usage"
                        }
                        if summary
                        else {},
                        "findings": [f.to_dict() for f in run_all([evs])] if evs else [],
                        # Session drill-down (cost/context-fill curves, the
                        # per-session model table, the full prompt list) —
                        # built from the same events, no extra query.
                        "detail": build_session_detail(evs),
                    },
                    200,
                )
            elif u.path == "/api/citation":
                self._json(self._citation((q.get("path") or [""])[0]))
            else:
                self._json({"error": "not found"}, 404)

        @staticmethod
        def _citation(path: str) -> dict:
            """The one place in AgentLens that reads a file's *content* off
            disk, and only when this exact endpoint is called with an
            explicit path — never as part of `/api/data` or any default
            payload build. See SECURITY.md's "Loop citations" section for
            the full contract this implements.

            Deliberately resilient: the repo this path pointed to may have
            moved, or this may not even be the machine the session ran on.
            Every failure mode degrades to `{"available": False, "reason": ...}`
            rather than raising — a stack trace here would be a worse
            experience than just saying the file isn't there.
            """
            if not path:
                return {"available": False, "reason": "no path given"}
            try:
                if not _ABS_PATH_RE.match(path):
                    return {"available": False, "reason": "not an absolute path"}
                p = Path(path)
                if not p.exists():
                    return {
                        "available": False,
                        "reason": "file not available on this machine",
                    }
                if not p.is_file():
                    return {"available": False, "reason": "not a regular file"}
                with p.open("r", encoding="utf-8", errors="replace") as fh:
                    lines = []
                    read_chars = 0
                    truncated = False
                    for i, line in enumerate(fh):
                        # Compare against a *character* budget, not the file's
                        # byte size — a multi-byte-UTF-8 file (an emoji, a
                        # non-ASCII path) has fewer characters than bytes, and
                        # comparing across those units flags files as
                        # truncated that were actually read in full.
                        if i >= _CITATION_MAX_LINES or read_chars >= _CITATION_MAX_BYTES:
                            truncated = True
                            break
                        lines.append(line.rstrip("\n"))
                        read_chars += len(line)
                    else:
                        # Loop completed without hitting a cap — but there
                        # may still be more after the last line read if the
                        # very last line was exactly at the boundary.
                        truncated = fh.read(1) != ""
                return {
                    "available": True,
                    "path": path,
                    "content": "\n".join(lines),
                    "truncated": truncated,
                }
            except OSError as exc:
                return {"available": False, "reason": f"could not read file: {exc}"}
            except Exception as exc:  # never let a citation request crash the server
                return {"available": False, "reason": f"unexpected error: {exc}"}

        def _read_json_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            return body if isinstance(body, dict) else {}

        def do_POST(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            if u.path == "/api/dismiss":
                return self._dismiss()
            if u.path == "/api/manual-session":
                return self._manual_session()
            if u.path != "/api/lab":
                return self._json({"error": "not found"}, 404)
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                text = body.get("text", "")
                model = body.get("model", "claude-sonnet-4-5")
                r = score_prompt(text, model)
                out = r.to_dict()

                # When the prompt came from a finding, we know which session it
                # belongs to — so the scaffold can be filled with that session's
                # real file, test command and error instead of placeholders.
                events = []
                sid = body.get("session_id")
                if sid:
                    with Store(db) as store:
                        events = store.events_for(sid)
                out["scaffold"] = scaffold(
                    text, events, body.get("seq"), [i.code for i in r.issues]
                ).to_dict()

                # Prompt Lab additions (section C): same prompt re-scored
                # against every model actually present in the user's
                # history, cheapest first, and the token budget by content
                # type. Both pure reuse of `score_prompt`/`count_tokens` —
                # see analytics/prompt_score.py.
                with Store(db) as store:
                    known_models = store.distinct_models() or [model]
                out["models_compared"] = [
                    rep.to_dict() for rep in score_across_models(text, known_models)
                ]
                out["content_budget"] = content_type_budget(text, model)
                self._json(out)
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

        def _dismiss(self) -> None:
            """Mark one piece of evidence as "not waste" — see `Store.dismiss_evidence`.
            Idempotent: dismissing the same evidence twice is a no-op, not an
            error, so a double-click or a stale second tab can't misbehave."""
            try:
                body = self._read_json_body()
                detector = str(body.get("detector") or "")
                session_id = str(body.get("session_id") or "")
                ts = str(body.get("ts") or "")
                if not (detector and session_id and ts):
                    return self._json(
                        {"error": "detector, session_id and ts are all required"}, 400
                    )
                reason = body.get("reason")
                reason = str(reason).strip()[:500] if reason else None
                with Store(db) as store:
                    store.dismiss_evidence(detector, session_id, ts, reason)
                    total = store.dismissed_count()
                self._json({"ok": True, "dismissed_total": total})
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

        def _manual_session(self) -> None:
            """The dashboard's other write endpoint — see `Store.add_manual_session`
            and `Provider.MANUAL`'s docstring. Local-only (127.0.0.1 by
            default), add-only: there is no edit here, and removal is a CLI
            command (`agentlens sessions --remove <id>`) on purpose, so an
            accidental click can't quietly delete something typed in five
            minutes ago."""
            try:
                body = self._read_json_body()
                label = str(body.get("label") or "").strip()
                if not label:
                    return self._json({"error": "a project/tool label is required"}, 400)
                date = str(body.get("date") or "").strip()
                model = (str(body.get("model") or "").strip()) or None

                def _int(key: str) -> int:
                    try:
                        return max(0, int(body.get(key) or 0))
                    except (TypeError, ValueError):
                        return 0

                usage = Usage(input=_int("tokens_input"), output=_int("tokens_output"))
                cost_override = body.get("cost_usd")
                if cost_override in (None, ""):
                    from ..pricing import cost_of

                    cost_usd = cost_of(model, usage)
                else:
                    try:
                        cost_usd = max(0.0, float(cost_override))
                    except (TypeError, ValueError):
                        return self._json({"error": "cost_usd must be a number"}, 400)

                import uuid
                from datetime import datetime

                ts = f"{date}T00:00:00" if date else datetime.now().isoformat(timespec="seconds")
                session_id = f"manual-{uuid.uuid4().hex[:16]}"
                with Store(db) as store:
                    store.add_manual_session(session_id, ts, label, model, usage, cost_usd)
                self._json({"ok": True, "session_id": session_id, "cost_usd": round(cost_usd, 6)})
            except Exception as exc:
                self._json({"error": str(exc)}, 500)

    return Handler


def serve(
    host: str = "127.0.0.1", port: int = 7878, db: str | None = None, open_browser: bool = True
) -> None:
    httpd = ThreadingHTTPServer((host, port), make_handler(db))
    url = f"http://{host}:{port}"
    print(f"AgentLens dashboard → {url}")
    print("All analysis is local. Nothing is uploaded. Ctrl-C to stop.")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
