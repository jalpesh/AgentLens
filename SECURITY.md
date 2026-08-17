# Security & privacy

## What AgentLens reads

Only the session history your coding agents already write to your own disk:

- `~/.claude/projects/**/*.jsonl`
- `$CODEX_HOME` (default `~/.codex`) → `sessions/`, `archived_sessions/`
- `~/.junie/history.json`
- `~/.local/share/opencode/storage/`
- `~/.gemini/tmp/**/chats/` ⚠️ experimental
- `~/.copilot/session-store.db` ⚠️ experimental — opened **read-only**
- `~/.local/share/goose/sessions/` (SQLite or legacy JSONL) ⚠️ experimental — SQLite opened **read-only**

The two SQLite-backed adapters connect with `mode=ro` in the connection URI, so
even a bug in AgentLens cannot write to, corrupt, or lock a live agent database
it's reading from.

One narrow, explicit exception to "session logs only": the dashboard's loop
citation drill-down can read the *content* of a specific file named in a
finding, on click. See "Loop citations" below for the full contract.

## Where it sends it

**By default: nowhere.** No telemetry, no analytics, no crash reporting, no
account, no API key. The default install works with the network interface off,
and `agentlens serve` binds to `127.0.0.1` only.

There is exactly **one** code path in the entire tool that can make a network
request, and it is opt-in twice over:

| | |
|---|---|
| Command | `agentlens lab --llm` |
| Requires | the `--llm` flag **and** `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` in your environment |
| Sends | the prompt you typed into the Prompt Lab, plus the offline scaffold |
| Without it | the Prompt Lab rewrite is generated locally and is fully offline |

Verify it yourself. The only network import in the package is inside
`llm_rewrite()`, imported at call time so that a normal run never even loads an
HTTP stack:

```bash
grep -rn "urllib.request\|requests\|httpx\|socket" src/agentlens/
# → src/agentlens/analytics/rewrite.py, inside llm_rewrite() only
```

The dashboard, ingest, detectors, playbook, export and every other command have
no network code at all.

*(Historical note: before the opt-in rewrite existed this file claimed there was
no HTTP client anywhere in the tree. That claim stopped being true the moment
`--llm` shipped, so it was rewritten rather than left standing. A
privacy-positioned tool that overstates its guarantees by one sentence loses
more trust than the feature gains.)*

## Loop citations — the one place AgentLens reads file *content*, not metadata

Everywhere else in this document, "what AgentLens reads" means session-log
metadata only — it never scans your repository. There is exactly one
deliberate, narrow exception, and it is opt-in per click, the same way
`--llm` is opt-in per flag:

| | |
|---|---|
| Where | The dashboard's loop/churn findings, in a collapsed "Cite the file →" row |
| Endpoint | `GET /api/citation?path=<absolute path>` — served only by the live `agentlens serve`, never present in a static `report --html` export |
| Triggers on | An explicit click on that one drill-down button for that one finding |
| Reads | Up to 400 lines / 512 KB of the **specific file** named in that finding's evidence (a path AgentLens already had as tool-call metadata — this is the first time it reads that file's *content*) |
| Never | Called as part of `/api/data`, `/api/session`, `report --html`, `export`, or any other default payload — `tests/test_citation.py` asserts `build_payload` never touches `Path.open` |
| Never | Written to `~/.agentlens/agentlens.db` or anywhere else — the read result lives only in that one HTTP response and the browser tab that requested it |
| Degrades to | `{"available": false, "reason": "..."}` — never a crash — when the file has moved, been deleted, or this isn't the machine the session ran on. This is the common case: reviewing history on a laptop where the repo is no longer checked out. |

Why this exists at all: a loop/churn finding names the file it kept failing
on, and being able to show the current contents of that one file — the same
citation the finding's evidence already describes in prose — is a genuinely
more concrete answer than a path string. It's scoped as tightly as the value
justifies: one file, one click, one response, never cached, never part of a
default page load, and refused outright for anything that isn't an absolute
path (no relative-path ambiguity about *whose* filesystem it resolves
against).

Verify the scope yourself — this is the only handler in the server that opens
a file named by a request parameter:

```bash
grep -n "def _citation" src/agentlens/web/server.py
```

## The two endpoints that write — both local-only, both add-only

Every other endpoint is read-only. These two write to
`~/.agentlens/agentlens.db` and nowhere else — never to a file on disk,
never off the machine, never reachable except from `127.0.0.1` by default.

| | `/api/dismiss` | `/api/manual-session` |
|---|---|---|
| Where | Waste tab, "Not waste? Dismiss →" on a piece of evidence | Sessions tab, "Add a session" form |
| Writes | One row to a `dismissed` table: `(detector, session_id, ts, reason)` | One synthetic aggregate event tagged `provider = "manual"` |
| Never | Touches your agent history files, or evidence/sessions it wasn't explicitly pointed at | Contains turn-by-turn data — it's a total, not a transcript, so it can't produce a detector finding |
| Undo | None in the UI — dismissals are meant to be occasional, not toggled | `agentlens sessions --remove <id>`, scoped to `provider = "manual"` rows only, so it can never delete real ingested history even if pointed at the wrong id |

Verify the scope yourself:

```bash
grep -n "def _dismiss\|def _manual_session" src/agentlens/web/server.py
```

## What it stores

A local SQLite database at `~/.agentlens/agentlens.db` (override with
`AGENTLENS_HOME`). Delete the directory to remove everything.

Repository paths are stored as salted 16-character hashes, never as paths, so
grouping by project never reveals which project.

## Sharing output

`agentlens export` redacts by default: URLs, home directories, email addresses,
IP addresses, and secret-shaped strings. Secrets (API keys, bearer tokens,
private keys, AWS keys) are stripped even with `--raw`.

Before sharing a screenshot of the dashboard, note that it shows prompt text
verbatim — that is the point of the tool, and it also means your prompts can
name your employer's systems.

## Reporting a vulnerability

Open a GitHub security advisory rather than a public issue.
