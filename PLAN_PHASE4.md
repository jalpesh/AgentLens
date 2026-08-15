# AgentLens — Phase 4 Plan

Source: frame-by-frame review of the original dashboard demo video
(`WhatsApp Video 2026-08-13 at 11.56.06.mp4`, 71 frames extracted at 2s
intervals, all 7 tabs walked). Compared against the current v0.3.0 build.
This closes the Tier 1 + Tier 2 gaps identified in that review. Tier 3 items
(live file-writes from the dashboard) are intentional differences, not gaps,
and are out of scope.

## Rules carried over from Phase 3 (unchanged)
1. Clean-room boundary — never copy code/data from the original repo.
2. No real session data in fixtures — synthetic only.
3. Every finding: user's own words + dollar figure + concrete rewrite.
4. Clean/converged sessions must stay silent — no crying wolf.
5. `ruff check` + `pytest -q` clean before any task is marked done.
6. Never name an external package that hasn't been verified to exist.

## Verified this session (WebSearch, 2026-08-15)
- `ast-grep` (`@ast-grep/cli` on npm, `ast-grep/ast-grep` on GitHub) — real,
  well-established structural search/lint/rewrite tool. Safe to name with a
  real install command.
- `@colbymchenry/codegraph` — real npm package + GitHub repo, MCP server,
  local code-graph index. Safe to name.
- `codebase-memory-mcp` (DeusData) — real npm package + GitHub repo, code
  intelligence MCP server. Safe to name.
- `code-review-graph` — **ambiguous**. At least four unrelated packages/repos
  share this exact name (juspay, tirth8205, n24q02m, a PyPI package), from
  different authors, doing similar-but-different things. There is no single
  canonical package this name resolves to. Per Rule 6, this stays in the
  `unverified_category` bucket (placeholder command, not a real npx line) —
  the ambiguity itself is the reason, not just "couldn't find it."

## A. Session drill-down (store + payload + frontend)
Currently a session shows totals only. Original shows the turn-by-turn story.

- `Store`: add a per-event-ordered accessor for a single session (already
  have `iter_sessions`/scoping — extend to expose per-session Event list with
  running totals attached, not just aggregates).
- New `analytics/session_detail.py`: `build_session_detail(events) -> dict`
  producing:
  - `cost_curve`: `[{seq, cumulative_usd}]` — one point per LLM call.
  - `context_fill_curve`: `[{seq, pct}]` — reuses the existing context-fill
    clamp logic from `detectors/context.py`, don't reimplement.
  - `models_used`: per-model calls/fresh_in/cached_in/out/cost table, scoped
    to this session (we already compute this session-wide in `rollups.py` —
    factor out the per-session slice instead of duplicating).
  - `prompts`: every user prompt in the session with `est_tokens` and a
    flag for whether it has Prompt-Lab-round-trip evidence (`kind == "prompt"`).
- `web/server.py`: extend `/api/session/<id>` (new endpoint, GET only) to
  return this payload.
- `index.html`: session detail pane gets two new sparkline/line charts
  (reuse the existing chart-drawing helper already used for Daily Token
  Usage — don't add a new charting dependency) + a models table + the full
  prompt list, each prompt row wired to the existing "Open in Lab" handler.

## B. Loop attempt detail
Currently a loop finding shows occurrence count + total cost. Original opens
into a per-attempt breakdown: first ask vs. each repeat, token/cost/call
delta, and (in the original) a citation of the specific file+line that
explains the repeat.

- `LoopDetector`/`SameTargetChurnDetector`/`CrossSessionLoopDetector`: extend
  `Finding` (or a new `Finding.attempts: list[Attempt]` field) with per-
  occurrence detail: `{when, prompt_excerpt, tokens, cost_usd, calls}` for
  the first attempt and every repeat. This is derivable entirely from
  events already in the store — no new data source needed.
- Line-citation ("detected because you referenced X, and Y no longer
  matches") is the one piece of the original that reads the actual file
  content on disk at analysis time, not just event metadata. That's a genuine
  conflict with the no-filesystem-scanning rule we set in Phase 3 for
  fingerprinting. See open question 1 below — needs your call before I build
  it either way.

## C. Prompt Lab additions
- "Same prompt across models": re-run the existing scoring/tokenization
  against every model family actually present in the user's ingested
  history (`Store.projects()`/model list already exists), cheapest first.
  Pure reuse of `analytics/prompt_score.py` + `pricing.py` — no new logic,
  just a loop over known models.
- "Token budget by content type" (prose vs. code vs. paths) — extend the
  existing prompt tokenizer breakdown (`prompt_score.py` already classifies
  content for the efficiency score; expose the breakdown instead of just the
  final number).
- "Reclaimable" tokens on a single prompt (currently only shown as an
  aggregate on Overview) — surface the existing auto-clean-up delta per
  prompt instead of only in aggregate.

## D. Agents & Skills — reinstate the generic bucket
Add a second, explicitly-labeled section alongside the existing
trigger-only suggestions:

> "Generic best practices — not derived from your data. Useful for most
> projects regardless of what we found."

Contents (all verified above): `ast-grep` for structural search, `codegraph`
for a local code-graph MCP, `codebase-memory-mcp` for persistent code
memory. Each gets its own real install command. This section is visually
and structurally separate from the triggered suggestions (different
heading, different card style) so the "no suggestion without a trigger" test
still holds for the triggered section — it just gets a sibling, not a
replacement. Existing tests (`test_every_suggestion_names_its_trigger` etc.)
stay green because they'll scope to the triggered list only; new tests
assert the generic section is present, correctly labeled, and never
inherits a fake `triggered_by`.

## E. Engineering tab metrics
Add four top tiles, computed from existing tool-call events — no new
detector needed, this is a rollup:
- Discovery Share % — (search/grep tool calls) / (total agent steps).
- Searches & Greps — count + breakdown (project search vs. shell grep).
- Files Opened — distinct file count + "opened N+ times" repeat count.
- Loop Cost $ — sum of `wasted_usd` across all loop-family findings for the
  current scope (already computed per-finding, just needs summing here).

Tie the `ast-grep`/`codegraph` mentions from section D into this tab too
*when triggered* (e.g., discovery share above a threshold) — same triggered
suggestion, just also linked from here since that's where the original
puts it. No duplicate logic, just a second render location for the same
`Suggestion` objects.

## F. History tab trend charts
- "Prompt size over time" and "Prompt quality over time" — both are trivial
  once `analytics/prompt_score.py` runs across full history ordered by
  timestamp; we already compute the per-prompt score, this is just not
  plotted yet.
- "Heaviest prompts" table — sort existing per-prompt records by token
  count, cap at top 10, link each to Prompt Lab.

## Test plan
- `test_session_detail.py` — cost curve monotonic non-decreasing, context
  fill clamped ≤100%, models_used sums match session total, converged
  session still produces a full curve (this isn't a waste finding, so no
  suppression logic needed here — just correctness).
- `test_loops.py` additions — attempts list ordered, first attempt cost <
  total finding cost (sanity), synthetic fixture with 3 repeats produces 3
  attempt records.
- `test_skills.py` additions — generic section never sets `triggered_by`,
  triggered section never includes the three generic-only packages unless
  also organically triggered, both sections present given a rich fixture.
- `test_rewrite.py` / prompt-lab additions — same-prompt-across-models
  output ordered cheapest-first, token-budget-by-type sums to the prompt
  total.
- Full `ruff check` + `pytest -q` before packaging.

## Open questions before I start

1. **Loop line-citation.** The original reads actual file lines off disk to
   explain a loop ("detected because cicd/azure-pipelines.yml lines 14–16
   still say X"). We deliberately don't scan the filesystem elsewhere in
   AgentLens (fingerprinting is event-only, by design, for privacy/offline
   guarantees, and so it works on a machine that doesn't have the repo
   checked out). Do you want loop citations to: (a) stay metadata-only —
   cite the target file *path* and the tool-call diff/error text already in
   the session log, no disk reads, weaker but consistent with the rest of
   the tool; or (b) add a narrow, explicit opt-in disk read — only the
   specific file, only when generating this one view, never cached to
   SQLite — to match the original's line-citation exactly, which is more
   impressive but is a new category of access AgentLens hasn't needed before?
2. **Sequencing.** All of A–F in one pass again, like Phase 3, or split into
   waves so you can react to the session drill-down (the highest-impact
   piece) before I build the rest?
