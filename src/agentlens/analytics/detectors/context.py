"""Context-shape detectors: compaction thrash, vague instructions, model mismatch."""

from __future__ import annotations

import re

from ...pricing import canonical, cost_of, lookup
from ...schema import Event, Role, ToolKind
from .base import Detector, Evidence, Finding

_FILE_REF = re.compile(r"[@\w./\\-]+\.[a-zA-Z]{1,6}\b|@[\w./-]+")
_ACCEPTANCE = re.compile(
    r"\b(must pass|should pass|so that|expect\w*|assert\w*|acceptance|verify|"
    r"until .* (pass|work)|test[s]? (pass|green)|returns?|output should)\b",
    re.I,
)
_ERROR_EVIDENCE = re.compile(
    r"\b(error|exception|traceback|failed|stderr|exit code|assertion\w*|"
    r"stack trace|line \d+|got \d{3}|expected)\b",
    re.I,
)
_LOG_PASTE = re.compile(r"^\s*\d{2}-\d{2} \d{2}:\d{2}:\d{2}|^\s*\d{4}-\d{2}-\d{2}[ T]\d{2}:", re.M)
#: A "mechanical" turn is small; more tool calls than this is real agent work.
MECHANICAL_MAX_CALLS = 10
_VAGUE_OPENER = re.compile(
    r"^\s*(fix|make it work|it'?s broken|improve|clean ?up|optimi[sz]e|refactor|"
    r"update|change|handle|do it|continue|carry on|go ahead|check|look at|see)\b",
    re.I,
)


class CompactionThrashDetector(Detector):
    """Compaction is a safety net, not a strategy.

    When the window fills, the agent summarises history to survive — and pays
    to re-read that summary forever after, while having lost detail.
    """

    name = "compaction_thrash"
    title = "Sessions running until the context window fills"
    min_usd = 0.0

    def run(self, events: list[Event]) -> Finding | None:
        compactions = [e for e in events if e.role is Role.COMPACTION]
        peak = max(
            (
                min(100.0, 100.0 * e.context_tokens / e.context_limit)
                for e in events
                if e.context_tokens and e.context_limit
            ),
            default=0.0,
        )
        if not compactions and peak < 75:
            return None

        session_cost = sum(e.cost_usd for e in events)
        sid = events[0].session_id if events else ""
        estimate = _summary_resend_cost(events, compactions)

        detail = (
            f"{len(compactions)} compaction(s); peak context fill {peak:.0f}%."
            if compactions
            else f"Peak context fill {peak:.0f}% — one more long turn triggers compaction."
        )

        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "Compaction summarises history to survive, then re-sends that summary on "
                    "every later turn — you pay for it forever and lose detail you can't get "
                    "back."
                ),
                prescription=(
                    "Start a fresh session when the topic changes. Carry forward three lines "
                    "of state, not the whole transcript."
                ),
                wasted_usd=estimate,
                wasted_tokens=0,
                occurrences=max(1, len(compactions)),
                evidence=[
                    Evidence(
                        session_id=sid,
                        ts=(compactions[0].ts if compactions else events[0].ts),
                        quote=detail,
                        detail=f"Session spend ${session_cost:.2f}.",
                        cost_usd=round(estimate, 6),
                    )
                ],
            )
        )


def _summary_resend_cost(events: list[Event], compactions: list[Event]) -> float:
    """What carrying compaction summaries cost versus a fresh session.

    The summary isn't logged, so its size is the first post-compaction turn's
    context minus the session's opening context (system prompt and tools, which
    a fresh session pays too). That excess, re-sent on every turn until the
    next compaction, is priced at the turn's cache-read rate. Replaces a flat
    10-30% of session spend that had no measurement behind it.
    """
    turns = [e for e in events if e.role is Role.ASSISTANT and e.usage]
    if not turns or not compactions:
        return 0.0

    def ctx(e: Event) -> int:
        return e.usage.input + e.usage.cache_read + e.usage.cache_write

    base = ctx(turns[0])
    cuts = sorted(c.seq for c in compactions)
    cost = 0.0
    for i, c in enumerate(cuts):
        end = cuts[i + 1] if i + 1 < len(cuts) else float("inf")
        after = [t for t in turns if c < t.seq < end]
        if not after:
            continue
        summary = max(0, ctx(after[0]) - base)
        for t in after[1:]:
            p = lookup(t.model)
            cost += summary * (p.cache_read or p.input) / 1_000_000
    return min(cost, sum(e.cost_usd for e in events))


class VagueInstructionDetector(Detector):
    """Prompts with no file reference and no acceptance criterion.

    A vague prompt makes the agent go find out what you meant — reading files,
    grepping, guessing. That discovery is billed to you, and it is avoidable.
    """

    name = "vague_instruction"
    title = "Vague prompts that make the agent go hunting"
    min_usd = 0.01
    min_words = 4

    def run(self, events: list[Event]) -> Finding | None:
        prompts = [e for e in events if e.role is Role.USER and e.text and e.text.strip()]
        if not prompts:
            return None

        wasted = 0.0
        tokens = 0
        evidence: list[Evidence] = []

        for i, p in enumerate(prompts):
            text = p.text or ""
            words = text.split()
            if len(words) < self.min_words:
                continue
            has_file = bool(_FILE_REF.search(text))
            has_accept = bool(_ACCEPTANCE.search(text))
            vague_open = bool(_VAGUE_OPENER.match(text))
            if has_file and has_accept:
                continue
            if not (vague_open or (not has_file and not has_accept and len(words) < 25)):
                continue

            nxt = prompts[i + 1].seq if i + 1 < len(prompts) else max(e.seq for e in events)
            # Only the exploration the agent had to do counts: search/read calls
            # before it could act.
            explore = [
                e
                for e in events
                if p.seq < e.seq <= nxt
                and e.role is Role.TOOL_CALL
                and e.tool
                and e.tool.kind in (ToolKind.SEARCH, ToolKind.READ)
            ]
            if len(explore) < 3:
                continue

            cost, toks = self.cost_between(events, p.seq, nxt)
            # Attribute the share of the turn spent orienting rather than doing.
            share = min(0.5, 0.08 * len(explore))
            wasted += cost * share
            tokens += int(toks * share)

            missing = []
            if not has_file:
                missing.append("no file reference")
            if not has_accept:
                missing.append("no acceptance check")

            evidence.append(
                Evidence(
                    session_id=p.session_id,
                    ts=p.ts,
                    quote=self.quote(text),
                    full_text=text,
                    kind="prompt",
                    detail=(
                        f"{', '.join(missing)} — the agent ran {len(explore)} "
                        f"search/read calls before it could act."
                    ),
                    cost_usd=round(cost * share, 6),
                    tokens=int(toks * share),
                )
            )

        if not evidence:
            return None
        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "With no file and no acceptance criterion the agent has to discover what "
                    "you meant, and that discovery is billed to you at full rate."
                ),
                prescription=(
                    "Name the file, the expected behaviour, and the check that proves it: "
                    '"In @src/auth.py, make login() reject expired tokens; '
                    '`pytest tests/test_auth.py` must pass."'
                ),
                wasted_usd=wasted,
                wasted_tokens=tokens,
                occurrences=len(evidence),
                evidence=evidence,
            )
        )


class ModelMismatchDetector(Detector):
    """Premium model on mechanical work.

    Renaming a variable does not need the most expensive model you have access
    to. This finds turns where a frontier model did trivially mechanical work.
    """

    name = "model_mismatch"
    title = "Expensive model on mechanical work"
    min_usd = 0.02

    #: Models we treat as premium by their output rate ($/1M).
    premium_output_rate = 20.0
    cheap_alternative = {
        "claude": "claude-haiku-4-5",
        "gpt": "gpt-5-mini",
        "gemini": "gemini-3-flash",
    }

    def run(self, events: list[Event]) -> Finding | None:
        prompts = [e for e in events if e.role is Role.USER and e.text]
        if not prompts:
            return None

        wasted = 0.0
        evidence: list[Evidence] = []

        for i, p in enumerate(prompts):
            text = p.text or ""
            words = text.split()
            if len(words) > 30:
                continue  # a long, specified prompt is probably real work
            # Debugging is exactly what you want the expensive model for. If the
            # user supplied an error, a traceback or an acceptance criterion,
            # this is reasoning work — never flag it.
            if _ERROR_EVIDENCE.search(text) or _ACCEPTANCE.search(text):
                continue
            # Harness-injected messages (<task-notification>…) aren't the
            # user's prompt, and pasted logs are debugging input.
            if text.lstrip().startswith("<") or _LOG_PASTE.search(text):
                continue
            nxt = prompts[i + 1].seq if i + 1 < len(prompts) else max(e.seq for e in events)
            turn = [e for e in events if p.seq < e.seq <= nxt]

            edits = sum(
                1 for e in turn if e.role is Role.TOOL_CALL and e.tool
                and e.tool.kind is ToolKind.EDIT
            )
            reasoning_needed = any(
                e.role is Role.TOOL_CALL and e.tool and e.tool.kind in
                (ToolKind.TASK, ToolKind.TEST) for e in turn
            )
            calls = sum(1 for e in turn if e.role is Role.TOOL_CALL)
            # Many tool calls = investigation or agentic work, which is what the
            # expensive model is for, however short the prompt was.
            if reasoning_needed or edits > 3 or calls > MECHANICAL_MAX_CALLS:
                continue

            models = {e.model for e in turn if e.model and e.usage}
            cost = sum(e.cost_usd for e in turn)
            if cost < 0.01:
                continue

            for m in models:
                price = lookup(m)
                if price.output < self.premium_output_rate:
                    continue
                alt = next(
                    (v for k, v in self.cheap_alternative.items() if k in canonical(m)),
                    None,
                )
                if not alt:
                    continue
                alt_price = lookup(alt)
                if alt_price.output <= 0:
                    continue
                # A turn whose context won't fit the cheaper model couldn't
                # have run there at all.
                used = [e for e in turn if e.usage]
                if any(e.usage.input + e.usage.cache_read + e.usage.cache_write
                       > alt_price.context_limit for e in used):
                    continue
                # Reprice the turn's real token mix, not output price alone —
                # most of a turn's cost is usually re-sent context.
                saving = max(0.0, cost - sum(cost_of(alt, e.usage) for e in used))
                if saving <= 0:
                    continue
                wasted += saving
                evidence.append(
                    Evidence(
                        session_id=p.session_id,
                        ts=p.ts,
                        quote=self.quote(p.text),
                        full_text=p.text,
                        kind="prompt",
                        detail=(
                            f"Ran on {canonical(m)} (${price.output:.0f}/1M out) for "
                            f"{edits} edit(s) and no test or sub-task. "
                            f"{alt} would have cost ~${cost - saving:.3f} instead of "
                            f"${cost:.3f}."
                        ),
                        cost_usd=round(saving, 6),
                    )
                )
                break

        if not evidence:
            return None
        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "Frontier-model pricing buys reasoning depth. Mechanical edits don't use "
                    "it, so you pay the premium for nothing."
                ),
                prescription=(
                    "Route mechanical work — renames, formatting, boilerplate, single-file "
                    "edits with a clear spec — to a cheap model. Keep the expensive one for "
                    "design and debugging."
                ),
                wasted_usd=wasted,
                wasted_tokens=0,
                occurrences=len(evidence),
                evidence=evidence,
            )
        )
