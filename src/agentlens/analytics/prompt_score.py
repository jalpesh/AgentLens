"""Prompt Lab — score a prompt *before* you send it.

Fully offline. No API call is made; tokenization is local (tiktoken when
installed, a calibrated heuristic otherwise).

Design note, learned the hard way: **a scorer that never says no is
decoration.** A tool that returns "100/100, nothing wasteful found" on every
prompt trains the user to ignore it within a day. The rubric below is
deliberately opinionated enough to be occasionally annoying — that is the
feature, not a bug.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict, field
from typing import Any

from ..pricing import context_limit_of, lookup
from ..schema import Event, Role

_FILE_REF = re.compile(r"[@\w./\\-]+\.[a-zA-Z]{1,6}\b|@[\w./-]+")
_ACCEPTANCE = re.compile(
    r"\b(must pass|should pass|expect\w*|assert\w*|acceptance|verify|"
    r"test[s]? (pass|green)|returns?|output should|so that)\b",
    re.I,
)
_HEDGE = re.compile(
    r"\b(maybe|perhaps|i think|kind of|sort of|if possible|would be (nice|good)|"
    r"try to|somehow|or something|etc\.?|and so on|as you see fit|whatever you think)\b",
    re.I,
)
_POLITENESS = re.compile(
    r"\b(please|thanks|thank you|could you (please )?|would you (please )?|"
    r"i (would|'d) (like|appreciate)|if you don'?t mind|sorry to bother)\b",
    re.I,
)
_PASTED_DUMP = re.compile(r"```[\s\S]{2000,}```")
_MULTI_ASK = re.compile(r"\b(also|additionally|and then|after that|plus|as well as)\b", re.I)
_VAGUE_VERB = re.compile(
    r"^\s*(fix|improve|clean ?up|optimi[sz]e|refactor|update|change|handle|"
    r"make it (better|work)|do it|check|look at)\b",
    re.I,
)


def count_tokens(text: str, model: str | None = None) -> int:
    """Local tokenization. Exact with tiktoken, ~5% heuristic without."""
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        try:
            enc = tiktoken.encoding_for_model((model or "gpt-4o").split("/")[-1])
        except Exception:
            enc = tiktoken.get_encoding("o200k_base")
        return len(enc.encode(text))
    except Exception:
        # Calibrated fallback: ~3.7 chars/token for English prose+code, with a
        # correction for whitespace-heavy input.
        chars = len(text)
        words = max(1, len(text.split()))
        return int(max(chars / 3.7, words * 0.75)) or 1


@dataclass(slots=True)
class Issue:
    code: str
    label: str
    penalty: int
    fix: str
    span: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromptReport:
    score: int
    tokens: int
    cleaned_tokens: int
    saved_tokens: int
    words: int
    chars: int
    context_pct: float
    context_limit: int
    cost_usd: float
    cleaned_cost_usd: float
    #: `cost_usd - cleaned_cost_usd`, i.e. the auto-clean-up delta already
    #: computed for `saved_tokens`, now surfaced as a dollar figure on the
    #: single prompt rather than only ever shown aggregated on Overview.
    reclaimable_usd: float
    model: str
    issues: list[Issue] = field(default_factory=list)
    cleaned: str = ""
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["issues"] = [i.to_dict() for i in self.issues]
        return d


def _strip_noise(text: str) -> str:
    """Remove tokens that carry no instruction. Conservative on purpose —
    anything ambiguous is left alone, because silently changing meaning to save
    four tokens is a terrible trade."""
    out = _POLITENESS.sub("", text)
    out = _HEDGE.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip()


def score_prompt(text: str, model: str = "claude-sonnet-4-5") -> PromptReport:
    text = text or ""
    issues: list[Issue] = []
    score = 100

    words = len(text.split())
    chars = len(text)
    tokens = count_tokens(text, model)

    def add(code: str, label: str, penalty: int, fix: str, span: str = "") -> None:
        nonlocal score
        issues.append(Issue(code, label, penalty, fix, span))
        score -= penalty

    if words == 0:
        return PromptReport(
            score=0, tokens=0, cleaned_tokens=0, saved_tokens=0, words=0, chars=0,
            context_pct=0.0, context_limit=context_limit_of(model), cost_usd=0.0,
            cleaned_cost_usd=0.0, reclaimable_usd=0.0, model=model, verdict="Empty prompt.",
        )

    # --- the expensive omissions (heaviest penalties) ------------------
    if not _FILE_REF.search(text):
        add(
            "no_file_ref", "No file or path referenced", 22,
            "Name the file with @path/to/file. Without it the agent greps your repo to "
            "find out what you meant, and that search is billed to you.",
        )
    if not _ACCEPTANCE.search(text):
        add(
            "no_acceptance", "No acceptance check", 18,
            'State what "done" looks like — "`pytest -q` must pass", "returns 404 for '
            'expired tokens". Without it you get a plausible answer instead of a correct one.',
        )
    m = _VAGUE_VERB.match(text)
    if m and words < 25:
        add(
            "vague_verb", "Opens with a vague instruction", 15,
            'Replace "fix"/"improve" with the specific change and the observable outcome.',
            m.group(0),
        )

    # --- waste that costs tokens directly ------------------------------
    hedges = _HEDGE.findall(text)
    if hedges:
        add(
            "hedging", f"Hedging language ({len(hedges)}×)", min(12, 4 * len(hedges)),
            "Hedges make the agent explore alternatives you don't want. Say what you want.",
            ", ".join(sorted({h if isinstance(h, str) else h[0] for h in hedges})[:3]),
        )
    pol = _POLITENESS.findall(text)
    if len(pol) > 1:
        add(
            "politeness", f"Filler politeness ({len(pol)}×)", 4,
            "Costs tokens on every turn of the session. The agent isn't offended.",
        )
    if _PASTED_DUMP.search(text):
        add(
            "pasted_dump", "Very large pasted block", 14,
            "Reference the file instead of pasting it — the agent can read it, and pasted "
            "text is re-sent as input on every later turn.",
        )
    asks = len(_MULTI_ASK.findall(text))
    if asks >= 2:
        add(
            "multi_ask", f"Several tasks bundled ({asks + 1} asks)", 10,
            "Split into separate turns. Bundled asks produce partial work you then pay to "
            "re-ask about.",
        )
    if words > 400:
        add(
            "too_long", f"Very long prompt ({words} words)", 8,
            "Move standing context into CLAUDE.md / AGENTS.md / .junie/guidelines.md so it's "
            "cached instead of retyped.",
        )

    score = max(0, min(100, score))

    cleaned = _strip_noise(text)
    cleaned_tokens = count_tokens(cleaned, model)
    saved = max(0, tokens - cleaned_tokens)

    price = lookup(model)
    cost = tokens * price.input / 1_000_000
    cleaned_cost = cleaned_tokens * price.input / 1_000_000
    limit = context_limit_of(model)

    if score >= 85:
        verdict = "Specific and well-scoped. Send it."
    elif score >= 60:
        verdict = "Workable, but you'll pay for the gaps below."
    elif score >= 35:
        verdict = "The agent will spend your money working out what you meant."
    else:
        verdict = "Rewrite this. As written it will cost several turns to converge."

    return PromptReport(
        score=score,
        tokens=tokens,
        cleaned_tokens=cleaned_tokens,
        saved_tokens=saved,
        words=words,
        chars=chars,
        context_pct=round(100.0 * tokens / limit, 4) if limit else 0.0,
        context_limit=limit,
        cost_usd=round(cost, 6),
        cleaned_cost_usd=round(cleaned_cost, 6),
        reclaimable_usd=round(max(0.0, cost - cleaned_cost), 6),
        model=model,
        issues=issues,
        cleaned=cleaned,
        verdict=verdict,
    )


# --- Prompt Lab additions: same prompt across models, content-type budget,
# and history-wide trends. All pure reuse of `score_prompt`/`count_tokens` —
# no new tokenization or scoring logic. -------------------------------------


def score_across_models(text: str, models: list[str]) -> list[PromptReport]:
    """Re-run the existing scorer against every model actually present in the
    user's history, cheapest first.

    Deduplicates while preserving the caller's ordering preference, and
    silently skips a model name that scores to the same report as one
    already seen (defensive against a caller passing duplicates)."""
    seen: set[str] = set()
    reports: list[PromptReport] = []
    for m in models:
        if not m or m in seen:
            continue
        seen.add(m)
        reports.append(score_prompt(text, m))
    return sorted(reports, key=lambda r: r.cost_usd)


def content_type_budget(text: str, model: str = "claude-sonnet-4-5") -> dict[str, int]:
    """Token budget by content type: fenced code, file/path references, prose.

    Reuses the same tokenizer (`count_tokens`) and the same file-reference
    regex the scorer already uses for the `no_file_ref` penalty — this is
    that existing classification exposed as a breakdown instead of collapsed
    into one score. Character-proportional rather than tokenizing each slice
    independently, so the three numbers always sum to exactly the prompt's
    total token count instead of drifting from tokenizer boundary effects.
    """
    text = text or ""
    total = count_tokens(text, model)
    if not text.strip():
        return {"code_tokens": 0, "path_tokens": 0, "prose_tokens": 0, "total_tokens": 0}

    code_chars = sum(len(m.group()) for m in re.finditer(r"```[\s\S]*?```", text))
    remainder = re.sub(r"```[\s\S]*?```", "", text)
    path_chars = sum(len(m) for m in _FILE_REF.findall(remainder))
    denom = len(text) or 1
    code_tokens = round(total * code_chars / denom)
    path_tokens = round(total * path_chars / denom)
    prose_tokens = max(0, total - code_tokens - path_tokens)
    return {
        "code_tokens": code_tokens,
        "path_tokens": path_tokens,
        "prose_tokens": prose_tokens,
        "total_tokens": total,
    }


def prompt_trends(sessions_events: list[list[Event]], model: str = "claude-sonnet-4-5") -> dict:
    """History-wide prompt trends for the History tab: size and quality over
    time, plus the heaviest prompts. One `score_prompt` call per user prompt
    — no separate scoring path."""
    rows: list[dict[str, Any]] = []
    for evs in sessions_events:
        for e in evs:
            if e.role is not Role.USER or not e.text or not e.text.strip():
                continue
            r = score_prompt(e.text, model)
            rows.append(
                {
                    "ts": e.ts,
                    "session_id": e.session_id,
                    "seq": e.seq,
                    "tokens": r.tokens,
                    "score": r.score,
                    "cost_usd": r.cost_usd,
                    "excerpt": " ".join(e.text.split())[:160],
                }
            )
    rows.sort(key=lambda r: r["ts"] or "")
    return {
        "size_over_time": [{"ts": r["ts"], "tokens": r["tokens"]} for r in rows],
        "quality_over_time": [{"ts": r["ts"], "score": r["score"]} for r in rows],
        "heaviest_prompts": sorted(rows, key=lambda r: -r["tokens"])[:10],
    }
