"""Historical Prompt Pattern Mining — what you keep telling every agent.

Per-session habits (loops, re-asking, vague prompts, model mismatch) live in
the detectors, and context dumping lives in Context Efficiency. This module
adds the two patterns that only show up across history:

1. **Repeated instructions** — the same directive typed into 3+ sessions
   ("use the existing repository layer", "don't create another service").
   That is a project rule living in your head instead of the agent's
   instructions file.
2. **Repeated corrections** — prompts that push back on what the agent just
   did ("no, reuse the existing API"). Sessions with several of them point
   at a constraint the agent wasn't given up front.

Everything is heuristic and labelled so. Prompt text stays local: examples
are the person's own sentences, shown only in their own dashboard.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..schema import Event, Provider, Role
from .confidence import Confidence, derived, direct, heuristic, temper, unavailable
from .context_efficiency import _input_price

BYTES_PER_TOKEN = 4
#: Seen in fewer sessions than this, a repeat is coincidence, not a habit.
MIN_SESSIONS = 3
#: A session with this many corrections was probably missing a constraint.
CORRECTION_HEAVY = 3

#: Where each agent reads standing project instructions.
INSTRUCTION_FILE = {
    Provider.CLAUDE_CODE: "CLAUDE.md",
    Provider.CODEX: "AGENTS.md",
    Provider.OPENCODE: "AGENTS.md",
    Provider.JUNIE: ".junie/guidelines.md",
    Provider.GEMINI_CLI: "GEMINI.md",
    Provider.COPILOT_CLI: ".github/copilot-instructions.md",
    Provider.GOOSE: ".goosehints",
}

_HARNESS = re.compile(r"^\s*<")
_REPLAY = re.compile(r"^\[\d+\] (user|assistant):", re.M)
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
#: A sentence must read as a standing rule, not a one-off request.
_DIRECTIVE = re.compile(
    r"\b(don'?t|dont|do not|never|always|must|should|make sure|avoid|prefer|only|"
    r"follow|reuse|use (the )?existing|instead of|no need to|please)\b",
    re.I,
)
_CORRECTION = re.compile(
    r"^\s*(no\b|nope\b|don'?t\b|dont\b|do not\b|that'?s (wrong|not)|thats (wrong|not)|"
    r"not what|i said|i asked|wrong\b|why did you|you (didn'?t|forgot|missed)|"
    r"instead\b|undo\b|revert\b)",
    re.I,
)


def _norm(sentence: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", sentence.lower()).split())


def _prompts(evs: list[Event]):
    for e in evs:
        if e.role is Role.USER and e.text and e.text.strip():
            if not _HARNESS.match(e.text) and not _REPLAY.search(e.text):
                yield e


@dataclass(slots=True)
class Pattern:
    kind: str
    title: str
    problem: str
    recommendation: str
    confidence: Confidence
    sessions: set[str] = field(default_factory=set)
    occurrences: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    examples: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "problem": self.problem,
            "recommendation": self.recommendation,
            "confidence": self.confidence.value,
            "sessions": len(self.sessions),
            "occurrences": self.occurrences,
            "tokens": self.tokens,
            "cost_usd": round(self.cost_usd, 4),
            "examples": self.examples[:3],
            "targets": self.targets,
        }


def prompt_patterns(sessions_events: list[list[Event]]) -> dict:
    sessions = [evs for evs in sessions_events if evs and evs[0].provider is not Provider.MANUAL]
    # normalized sentence -> session -> (original text, times seen there)
    seen: dict[str, dict[str, list]] = defaultdict(dict)
    provider_of: dict[str, Provider] = {}
    price_of: dict[str, float] = {}
    corrections: dict[str, list[str]] = defaultdict(list)
    n_prompts = 0

    for evs in sessions:
        sid = evs[0].session_id
        provider_of[sid] = evs[0].provider
        price_of[sid] = _input_price(evs)
        agent_acted = False
        prompts = set(id(p) for p in _prompts(evs))
        for e in evs:
            if e.role is Role.TOOL_CALL:
                agent_acted = True
            if id(e) not in prompts:
                continue
            n_prompts += 1
            # A correction pushes back on something the agent just did.
            if agent_acted and _CORRECTION.match(e.text):
                corrections[sid].append(" ".join(e.text.split())[:140])
            agent_acted = False
            for s in _SPLIT.split(e.text):
                n = _norm(s)
                if 5 <= len(n.split()) <= 60 and _DIRECTIVE.search(s):
                    hit = seen[n].setdefault(sid, [s.strip(), 0])
                    hit[1] += 1

    # Sentences that always travel together (one pasted rules block) are one
    # habit, not ten: group them by the exact set of sessions they appear in.
    blocks: dict[frozenset, list[str]] = defaultdict(list)
    for n, by_sid in seen.items():
        if len(by_sid) >= MIN_SESSIONS:
            blocks[frozenset(by_sid)].append(n)

    patterns: list[Pattern] = []
    for sids, sentences in blocks.items():
        targets = sorted({INSTRUCTION_FILE[provider_of[s]] for s in sids
                          if provider_of[s] in INSTRUCTION_FILE})
        p = Pattern(
            "repeated_instruction",
            seen[sentences[0]][next(iter(sids))][0][:120],
            f"The same instruction was typed into {len(sids)} separate sessions.",
            "Move it into the project's standing agent instructions"
            + (f" ({', '.join(targets)})" if targets else "")
            + " so every session starts with it. If it's already there, the agent isn't "
            "following it: make it shorter, more specific and nearer the top.",
            temper(Confidence.MEDIUM, len(sids)),
            sessions=set(sids),
            targets=targets,
        )
        for n in sentences:
            hits = seen[n]
            total = sum(c for _, c in hits.values())
            p.occurrences = max(p.occurrences, total)
            p.examples.extend(text[:160] for text, _ in list(hits.values())[:1])
            # Every send after the first could have come from the file instead.
            t = max(1, len(n) // BYTES_PER_TOKEN)
            p.tokens += t * (total - 1)
            p.cost_usd += t * sum(c * price_of[s] for s, (_, c) in hits.items()) * (total - 1) / total
        patterns.append(p)

    heavy = {sid: c for sid, c in corrections.items() if len(c) >= CORRECTION_HEAVY}
    if heavy:
        patterns.append(Pattern(
            "repeated_correction",
            "Sessions needing repeated corrections",
            f"{len(heavy)} session(s) had {CORRECTION_HEAVY}+ prompts pushing back on what "
            "the agent had just done — often a constraint it wasn't told up front.",
            "Read the corrections below; any that recur belong in the project's agent "
            "instructions or the task's acceptance criteria.",
            temper(Confidence.LOW, len(heavy)),
            sessions=set(heavy),
            occurrences=sum(len(c) for c in heavy.values()),
            examples=[c for cs in heavy.values() for c in cs][:3],
        ))

    patterns.sort(key=lambda p: (-len(p.sessions), -p.tokens))
    n = len(sessions)
    if not n_prompts:
        measures = [unavailable("Prompts analysed", "No typed prompts in scope.")]
    else:
        n_corr = sum(len(c) for c in corrections.values())
        rep = [p for p in patterns if p.kind == "repeated_instruction"]
        measures = [
            direct("Prompts analysed", n_prompts,
                   "User prompts, excluding harness-injected messages and transcript replays.",
                   sample_size=n),
            derived("Prompts/session", round(n_prompts / n, 1), "Prompts ÷ sessions.",
                    sample_size=n),
            heuristic("Correction rate", round(100.0 * n_corr / n_prompts, 1),
                      "Prompts that open by pushing back (\"no\", \"don't\", \"I said\"…) "
                      "after the agent acted, ÷ all prompts.",
                      temper(Confidence.LOW, n_prompts, minimum=30), "pct", n_prompts),
            heuristic("Repeated instructions", len(rep),
                      f"Directive sentences typed into {MIN_SESSIONS}+ sessions, grouped "
                      "when they always appear together.",
                      temper(Confidence.MEDIUM, n), "", n),
            heuristic("Tokens re-typing instructions", sum(p.tokens for p in rep),
                      "Every send after the first, as typed. A floor: each copy is also "
                      "re-sent on later turns.", temper(Confidence.MEDIUM, n), "tokens", n),
        ]
    return {"measures": measures, "patterns": [p.to_dict() for p in patterns]}
