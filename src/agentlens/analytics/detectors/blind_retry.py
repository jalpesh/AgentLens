"""Blind retry detector.

The single most expensive habit in agentic coding, and the one nobody measures.

The pattern: something didn't work, so you re-ask. But you re-ask in *different
words* without supplying any new information — no error text, no observed
output, no file reference. The model has no new evidence, so it produces a
variation on the same guess. Meanwhile the entire conversation so far is re-sent
as input on every attempt, so attempt three costs more than attempt one.

We detect it by looking for consecutive user turns that are semantically close
but carry no new hard evidence (no paths, no error strings, no code blocks, no
command output pasted in).
"""

from __future__ import annotations

import re

from ...schema import Event, Role
from .base import Detector, Evidence, Finding

_PATH = re.compile(r"[\w./\\-]+\.[a-zA-Z]{1,6}\b|/[\w./-]{4,}")
_CODEBLOCK = re.compile(r"```|\n\s{4,}\S")
_ERRORISH = re.compile(
    r"\b(error|exception|traceback|failed|stderr|exit code|assertion|"
    r"undefined|null pointer|segfault|stack trace|E\d{3,}|\bline \d+)\b",
    re.I,
)
_FRUSTRATION = re.compile(
    r"\b(still|again|not work\w*|doesn'?t work|didn'?t work|same (error|issue|problem)|"
    r"nope|no luck|try again|as i said|i told you|that'?s wrong|incorrect)\b",
    re.I,
)

_STOP = {
    "the", "a", "an", "and", "or", "but", "is", "are", "to", "of", "in", "it",
    "this", "that", "for", "with", "on", "you", "i", "please", "can", "do",
    "not", "be", "have", "has", "my", "me", "we", "your", "so", "if", "then",
}


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", text.lower()) if w not in _STOP and len(w) > 2}


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)  # Jaccard


def _carries_new_evidence(text: str) -> bool:
    """Did the user actually supply the model something it didn't have?"""
    return bool(_PATH.search(text) or _CODEBLOCK.search(text) or _ERRORISH.search(text))


class BlindRetryDetector(Detector):
    name = "blind_retry"
    title = "Re-asking without new information"
    min_usd = 0.01

    #: Below this, the two prompts are genuinely different questions.
    similarity_threshold = 0.34
    #: A frustration follow-up longer than this is probably carrying real detail
    #: even if our evidence regexes didn't recognise the shape of it.
    short_words = 25

    def run(self, events: list[Event]) -> Finding | None:
        prompts = [e for e in events if e.role is Role.USER and e.text and e.text.strip()]
        if len(prompts) < 2:
            return None

        wasted = 0.0
        tokens = 0
        evidence: list[Evidence] = []

        for prev, cur in zip(prompts, prompts[1:]):
            a, b = prev.text or "", cur.text or ""

            # A blind retry is defined by what it *lacks*, not by what it says.
            # If the follow-up brought real evidence, it isn't one — full stop.
            if _carries_new_evidence(b):
                continue

            sim = _similarity(a, b)
            frustrated = bool(_FRUSTRATION.search(b))

            # Two independent triggers, because real retries take two shapes:
            #
            #  * near-verbatim repeats  ("fix the webhook handler" ×2), caught
            #    by lexical similarity; and
            #  * frustration follow-ups ("still not working") which share almost
            #    NO vocabulary with the original ask and so score near zero on
            #    similarity — yet are the most expensive version of the habit.
            #
            # An earlier build only checked similarity and silently missed the
            # entire second class, which is the more common one.
            near_repeat = sim >= self.similarity_threshold
            blind_followup = frustrated and len(b.split()) <= self.short_words

            if not (near_repeat or blind_followup):
                continue

            # Everything spent between the two prompts bought nothing new.
            cost, toks = self.cost_between(events, prev.seq, cur.seq)
            if cost <= 0 and toks <= 0:
                continue

            wasted += cost
            tokens += toks
            evidence.append(
                Evidence(
                    session_id=cur.session_id,
                    ts=cur.ts,
                    quote=self.quote(b),
                    full_text=b,
                    kind="prompt",
                    detail=(
                        (
                            f"{int(sim * 100)}% overlap with your previous prompt"
                            if near_repeat
                            else "Reports failure without saying what failed"
                        )
                        + f' ("{self.quote(a, 60)}") — no error text, path or '
                        "output was supplied."
                    ),
                    cost_usd=round(cost, 6),
                    tokens=toks,
                )
            )

        if not evidence:
            return None

        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "The whole conversation is re-sent as input on every turn, so a second "
                    "attempt costs more than the first — and with no new evidence the model "
                    "just repeats the same guess in different words."
                ),
                prescription=(
                    'Say what "wrong" means before you re-send: the exact command you ran, '
                    "the output you observed, and the output you expected."
                ),
                wasted_usd=wasted,
                wasted_tokens=tokens,
                occurrences=len(evidence),
                evidence=evidence,
            )
        )
