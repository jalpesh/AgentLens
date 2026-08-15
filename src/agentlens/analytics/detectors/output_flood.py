"""Output flood detector.

A command that dumps 200 KB of output doesn't cost you once. It lands in the
context window verbatim and is re-sent as input tokens on *every subsequent
turn* of that session. One unpiped `npm ls` or `git log` early in a long session
can quietly dominate its whole bill.

This is the highest-leverage, easiest-to-fix waste in the entire dataset, which
is why it gets its own detector rather than being folded into a generic
"large tool output" warning.
"""

from __future__ import annotations

from ...schema import Event, Role
from .base import Detector, Evidence, Finding

#: Roughly 4 bytes per token. 40 KB ≈ 10k tokens — clearly worth flagging.
FLOOD_BYTES = 40_000
BYTES_PER_TOKEN = 4


class OutputFloodDetector(Detector):
    name = "output_flood"
    title = "Huge command output flooding the context window"
    min_usd = 0.01

    def run(self, events: list[Event]) -> Finding | None:
        if not events:
            return None

        # How many assistant turns follow, i.e. how many times this output gets
        # re-sent as input. That multiplier is the actual cost.
        assistant_seqs = [e.seq for e in events if e.role is Role.ASSISTANT and e.usage]
        if not assistant_seqs:
            return None

        # Blended input rate observed in this session — grounded in what the
        # user actually paid rather than a list price we assume.
        billed_in = sum(
            (e.usage.input + e.usage.cache_read) for e in events if e.usage
        )
        spend_in = sum(
            e.cost_usd * _input_share(e) for e in events if e.usage and e.cost_usd
        )
        rate = (spend_in / billed_in) if billed_in else 0.0

        wasted = 0.0
        tokens = 0
        evidence: list[Evidence] = []

        # Map each tool result back to the call that produced it, so the finding
        # can quote the actual command rather than just naming the tool.
        prev_call: Event | None = None

        for e in events:
            if e.role is Role.TOOL_CALL:
                prev_call = e
                continue
            if e.role is not Role.TOOL_RESULT or not e.tool:
                continue
            nbytes = e.tool.output_bytes
            if nbytes < FLOOD_BYTES:
                continue

            out_tokens = nbytes // BYTES_PER_TOKEN
            resends = sum(1 for s in assistant_seqs if s > e.seq)
            # The first occurrence is legitimate cost; the re-sends are waste.
            billable = out_tokens * resends
            cost = billable * rate

            cmd = (prev_call.text if prev_call else None) or (
                e.tool.target_path or e.tool.raw_name
            )
            wasted += cost
            tokens += billable
            evidence.append(
                Evidence(
                    session_id=e.session_id,
                    ts=e.ts,
                    quote=self.quote(cmd),
                    detail=(
                        f"{nbytes // 1024} KB of output (~{out_tokens:,} tokens) "
                        f"re-sent as input on {resends} later turn(s)."
                    ),
                    cost_usd=round(cost, 6),
                    tokens=billable,
                )
            )

        if not evidence:
            return None

        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "Command output lands in the context window verbatim and is re-sent as "
                    "input tokens on every later turn of the session, so one unpiped command "
                    "is billed dozens of times."
                ),
                prescription=(
                    "Pipe it: `| head -40`, `--quiet`, `-q`, or `2>&1 | tail -20`. "
                    "Ask for the specific lines you need, not the whole dump."
                ),
                wasted_usd=wasted,
                wasted_tokens=tokens,
                occurrences=len(evidence),
                evidence=evidence,
            )
        )


def _input_share(e: Event) -> float:
    """Fraction of this turn's cost attributable to input-side tokens."""
    if not e.usage:
        return 0.0
    inp = e.usage.input + e.usage.cache_read
    tot = e.usage.total
    return (inp / tot) if tot else 0.0
