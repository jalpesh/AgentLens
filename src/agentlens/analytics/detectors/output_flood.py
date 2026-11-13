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

from collections import defaultdict

from ...pricing import lookup
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
        # Compaction drops the output from context, so re-sends stop there.
        # Not every agent logs a compaction marker, so a turn whose context
        # is under half the previous turn's counts as one too.
        compactions = [e.seq for e in events if e.role is Role.COMPACTION]
        prev_ctx = 0
        for e in events:
            if e.role is Role.ASSISTANT and e.usage:
                ctx = e.usage.input + e.usage.cache_read + e.usage.cache_write
                if prev_ctx and ctx < prev_ctx / 2:
                    compactions.append(e.seq)
                prev_ctx = ctx
        compactions.sort()

        # Blended input rate observed in this session — grounded in what the
        # user actually paid rather than a list price we assume.
        billed_in = sum(
            (e.usage.input + e.usage.cache_read) for e in events if e.usage
        )
        spend_in = sum(
            e.cost_usd * _input_share(e) for e in events if e.usage and e.cost_usd
        )
        rate = (spend_in / billed_in) if billed_in else 0.0
        # A re-sent output is served from the prompt cache when the turn used
        # it, so it is billed at that model's cache-read price, not the
        # blended rate (which includes full-price input and cache writes).
        turn_rate = {}
        for e in events:
            if e.role is Role.ASSISTANT and e.usage:
                cr = lookup(e.model).cache_read / 1_000_000 if e.usage.cache_read else 0.0
                turn_rate[e.seq] = cr or rate

        wasted = 0.0
        tokens = 0
        evidence: list[Evidence] = []

        # Map each tool result back to the call that produced it, so the finding
        # can quote the actual command rather than just naming the tool.
        prev_call: Event | None = None
        floods: list[tuple[Evidence, int, list[int], set[int]]] = []  # (ev, over, later, set)

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
            end = next((c for c in compactions if c > e.seq), float("inf"))
            later = [s for s in assistant_seqs if e.seq < s < end]
            resends = len(later)
            # The first occurrence is legitimate cost, and so is the first
            # FLOOD_BYTES of it (an error boundary the agent needed); only the
            # flood above that, re-sent, is waste.
            over = (nbytes - FLOOD_BYTES) // BYTES_PER_TOKEN
            cmd = (prev_call.text if prev_call else None) or (
                e.tool.target_path or e.tool.raw_name
            )
            ev = Evidence(
                    session_id=e.session_id,
                    ts=e.ts,
                    quote=self.quote(cmd),
                    detail=(
                        f"{nbytes // 1024} KB of output (~{out_tokens:,} tokens) "
                        f"re-sent as input on {resends} later turn(s)."
                    ),
            )
            evidence.append(ev)
            floods.append((ev, over, later, set(later)))

        if not evidence:
            return None

        # Many floods can be live at once, but together they can't exceed the
        # context a turn actually sent. When they would, the oldest must have
        # been dropped (truncation, clearing, compaction the log didn't mark)
        # and never comes back. Without this, 352 floods in one long session
        # claimed 94% of all its cache reads (real data, 2026-11-13).
        ctx = {e.seq: e.usage.input + e.usage.cache_read for e in events
               if e.role is Role.ASSISTANT and e.usage}
        charged = [0] * len(floods)  # turns each flood was really re-sent on
        cost_of = [0.0] * len(floods)
        enters: dict[int, list[int]] = defaultdict(list)
        for i, f in enumerate(floods):
            if f[2]:
                enters[f[2][0]].append(i)
        live: list[int] = []  # oldest first
        for t in assistant_seqs:
            live = [i for i in live if t in floods[i][3]] + enters.get(t, [])
            while live and sum(floods[i][1] for i in live) > ctx[t]:
                live.pop(0)
            for i in live:
                charged[i] += 1
                cost_of[i] += floods[i][1] * turn_rate[t]
        for i, (ev, over, _later, _set) in enumerate(floods):
            ev.cost_usd = round(cost_of[i], 6)
            ev.tokens = over * charged[i]
            ev.detail = ev.detail.replace(f"on {len(_later)} later", f"on {charged[i]} later")
            wasted += cost_of[i]
            tokens += ev.tokens

        # Re-sends are input tokens: they can never cost more than this
        # session's whole input-side spend.
        if wasted > spend_in:
            scale = spend_in / wasted
            wasted = spend_in
            tokens = int(tokens * scale)
            for ev in evidence:
                ev.cost_usd = round(ev.cost_usd * scale, 6)

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
