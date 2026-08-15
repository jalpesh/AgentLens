"""Workflow detectors: edit-without-test, whole-file re-read, uncached docs.

These three share a shape — they compare tool-use ratios against what a
disciplined session looks like — so they live together.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from ...schema import Event, Role, ToolKind
from .base import Detector, Evidence, Finding


class EditWithoutTestDetector(Detector):
    """Many edits, few verification runs.

    Every edit made without a verification step is a coin flip. When the flip
    loses you pay for a rework loop, which costs several multiples of what the
    test run would have cost — because the whole conversation gets re-sent.
    """

    name = "edit_without_test"
    title = "Editing without running the tests"
    min_usd = 0.0  # ratio-based; dollar impact is estimated, not measured

    edit_threshold = 8
    ratio_threshold = 6.0  # edits per verification run

    def run(self, events: list[Event]) -> Finding | None:
        edits = [e for e in events if e.role is Role.TOOL_CALL and e.tool
                 and e.tool.kind is ToolKind.EDIT]
        tests = [e for e in events if e.role is Role.TOOL_CALL and e.tool
                 and e.tool.kind is ToolKind.TEST]
        if len(edits) < self.edit_threshold:
            return None
        ratio = len(edits) / max(1, len(tests))
        if ratio < self.ratio_threshold:
            return None

        session_cost = sum(e.cost_usd for e in events)
        # Conservative: assume a fifth of session spend is rework attributable
        # to unverified edits. Labelled as an estimate in the output.
        estimate = session_cost * 0.20
        sid = events[0].session_id

        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "An unverified edit that turns out wrong triggers a rework loop, and a "
                    "rework loop re-sends the entire conversation. A test run is far cheaper "
                    "than discovering the problem three turns later."
                ),
                prescription=(
                    'Put the acceptance check in the prompt itself: "…and `pytest -q` must '
                    'pass". Make the agent verify in the same turn it edits.'
                ),
                wasted_usd=estimate,
                wasted_tokens=0,
                occurrences=len(edits),
                evidence=[
                    Evidence(
                        session_id=sid,
                        ts=edits[0].ts,
                        quote=f"{len(edits)} edit operations, {len(tests)} verification run(s)",
                        detail=(
                            f"Ratio {ratio:.0f}:1. Estimated rework exposure "
                            f"${estimate:.2f} (20% of this session's ${session_cost:.2f})."
                        ),
                        cost_usd=round(estimate, 6),
                    )
                ],
            )
        )


class WholeFileRereadDetector(Detector):
    """Reading the same file over and over instead of reviewing the diff."""

    name = "whole_file_reread"
    title = "Re-reading whole files instead of diffs"
    min_usd = 0.005
    repeat_threshold = 3

    def run(self, events: list[Event]) -> Finding | None:
        reads: dict[str, list[Event]] = defaultdict(list)
        for e in events:
            if e.role is Role.TOOL_RESULT and e.tool and e.tool.kind is ToolKind.READ:
                if e.tool.target_path:
                    reads[e.tool.target_path].append(e)

        billed_in = sum((e.usage.input + e.usage.cache_read) for e in events if e.usage)
        spend = sum(e.cost_usd for e in events)
        rate = (spend / billed_in) if billed_in else 0.0

        wasted = 0.0
        tokens = 0
        evidence: list[Evidence] = []
        for path, evs in reads.items():
            if len(evs) < self.repeat_threshold:
                continue
            redundant = evs[1:]  # first read is legitimate
            btoks = sum(e.tool.output_bytes for e in redundant) // 4
            cost = btoks * rate
            wasted += cost
            tokens += btoks
            evidence.append(
                Evidence(
                    session_id=evs[0].session_id,
                    ts=evs[0].ts,
                    quote=path.rsplit("/", 1)[-1],
                    detail=(
                        f"Read {len(evs)} times in one session "
                        f"(~{btoks:,} redundant tokens)."
                    ),
                    cost_usd=round(cost, 6),
                    tokens=btoks,
                )
            )

        if not evidence:
            return None
        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "Each full re-read pushes the entire file into context again, and it stays "
                    "there for the rest of the session."
                ),
                prescription=(
                    "Review the change, not the file: ask for `git diff` on the specific path, "
                    "or point at line ranges you care about."
                ),
                wasted_usd=wasted,
                wasted_tokens=tokens,
                occurrences=len(evidence),
                evidence=evidence,
            )
        )


class UncachedDocsDetector(Detector):
    """Fetching the same page repeatedly. Fetched pages arrive raw and are
    billed in full every single time."""

    name = "uncached_docs"
    title = "Re-fetching the same documentation"
    min_usd = 0.005
    repeat_threshold = 2

    def run(self, events: list[Event]) -> Finding | None:
        fetches: Counter[str] = Counter()
        first_seen: dict[str, Event] = {}
        bytes_by_url: Counter[str] = Counter()

        pending: str | None = None
        for e in events:
            if e.role is Role.TOOL_CALL and e.tool and e.tool.kind is ToolKind.WEB:
                key = (e.tool.target_path or e.text or e.tool.raw_name or "").strip()[:200]
                if key:
                    pending = key
                    fetches[key] += 1
                    first_seen.setdefault(key, e)
            elif e.role is Role.TOOL_RESULT and e.tool and e.tool.kind is ToolKind.WEB:
                if pending:
                    bytes_by_url[pending] += e.tool.output_bytes
                pending = None

        billed_in = sum((e.usage.input + e.usage.cache_read) for e in events if e.usage)
        spend = sum(e.cost_usd for e in events)
        rate = (spend / billed_in) if billed_in else 0.0

        wasted = 0.0
        tokens = 0
        evidence: list[Evidence] = []
        for url, n in fetches.items():
            if n < self.repeat_threshold + 1:
                continue
            avg = bytes_by_url[url] / max(1, n)
            btoks = int(avg * (n - 1)) // 4
            cost = btoks * rate
            wasted += cost
            tokens += btoks
            src = first_seen[url]
            evidence.append(
                Evidence(
                    session_id=src.session_id,
                    ts=src.ts,
                    quote=self.quote(url, 90),
                    detail=f"Fetched {n} times (~{btoks:,} redundant tokens).",
                    cost_usd=round(cost, 6),
                    tokens=btoks,
                )
            )

        if not evidence:
            return None
        return self.emit(
            Finding(
                detector=self.name,
                title=self.title,
                mechanism=(
                    "Fetched pages arrive as raw markup and are billed in full on every fetch, "
                    "then re-sent as context for the rest of the session."
                ),
                prescription=(
                    "Save the answer once into a project note (CLAUDE.md / AGENTS.md / "
                    ".junie/guidelines.md) so the agent reads 20 lines instead of the page."
                ),
                wasted_usd=wasted,
                wasted_tokens=tokens,
                occurrences=len(evidence),
                evidence=evidence,
            )
        )
