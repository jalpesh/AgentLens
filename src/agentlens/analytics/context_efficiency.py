"""Context Efficiency — how much context the agent was given versus how much
of it shows signs of being unnecessary.

Usefulness can't be measured from local history, so this never claims tokens
were "wasted". It sums *potential excess* from four concrete, inspectable
signals and labels everything heuristic:

1. re-reading a file that hadn't changed since the last read
2. reading generated / vendored / lock files
3. single tool outputs far larger than a typical error boundary needs
4. the same large block pasted into several prompts

Works on the normalized event stream, so every adapter gets it. Sizes come
from tool output bytes (÷4 ≈ tokens, the same estimate the detectors use).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field

from ..schema import Event, Provider, Role, ToolKind
from .confidence import Confidence, direct, heuristic, temper, unavailable

BYTES_PER_TOKEN = 4
#: A single tool output beyond this is more than an error boundary needs;
#: only the part above it counts as potential excess.
OUTPUT_CAP_TOKENS = 10_000
#: Minimum size for a pasted block to be worth de-duplicating.
PASTE_MIN_TOKENS = 400

_GENERATED = re.compile(
    r"(^|/)(node_modules|dist|build|out|target|vendor|\.next|__pycache__|coverage|"
    r"generated|gen)/|\.min\.(js|css)$|\.map$|(^|/)(package-lock\.json|yarn\.lock|"
    r"pnpm-lock\.yaml|poetry\.lock|uv\.lock|Cargo\.lock|go\.sum)$|_pb2(_grpc)?\.py$|"
    r"\.(g|generated)\.\w+$",
    re.I,
)
_TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+$|[._-](test|spec)\.\w+$",
                        re.I)
_DOC_EXT = re.compile(r"\.(md|rst|txt|adoc)$", re.I)
_CONFIG_EXT = re.compile(r"\.(json|ya?ml|toml|ini|cfg|conf|xml|properties|lock)$|(^|/)\.", re.I)
#: Agent-replayed transcript after compaction ("[12] user: ..."), not a paste.
_REPLAY = re.compile(r"^\[\d+\] (user|assistant):", re.M)
_LOG_LINE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2}[ T]\d|\[?(INFO|WARN|ERROR|DEBUG)\]?|"
                       r"Traceback|\s+at [\w.$]+\(|E\s{2,})", re.M)


def _tokens(nbytes: int) -> int:
    return max(0, nbytes) // BYTES_PER_TOKEN


def _category(e: Event) -> str:
    kind, path = e.tool.kind, (e.tool.target_path or "").replace("\\", "/")
    if kind is ToolKind.READ:
        if _GENERATED.search(path):
            return "generated files"
        if _TEST_PATH.search(path):
            return "tests"
        if _DOC_EXT.search(path):
            return "documentation"
        if _CONFIG_EXT.search(path):
            return "configuration"
        return "code"
    return {
        ToolKind.TEST: "test / build output",
        ToolKind.BASH: "shell output",
        ToolKind.GIT: "shell output",
        ToolKind.SEARCH: "search results",
        ToolKind.WEB: "web / docs fetch",
        ToolKind.EDIT: "edit confirmations",
    }.get(kind, "other tool output")


def _gen_bucket(path: str) -> str:
    """Directory/pattern label for a generated path — never the full path."""
    m = _GENERATED.search(path.replace("\\", "/"))
    return m.group(0).strip("/") if m else "generated"


@dataclass(slots=True)
class Opportunity:
    kind: str
    title: str
    recommendation: str
    occurrences: int = 0
    sessions: set[str] = field(default_factory=set)
    tokens: int = 0
    cost_usd: float = 0.0
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        n = len(self.sessions)
        d["sessions"] = n
        d["tokens_per_session"] = round(self.tokens / n) if n else 0
        d["cost_usd"] = round(self.cost_usd, 4)
        d["examples"] = self.examples[:3]
        return d


def _input_price(events: list[Event]) -> float:
    """Session's average $ per input-side token — what one more token of
    context cost there on first send. 0 when unpriced (shows $0, not guessed).
    Excludes re-sending the token on later turns, so it is a floor."""
    toks = sum(e.usage.input + e.usage.cache_read + e.usage.cache_write
               for e in events if e.usage)
    cost = sum(e.cost_usd for e in events)
    return cost / toks if toks else 0.0


def context_efficiency(sessions_events: list[list[Event]]) -> dict:
    sessions = [evs for evs in sessions_events if evs and evs[0].provider is not Provider.MANUAL]
    sources: dict[str, int] = defaultdict(int)
    opps = {
        "reread": Opportunity(
            "reread", "Unchanged files re-read",
            "Ask for the specific function or a diff instead of the whole file again; "
            "keep long sessions focused so earlier reads stay in context."),
        "generated": Opportunity(
            "generated", "Generated / vendored files read into context",
            "Exclude these paths from the agent (its ignore or deny settings) and point it "
            "at the source that generates them."),
        "oversized": Opportunity(
            "oversized", "Oversized tool outputs",
            f"Pipe long commands through tail/grep around the error; output beyond "
            f"~{OUTPUT_CAP_TOKENS // 1000}k tokens rarely changes the next step."),
        "pasted": Opportunity(
            "pasted", "Same large block pasted into several prompts",
            "Move standing context (conventions, architecture notes) into project "
            "instructions such as AGENTS.md or CLAUDE.md; truncate repeated logs to the "
            "error boundary."),
    }
    block_seen: dict[str, tuple[str, int]] = {}  # hash -> (first session, tokens)
    loaded = prompt_tokens = cache_read = 0
    tool_outputs = 0

    for evs in sessions:
        sid = evs[0].session_id
        price = _input_price(evs)
        last_read: dict[str, int] = {}  # path -> seq of last read
        last_edit: dict[str, int] = {}
        for e in evs:
            if e.usage:
                cache_read += e.usage.cache_read
            if e.role is Role.USER and e.text:
                t = _tokens(len(e.text.encode("utf-8", "replace")))
                prompt_tokens += t
                sources["prompts (typed / pasted)"] += t
                for block in re.split(r"\n\s*\n", e.text):
                    bt = _tokens(len(block.encode("utf-8", "replace")))
                    if bt < PASTE_MIN_TOKENS or _REPLAY.search(block):
                        continue
                    h = hashlib.sha256(" ".join(block.split()).encode()).hexdigest()
                    if h in block_seen:
                        o = opps["pasted"]
                        o.occurrences += 1
                        o.sessions.add(sid)
                        o.tokens += bt
                        o.cost_usd += bt * price
                        if len(o.examples) < 3:
                            o.examples.append(("log block" if _LOG_LINE.search(block)
                                               else "text block") + f", ~{bt:,} tokens")
                    else:
                        block_seen[h] = (sid, bt)
                continue
            if not e.tool:
                continue
            path = (e.tool.target_path or "").replace("\\", "/")
            if e.role is Role.TOOL_CALL and e.tool.kind is ToolKind.EDIT and path:
                last_edit[path] = e.seq
            if e.role is not Role.TOOL_RESULT or not e.tool.output_bytes:
                continue
            t = _tokens(e.tool.output_bytes)
            tool_outputs += 1
            loaded += t
            sources[_category(e)] += t
            if e.tool.kind is ToolKind.READ and path:
                if _GENERATED.search(path):
                    o = opps["generated"]
                    o.occurrences += 1
                    o.sessions.add(sid)
                    o.tokens += t
                    o.cost_usd += t * price
                    label = _gen_bucket(path)
                    if label not in o.examples:
                        o.examples.append(label)
                prev = last_read.get(path)
                if prev is not None and last_edit.get(path, -1) < prev:
                    o = opps["reread"]
                    o.occurrences += 1
                    o.sessions.add(sid)
                    o.tokens += t
                    o.cost_usd += t * price
                last_read[path] = e.seq
            elif t > OUTPUT_CAP_TOKENS:
                o = opps["oversized"]
                o.occurrences += 1
                o.sessions.add(sid)
                over = t - OUTPUT_CAP_TOKENS
                o.tokens += over
                o.cost_usd += over * price
                if len(o.examples) < 3:
                    o.examples.append(f"{_category(e)}, ~{t:,} tokens")

    loaded += prompt_tokens
    n = len(sessions)
    ranked = sorted((o for o in opps.values() if o.occurrences), key=lambda o: -o.tokens)
    # Re-read and generated can overlap (re-reading a generated file); never
    # let excess exceed what was loaded.
    excess = min(sum(o.tokens for o in ranked), loaded)

    if not loaded:
        measures = [unavailable("Context loaded", "No tool output sizes or prompts in scope.",
                                "tokens")]
    else:
        conf = temper(Confidence.MEDIUM if tool_outputs else Confidence.LOW, n)
        measures = [
            heuristic("Context loaded", loaded,
                      "Tool output bytes ÷4 plus typed/pasted prompt text, as it first "
                      "entered the context.", conf, "tokens", n),
            heuristic("Potential excess", excess,
                      "Re-reads of unchanged files, generated files, tool output beyond "
                      f"{OUTPUT_CAP_TOKENS:,} tokens, and repeated pasted blocks.",
                      conf, "tokens", n),
            heuristic("Potential excess share", round(100.0 * excess / loaded, 1),
                      "Potential excess ÷ context loaded. Not proof the tokens were unused.",
                      conf, "pct", n),
            heuristic("Context efficiency", round(100.0 - 100.0 * excess / loaded, 1),
                      "100% − potential excess share.", conf, "pct", n),
        ]
        if cache_read:
            measures.append(direct("Context re-sent from cache", cache_read,
                                   "Agent-reported cache-read tokens: earlier context sent "
                                   "again on every later turn.", "tokens", n))

    total = sum(sources.values()) or 1
    return {
        "measures": measures,
        "sources": [
            {"category": k, "tokens": v, "pct": round(100.0 * v / total, 1)}
            for k, v in sorted(sources.items(), key=lambda kv: -kv[1])
        ],
        "opportunities": [o.to_dict() for o in ranked],
    }
