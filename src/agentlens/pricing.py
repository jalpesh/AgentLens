"""Model pricing.

Cost is **always recomputed** here, never trusted from the session log. Logs
record whatever the vendor felt like at the time, prices change, and you need to
be able to reprice history when they do.

Prices are USD per million tokens. The table ships embedded so the tool works
fully offline on first run; ``agentlens pricing --update`` can refresh it from a
local override file the user controls.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .schema import Usage


@dataclass(frozen=True, slots=True)
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float = 0.0
    cache_write: float = 0.0
    context_limit: int = 200_000


# Embedded fallback table. Values are per 1M tokens.
# Keys are matched with the longest-prefix rule in `lookup()`, so dated variants
# like `claude-opus-4-5-20251101` resolve against `claude-opus-4-5`.
PRICES: dict[str, Price] = {
    # --- Anthropic -----------------------------------------------------
    "claude-opus-4-5":    Price(5.00, 25.00, 0.50, 6.25, 200_000),
    "claude-opus-4-1":    Price(15.00, 75.00, 1.50, 18.75, 200_000),
    "claude-opus-4":      Price(15.00, 75.00, 1.50, 18.75, 200_000),
    "claude-sonnet-4-5":  Price(3.00, 15.00, 0.30, 3.75, 200_000),
    "claude-sonnet-4":    Price(3.00, 15.00, 0.30, 3.75, 200_000),
    "claude-haiku-4-5":   Price(1.00, 5.00, 0.10, 1.25, 200_000),
    "claude-3-7-sonnet":  Price(3.00, 15.00, 0.30, 3.75, 200_000),
    "claude-3-5-haiku":   Price(0.80, 4.00, 0.08, 1.00, 200_000),
    # --- OpenAI --------------------------------------------------------
    "gpt-5.1-codex":      Price(1.25, 10.00, 0.125, 0.0, 400_000),
    "gpt-5.1":            Price(1.25, 10.00, 0.125, 0.0, 400_000),
    "gpt-5-codex":        Price(1.25, 10.00, 0.125, 0.0, 400_000),
    "gpt-5-mini":         Price(0.25, 2.00, 0.025, 0.0, 400_000),
    "gpt-5":              Price(1.25, 10.00, 0.125, 0.0, 400_000),
    "gpt-4.1-mini":       Price(0.40, 1.60, 0.10, 0.0, 1_047_576),
    "gpt-4.1":            Price(2.00, 8.00, 0.50, 0.0, 1_047_576),
    "o4-mini":            Price(1.10, 4.40, 0.275, 0.0, 200_000),
    # --- Google --------------------------------------------------------
    "gemini-3-pro":       Price(2.00, 12.00, 0.20, 0.0, 1_000_000),
    "gemini-3-flash":     Price(0.30, 2.50, 0.03, 0.0, 1_000_000),
    "gemini-2.5-pro":     Price(1.25, 10.00, 0.31, 0.0, 1_048_576),
    "gemini-2.5-flash":   Price(0.30, 2.50, 0.075, 0.0, 1_048_576),
}

_UNKNOWN = Price(0.0, 0.0, 0.0, 0.0, 200_000)

_OVERRIDE_ENV = "AGENTLENS_PRICING_FILE"


def _load_overrides() -> dict[str, Price]:
    path = os.environ.get(_OVERRIDE_ENV)
    if not path:
        default = Path.home() / ".agentlens" / "pricing.json"
        path = str(default) if default.exists() else ""
    if not path or not Path(path).exists():
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out: dict[str, Price] = {}
    for key, v in raw.items():
        try:
            out[key.lower()] = Price(
                input=float(v["input"]),
                output=float(v["output"]),
                cache_read=float(v.get("cache_read", 0.0)),
                cache_write=float(v.get("cache_write", 0.0)),
                context_limit=int(v.get("context_limit", 200_000)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


_OVERRIDES = _load_overrides()

# Strip trailing date stamps: claude-opus-4-5-20251101 -> claude-opus-4-5
_DATE_SUFFIX = re.compile(r"-(?:20\d{6}|\d{4}-\d{2}-\d{2}|latest|preview)$")


def canonical(model: str | None) -> str:
    if not model:
        return "unknown"
    m = model.strip().lower()
    # Strip provider routing prefixes: anthropic/claude-..., openai/gpt-...
    if "/" in m:
        m = m.rsplit("/", 1)[-1]
    prev = None
    while prev != m:
        prev = m
        m = _DATE_SUFFIX.sub("", m)
    return m


def lookup(model: str | None) -> Price:
    """Longest-prefix match so unseen dated variants still price correctly."""
    key = canonical(model)
    table = {**PRICES, **_OVERRIDES}
    if key in table:
        return table[key]
    best: tuple[int, Price] | None = None
    for name, price in table.items():
        if key.startswith(name) and (best is None or len(name) > best[0]):
            best = (len(name), price)
    return best[1] if best else _UNKNOWN


def cost_of(model: str | None, usage: Usage | None) -> float:
    if usage is None:
        return 0.0
    p = lookup(model)
    return (
        usage.input * p.input
        + usage.output * p.output
        + usage.cache_read * p.cache_read
        + usage.cache_write * p.cache_write
        + usage.reasoning * p.output
    ) / 1_000_000


def context_limit_of(model: str | None) -> int:
    return lookup(model).context_limit


def is_known(model: str | None) -> bool:
    return lookup(model) is not _UNKNOWN
