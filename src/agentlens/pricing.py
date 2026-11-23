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
    # Claude 5 / 4.6+ input and output from Anthropic's published list prices
    # (Nov 2026). ponytail: where a cache rate wasn't published (Opus 5,
    # Sonnet 5, Fable 5, Haiku 5.5) it uses the usual 0.1x read / 1.25x write.
    # Haiku 5.5 doubles past 100K input; that tier is ignored.
    "claude-fable-5-2":   Price(8.00, 40.00, 0.20, 10.00, 1_000_000),
    "claude-fable-5-1":   Price(10.00, 50.00, 1.00, 12.50, 1_000_000),
    "claude-fable-5":     Price(10.00, 50.00, 1.00, 12.50, 1_000_000),
    "claude-opus-5-5":    Price(4.00, 20.00, 0.20, 5.00, 1_000_000),
    "claude-opus-5":      Price(5.00, 25.00, 0.50, 6.25, 1_000_000),
    "claude-sonnet-5-5":  Price(2.00, 10.00, 0.10, 2.50, 1_000_000),
    "claude-sonnet-5":    Price(2.00, 10.00, 0.20, 2.50, 1_000_000),
    "claude-haiku-5-5":   Price(0.10, 0.50, 0.01, 0.125, 1_000_000),
    "claude-opus-4-8":    Price(5.00, 25.00, 0.50, 6.25, 1_000_000),
    "claude-opus-4-7":    Price(5.00, 25.00, 0.50, 6.25, 1_000_000),
    "claude-opus-4-6":    Price(5.00, 25.00, 0.50, 6.25, 1_000_000),
    "claude-sonnet-4-6":  Price(3.00, 15.00, 0.30, 3.75, 1_000_000),
    "claude-opus-4-5":    Price(5.00, 25.00, 0.50, 6.25, 200_000),
    "claude-opus-4-1":    Price(15.00, 75.00, 1.50, 18.75, 200_000),
    "claude-opus-4":      Price(15.00, 75.00, 1.50, 18.75, 200_000),
    "claude-sonnet-4-5":  Price(3.00, 15.00, 0.30, 3.75, 200_000),
    "claude-sonnet-4":    Price(3.00, 15.00, 0.30, 3.75, 200_000),
    "claude-haiku-4-5":   Price(1.00, 5.00, 0.10, 1.25, 200_000),
    "claude-3-7-sonnet":  Price(3.00, 15.00, 0.30, 3.75, 200_000),
    "claude-3-5-haiku":   Price(0.80, 4.00, 0.08, 1.00, 200_000),
    # --- OpenAI --------------------------------------------------------
    # From developers.openai.com/api/docs/pricing, Nov 2026, standard tier.
    # ponytail: the >272K-input tier (2x input) is ignored; GPT-5.6 Sol is on
    # promotional pricing until 2026-11-30.
    "gpt-5.6-sol":        Price(4.00, 20.00, 0.40, 0.0, 400_000),
    "gpt-5.6-terra":      Price(2.00, 12.00, 0.20, 0.0, 400_000),
    "gpt-5.6-luna":       Price(0.20, 1.20, 0.02, 0.0, 400_000),
    "gpt-5.5":            Price(5.00, 30.00, 0.50, 0.0, 400_000),
    "gpt-5.4":            Price(2.50, 15.00, 0.25, 0.0, 400_000),
    "gpt-5.3-codex":      Price(1.75, 14.00, 0.175, 0.0, 400_000),
    "gpt-4o-mini":        Price(0.15, 0.60, 0.075, 0.0, 128_000),
    "gpt-4o":             Price(2.50, 10.00, 1.25, 0.0, 128_000),
    "gpt-5.1-codex":      Price(1.25, 10.00, 0.125, 0.0, 400_000),
    "gpt-5.1":            Price(1.25, 10.00, 0.125, 0.0, 400_000),
    "gpt-5-codex":        Price(1.25, 10.00, 0.125, 0.0, 400_000),
    "gpt-5-mini":         Price(0.25, 2.00, 0.025, 0.0, 400_000),
    "gpt-5":              Price(1.25, 10.00, 0.125, 0.0, 400_000),
    "gpt-4.1-mini":       Price(0.40, 1.60, 0.10, 0.0, 1_047_576),
    "gpt-4.1":            Price(2.00, 8.00, 0.50, 0.0, 1_047_576),
    "o4-mini":            Price(1.10, 4.40, 0.275, 0.0, 200_000),
    # --- Google --------------------------------------------------------
    # From aistudio.google.com/docs/pricing, Nov 2026. ponytail: 3.8/3.6 Flash
    # prices double on 2027-01-01; one price per model, so update then.
    "gemini-3.8-flash":   Price(0.75, 3.75, 0.075, 0.0, 1_000_000),
    "gemini-3.6-flash":   Price(0.75, 3.75, 0.075, 0.0, 1_000_000),
    "gemini-3.1-pro":     Price(2.00, 12.00, 0.20, 0.0, 1_000_000),
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
