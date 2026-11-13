"""Confidence framework tests.

The rule this file exists to enforce: **a weak estimate must never be
indistinguishable from a hard fact, and "unknown" must never collapse into
"zero".** Every test here is a specific way that promise could quietly break.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.analytics.confidence import (  # noqa: E402
    Confidence,
    Measure,
    Source,
    data_quality,
    direct,
    summary,
    temper,
    unavailable,
)
from agentlens.pricing import cost_of  # noqa: E402
from agentlens.schema import Event, Provider, Role, Usage  # noqa: E402


# --- the Measure contract ---------------------------------------------------


def test_unavailable_is_not_zero():
    """The core promise: a number we can't compute is None, never 0.0 — a
    renderer must be able to tell 'no data' apart from 'measured zero'."""
    m = unavailable("Estimated time saved", method="No baseline configured.")
    assert m.value is None
    assert m.confidence is Confidence.NONE
    assert m.available is False


def test_present_measure_cannot_claim_none_confidence():
    """NONE is reserved for UNAVAILABLE. A present value owning NONE would be a
    number with no trust rating at all — exactly the ambiguity to forbid."""
    with pytest.raises(ValueError, match="UNAVAILABLE"):
        Measure("x", 5, Source.DIRECT, Confidence.NONE, "method")


def test_direct_measure_is_high_and_available():
    m = direct("Tokens analysed", 125_000, method="Summed from usage records.", unit="tokens")
    assert m.source is Source.DIRECT
    assert m.confidence is Confidence.HIGH
    assert m.available is True
    assert m.to_dict()["confidence"] == "high"


# --- sample-size tempering --------------------------------------------------


def test_thin_sample_drops_confidence_one_rung():
    assert temper(Confidence.HIGH, sample_size=4, minimum=30) is Confidence.MEDIUM
    assert temper(Confidence.MEDIUM, sample_size=4, minimum=30) is Confidence.LOW


def test_sufficient_sample_keeps_confidence():
    assert temper(Confidence.HIGH, sample_size=500, minimum=30) is Confidence.HIGH


def test_tempering_never_goes_below_low_or_touches_none():
    assert temper(Confidence.LOW, sample_size=1, minimum=30) is Confidence.LOW
    assert temper(Confidence.NONE, sample_size=1, minimum=30) is Confidence.NONE


# --- the classifier over real-shaped events ---------------------------------


def _priced_event(seq: int, tokens: int) -> Event:
    u = Usage(input=tokens)
    return Event(
        session_id="s1", provider=Provider.CLAUDE_CODE, seq=seq,
        ts="2026-01-01T00:00:00", role=Role.ASSISTANT,
        model="claude-sonnet-4-5", usage=u, cost_usd=cost_of("claude-sonnet-4-5", u),
    )


def _unpriced_event(seq: int, tokens: int) -> Event:
    u = Usage(input=tokens)
    return Event(
        session_id="s1", provider=Provider.CLAUDE_CODE, seq=seq,
        ts="2026-01-01T00:00:01", role=Role.ASSISTANT,
        model="claude-sonnet-9", usage=u, cost_usd=cost_of("claude-sonnet-9", u),
    )


def _get(measures, label):
    return next(m for m in measures if m.label == label)


def test_empty_scope_yields_available_zeros_and_honest_unavailables():
    measures = data_quality([], [])
    tokens = _get(measures, "Tokens analysed")
    assert tokens.value == 0 and tokens.available is True  # measured zero, not unknown
    time_saved = _get(measures, "Estimated time saved")
    assert time_saved.available is False  # unknown, not zero


def test_all_priced_models_give_high_confidence_spend():
    events = [[_priced_event(1, 1000), _priced_event(2, 2000)]]
    spend = _get(data_quality(events, []), "Total spend")
    assert spend.source is Source.DERIVED
    assert spend.confidence is Confidence.HIGH


def test_heavy_unpriced_usage_degrades_spend_confidence():
    """The exact failure the pricing-coverage story exists to surface: most
    tokens on a model with no price. Spend must self-report as untrustworthy
    rather than imply the work was nearly free."""
    events = [[_priced_event(1, 1000), _unpriced_event(2, 500_000)]]
    spend = _get(data_quality(events, []), "Total spend")
    assert spend.confidence is Confidence.LOW
    assert "%" in spend.method  # states the priced share out loud


def test_loop_cost_is_labelled_heuristic_not_direct():
    events = [[_priced_event(1, 1000)]]
    loop = _get(data_quality(events, []), "Loop / rework cost")
    assert loop.source is Source.HEURISTIC
    assert loop.confidence is not Confidence.HIGH


def test_discovery_share_is_tempered_on_a_thin_sample():
    """A handful of steps can't support a HIGH rating even though the ratio
    itself is exact."""
    events = [[_priced_event(1, 1000), _priced_event(2, 2000)]]
    disc = _get(data_quality(events, []), "Discovery share")
    assert disc.sample_size is not None and disc.sample_size < 30
    assert disc.confidence in (Confidence.MEDIUM, Confidence.LOW)


def test_every_measure_carries_a_plain_english_method():
    """No number without a 'why you can trust me' line — that string is what
    the UI shows on hover and what makes the framework more than decoration."""
    for m in data_quality([[_priced_event(1, 1000)]], []):
        assert m.method and len(m.method) > 15


def test_summary_counts_add_up():
    measures = data_quality([[_priced_event(1, 1000)]], [])
    counts = summary(measures)
    assert sum(counts.values()) == len(measures)
