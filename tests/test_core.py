"""Pricing, prompt scoring, redaction and store idempotency."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.analytics.prompt_score import score_prompt  # noqa: E402
from agentlens.pricing import canonical, cost_of, lookup  # noqa: E402
from agentlens.redact import redact_obj, redact_text  # noqa: E402
from agentlens.schema import Usage  # noqa: E402
from agentlens.store import Store  # noqa: E402
from generate import build  # noqa: E402


# --- pricing --------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("claude-opus-4-5-20251101", "claude-opus-4-5"),
        ("anthropic/claude-sonnet-4-5-20250929", "claude-sonnet-4-5"),
        ("gpt-5.1-codex", "gpt-5.1-codex"),
        ("gemini-3-pro-preview", "gemini-3-pro"),
    ],
)
def test_dated_variants_resolve_to_a_price(raw, expected):
    assert canonical(raw) == expected
    assert lookup(raw).input > 0, "an unseen date stamp must not zero out cost"


def test_cache_reads_are_priced_separately():
    """Two sessions with identical token totals can differ several-fold in
    cost purely on cache hit rate. Collapsing them is the classic bug."""
    m = "claude-sonnet-4-5"
    hot = cost_of(m, Usage(input=1_000, cache_read=99_000))
    cold = cost_of(m, Usage(input=100_000))
    assert hot < cold / 5, "cache reads must be dramatically cheaper"


def test_unknown_model_costs_zero_rather_than_guessing():
    assert cost_of("some-model-we-have-never-seen", Usage(input=1000, output=100)) == 0.0


# --- prompt lab -----------------------------------------------------------

def test_scorer_discriminates_good_from_bad():
    """A scorer that returns 100/100 for everything is decoration. This is the
    property the original build got wrong."""
    bad = score_prompt("fix the auth stuff, maybe clean it up if possible. thanks!")
    good = score_prompt(
        "In @src/auth.py make login() reject expired tokens; "
        "`pytest tests/test_auth.py -q` must pass."
    )
    assert bad.score < 50
    assert good.score >= 90
    assert good.score - bad.score > 40


def test_scorer_names_what_is_missing():
    codes = {i.code for i in score_prompt("fix it").issues}
    assert "no_file_ref" in codes and "no_acceptance" in codes


def test_cleanup_never_changes_meaning():
    r = score_prompt("Please could you update @src/api.py so it returns 404. Thanks!")
    assert "@src/api.py" in r.cleaned
    assert "404" in r.cleaned
    assert r.cleaned_tokens <= r.tokens


def test_empty_prompt_is_handled():
    assert score_prompt("").score == 0


# --- redaction ------------------------------------------------------------

def test_redaction_strips_identifiers_and_secrets():
    text = (
        "clone https://dev.azure.com/AcmeCorp/Common/_git/toolkit into "
        "/Users/jalpesh/work/app with token sk-abcdefghij1234567890 "
        "and email me at someone@acme.com from 10.1.2.3"
    )
    out = redact_text(text)
    for leak in ("AcmeCorp", "jalpesh", "sk-abcdefghij", "acme.com", "10.1.2.3"):
        assert leak not in out, f"{leak!r} leaked through redaction"


def test_secrets_are_stripped_even_with_redaction_off():
    out = redact_text("export API_KEY=sk-abcdefghij1234567890", full=False)
    assert "sk-abcdefghij1234567890" not in out


def test_redact_obj_shortens_paths_but_keeps_the_leaf():
    out = redact_obj({"cwd": "/Users/jalpesh/work/secret-client/api"})
    assert out["cwd"] == ".../api"


# --- store ----------------------------------------------------------------

def test_ingest_is_idempotent(tmp_path):
    """You will re-scan the same files hundreds of times. If that double-counts,
    every number in the product becomes untrustworthy."""
    paths = build(tmp_path / "projects")
    adapter = ClaudeCodeAdapter()
    db = tmp_path / "t.db"

    with Store(db) as s:
        for p in paths:
            s.add_events(adapter.parse(SourceFile.of(p)))
        first = s.counts()
        for p in paths:  # deliberate re-ingest
            s.add_events(adapter.parse(SourceFile.of(p)))
        second = s.counts()

    assert first == second, "re-ingesting identical files must be a no-op"
    assert first["events"] > 0


def test_store_roundtrips_events(tmp_path):
    paths = build(tmp_path / "projects")
    adapter = ClaudeCodeAdapter()
    original = list(adapter.parse(SourceFile.of(paths[0])))

    with Store(tmp_path / "r.db") as s:
        s.add_events(original)
        back = s.events_for(original[0].session_id)

    assert len(back) == len(original)
    assert round(sum(e.cost_usd for e in back), 6) == round(
        sum(e.cost_usd for e in original), 6
    )
