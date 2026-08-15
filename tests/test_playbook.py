"""Playbook tests: detect → prescribe → install-the-fix.

Two properties matter: the personalised section must actually derive from the
user's own findings (not boilerplate), and it must stay clearly separated from
the curated reference — mixing the two would make the tool's one real
differentiator look the same as content anyone could write.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.analytics.detectors import run_all  # noqa: E402
from agentlens.analytics.playbook import (  # noqa: E402
    CURATED_REFERENCE,
    DEFAULT_LIMIT,
    build_playbook,
    render,
)
from generate import build  # noqa: E402


@pytest.fixture(scope="module")
def findings(tmp_path_factory):
    paths = build(tmp_path_factory.mktemp("cc") / "projects")
    a = ClaudeCodeAdapter()
    sessions = [list(a.parse(SourceFile.of(p))) for p in paths]
    return run_all(sessions)


def test_rules_derive_from_real_findings(findings):
    rules = build_playbook(findings)
    assert rules
    prescriptions = {f.prescription for f in findings}
    for r in rules:
        assert r.text in prescriptions
        assert r.evidence_usd >= 0
        assert r.source


def test_rules_are_ranked_by_dollar_evidence(findings):
    rules = build_playbook(findings)
    costs = [r.evidence_usd for r in rules]
    assert costs == sorted(costs, reverse=True)


def test_rules_are_capped_by_default():
    """A 30-rule guidelines file is itself the kind of bloat this tool exists
    to catch — the cap is a correctness requirement, not a nicety."""
    from agentlens.analytics.detectors.base import Evidence, Finding

    many = [
        Finding(
            detector=f"d{i}", title=f"t{i}", prescription=f"do thing {i}",
            mechanism="because", wasted_usd=100 - i, occurrences=5,
            evidence=[Evidence(session_id="s", ts="", quote="q")],
        )
        for i in range(20)
    ]
    rules = build_playbook(many)
    assert len(rules) == DEFAULT_LIMIT


def test_empty_findings_render_gracefully():
    text = render(build_playbook([]), fmt="markdown")
    assert "No recurring waste patterns" in text
    assert CURATED_REFERENCE.strip() in text


def test_all_formats_render_valid_markdown(findings):
    rules = build_playbook(findings)
    for fmt in ("claude", "agents", "junie", "markdown"):
        text = render(rules, fmt=fmt)
        assert text.strip()
        assert rules[0].text in text
        # every rule carries its dollar evidence as an inline comment
        assert f"${rules[0].evidence_usd:.2f}" in text


def test_unknown_format_rejected(findings):
    with pytest.raises(ValueError):
        render(build_playbook(findings), fmt="not-a-format")


def test_personalised_and_reference_sections_are_clearly_separated(findings):
    """The personalised section must never be silently merged with generic
    advice — a reader has to be able to tell which lines came from their own
    data at a glance."""
    text = render(build_playbook(findings), fmt="markdown", include_reference=True)
    personal_idx = text.index("Rules from your own history")
    reference_idx = text.index("General reference")
    assert personal_idx < reference_idx, "personalised rules must render first"

    without_ref = render(build_playbook(findings), fmt="markdown", include_reference=False)
    assert "General reference" not in without_ref


def test_curated_reference_has_no_dollar_evidence():
    """The reference section must never claim data-backed evidence it doesn't
    have — that would blur the one distinction this feature depends on."""
    assert "$" not in CURATED_REFERENCE
