"""Prompt rewrite tests.

The scaffold's whole value is that it is *not* a generic template — it fills
itself from the user's own session. These tests check that it pulls real
values, and that it says so honestly when it can't.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.analytics.rewrite import (  # noqa: E402
    detect_test_command,
    llm_available,
    scaffold,
)
from generate import build  # noqa: E402

LOOPING = "66666666-6666-4666-8666-666666666666"
BLIND = "11111111-1111-4111-8111-111111111111"


@pytest.fixture(scope="module")
def sessions(tmp_path_factory) -> dict[str, list]:
    paths = build(tmp_path_factory.mktemp("cc") / "projects")
    a = ClaudeCodeAdapter()
    return {p.stem: list(a.parse(SourceFile.of(p))) for p in paths}


def test_scaffold_fills_from_real_session(sessions):
    s = scaffold("fix the signature check", sessions[LOOPING], None, ["no_file_ref"])
    assert "verify.py" in s.text
    assert s.filled.get("file", "").endswith("verify.py")


def test_scaffold_uses_the_real_test_command(sessions):
    s = scaffold("fix it", sessions[LOOPING], None, ["no_acceptance"])
    assert "pytest tests/test_verify.py" in s.text


def test_scaffold_includes_the_observed_error(sessions):
    s = scaffold("fix it", sessions[LOOPING], None, ["no_file_ref", "no_acceptance"])
    assert "AssertionError" in s.text
    assert "observed" in s.filled


def test_scaffold_is_honest_about_what_it_could_not_fill():
    """With no session context there is nothing to infer — it must say so
    rather than inventing a plausible-looking file path."""
    s = scaffold("make it work", [], None, ["no_file_ref", "no_acceptance"])
    assert "<FILE>" in s.text
    assert "<FILE>" in s.placeholders
    assert not s.filled


def test_scaffold_preserves_user_intent(sessions):
    """The rewrite adds specifics; it must not paraphrase away what was asked."""
    s = scaffold("reject expired tokens", sessions[LOOPING], None, [])
    assert "reject expired tokens" in s.text


def test_detect_test_command_ignores_prose_mentioning_a_command(sessions):
    """A prompt that *describes* running pytest is not a test command.

    An earlier version scanned every event's text and cheerfully returned a
    whole sentence ("I ran `pytest ...` and got AssertionError...") as the
    acceptance command.
    """
    cmd = detect_test_command(sessions[BLIND])
    assert cmd is None or cmd.startswith(("pytest", "npm", "yarn", "go", "cargo"))
    assert cmd is None or "I ran" not in cmd


def test_detect_test_command_finds_real_commands(sessions):
    assert detect_test_command(sessions[LOOPING]) == "pytest tests/test_verify.py -q"


def test_llm_is_not_available_without_a_key(monkeypatch):
    """The default install must never make a network call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert llm_available() is None


def test_llm_rewrite_refuses_without_a_key(monkeypatch):
    from agentlens.analytics.rewrite import llm_rewrite

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No API key"):
        llm_rewrite("x", "y")


def test_package_import_pulls_in_no_http_stack():
    """SECURITY.md claims the default install has no network code loaded.
    This test is what keeps that claim honest."""
    import subprocess

    code = (
        "import sys, agentlens, agentlens.cli, agentlens.analytics;"
        "mods = set(sys.modules);"
        "bad = [m for m in mods if m.split('.')[0] in "
        "{'requests','httpx','aiohttp','urllib3'}];"
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin"},
    )
    assert out.stdout.strip() == "", f"HTTP stack loaded on import: {out.stdout}"


# --- evidence kinds ---------------------------------------------------------


def test_prompt_evidence_is_labelled_as_prompt(sessions):
    from agentlens.analytics.detectors import run_session

    findings = run_session(sessions[BLIND])
    f = next(x for x in findings if x.detector == "blind_retry")
    assert all(e.kind == "prompt" for e in f.evidence if e.full_text)


def test_error_evidence_is_not_labelled_as_prompt(sessions):
    """Loop evidence is a stack trace. Offering "score this as a prompt" on a
    traceback is meaningless, so the UI needs to be able to tell them apart."""
    from agentlens.analytics.detectors import run_session

    f = next(x for x in run_session(sessions[LOOPING]) if x.detector == "loop")
    assert all(e.kind == "error" for e in f.evidence if e.full_text)


# --- Prompt Lab additions (Phase 4, section C) ------------------------------


def test_same_prompt_across_models_is_ordered_cheapest_first():
    from agentlens.analytics.prompt_score import score_across_models

    text = "In @src/auth.py make login() reject expired tokens; `pytest -q` must pass."
    reports = score_across_models(
        text, ["claude-opus-4-5", "claude-haiku-4-5", "claude-sonnet-4-5"]
    )
    costs = [r.cost_usd for r in reports]
    assert costs == sorted(costs)
    assert reports[0].model == "claude-haiku-4-5"


def test_same_prompt_across_models_deduplicates():
    from agentlens.analytics.prompt_score import score_across_models

    reports = score_across_models("fix it", ["claude-sonnet-4-5", "claude-sonnet-4-5", ""])
    assert len(reports) == 1


def test_token_budget_by_type_sums_to_the_prompt_total():
    from agentlens.analytics.prompt_score import content_type_budget, count_tokens

    text = (
        "In @src/auth.py make login() reject expired tokens.\n"
        "```python\ndef login(): ...\n```\n"
        "Please could you also check the edge cases."
    )
    budget = content_type_budget(text)
    assert budget["total_tokens"] == count_tokens(text)
    assert (
        budget["code_tokens"] + budget["path_tokens"] + budget["prose_tokens"]
        == budget["total_tokens"]
    )


def test_token_budget_handles_empty_prompt():
    from agentlens.analytics.prompt_score import content_type_budget

    assert content_type_budget("") == {
        "code_tokens": 0, "path_tokens": 0, "prose_tokens": 0, "total_tokens": 0,
    }


def test_reclaimable_usd_is_the_cleanup_delta():
    from agentlens.analytics.prompt_score import score_prompt

    r = score_prompt("Please could you update @src/api.py so it returns 404. Thanks!")
    assert r.reclaimable_usd == round(r.cost_usd - r.cleaned_cost_usd, 6)
    assert r.reclaimable_usd >= 0


def test_prompt_trends_reports_size_quality_and_heaviest(sessions):
    from agentlens.analytics.prompt_score import prompt_trends

    all_sessions = list(sessions.values())
    trends = prompt_trends(all_sessions)
    assert trends["size_over_time"]
    assert trends["quality_over_time"]
    assert len(trends["heaviest_prompts"]) <= 10
    tokens = [p["tokens"] for p in trends["heaviest_prompts"]]
    assert tokens == sorted(tokens, reverse=True)
    stamps = [p["ts"] for p in trends["size_over_time"]]
    assert stamps == sorted(stamps)
