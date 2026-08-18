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


# --- multi_ask: split into separate turns, don't relabel the blob ----------
#
# Found from a real report: a three-paragraph, three-topic prompt correctly
# got flagged "Several tasks bundled (3 asks)" by score_prompt, and then
# scaffold()'s "fix" for that diagnosis was nothing — it glued `In @<FILE>, `
# onto the entire unmodified three-paragraph blob and called it a rewrite.
# These pin the actual fix: paragraph-separated multi_ask prompts come back
# as separate, individually sendable turns.


def test_multi_ask_with_paragraph_breaks_splits_into_separate_turns():
    prompt = (
        "Fix the login bug where sessions expire early.\n\n"
        "Also add a dark mode toggle to settings.\n\n"
        "Also update the README to mention the new toggle."
    )
    s = scaffold(prompt, [], None, ["multi_ask"])
    assert s.text.startswith("Send these as separate turns, not one prompt:")
    assert "1. In @<FILE>, Fix the login bug where sessions expire early." in s.text
    assert "2. In @<FILE>, Also add a dark mode toggle to settings." in s.text
    assert "3. In @<FILE>, Also update the README" in s.text
    # One shared placeholder note, not one per repeated occurrence in the text.
    assert s.placeholders == ["<FILE>"]


def test_multi_ask_reuses_a_real_file_hint_across_every_turn(sessions):
    prompt = "Fix the failing check.\n\nAlso clean up the imports."
    s = scaffold(prompt, sessions[LOOPING], None, ["multi_ask"])
    assert s.text.count("verify.py") == 2
    assert "<FILE>" not in s.text
    assert s.filled.get("file", "").endswith("verify.py")


def test_multi_ask_without_paragraph_breaks_falls_back_to_single_intent():
    """A run-on sentence with "also"/"and then" connectors but no blank
    lines has no safe place to cut — splitting mid-sentence would be a
    worse guess than not splitting at all, so this must fall through to
    the ordinary single-intent scaffold rather than mis-splitting."""
    prompt = "Fix the bug and then also update the docs and also run the tests"
    s = scaffold(prompt, [], None, ["multi_ask"])
    assert "Send these as separate turns" not in s.text
    assert s.text.startswith("In @<FILE>, Fix the bug")


def test_multi_ask_real_report_reproduction():
    """The exact input that surfaced this bug — three topics, one per
    paragraph, no session context. Before the fix this returned the whole
    unmodified prompt with `In @<FILE>, ` glued to the front; it must now
    come back as three separate, individually sendable turns."""
    prompt = (
        "mke sure loading screen is shown when hugh amount of data is being "
        "loaded on some machines . Give a date time filter.\n\n"
        "One thing which I want to add is subagents and sub agent auto drive "
        "skills. Not all systems have automatic model routing, for those I "
        "want sub agent workers who automatically use cheaper models. This "
        "should be an automatically available skill and also allow users to "
        "download from the dashboard.\n\n"
        "Also for the current project, check which skills are missing and "
        "add those right now. Additionally, write to the files so it's in "
        "use ahead of time."
    )
    from agentlens.analytics.prompt_score import score_prompt

    r = score_prompt(prompt)
    assert "multi_ask" in {i.code for i in r.issues}
    s = scaffold(prompt, [], None, [i.code for i in r.issues])
    assert s.text.startswith("Send these as separate turns, not one prompt:")
    assert "1. In @<FILE>, mke sure loading screen" in s.text
    assert "2. In @<FILE>, One thing which I want to add" in s.text
    assert "3. In @<FILE>, Also for the current project" in s.text


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
