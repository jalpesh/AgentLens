"""Agents & Skills tests.

The rule this file exists to enforce: **no suggestion without a triggering
finding or a fingerprint fact**. The moment that slips, this feature becomes a
list of generic advice anyone could have written, and the reason to run the
tool disappears.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.analytics.detectors import run_all  # noqa: E402
from agentlens.analytics.fingerprint import ProjectProfile, fingerprint_all  # noqa: E402
from agentlens.analytics.rollups import engineering_metrics  # noqa: E402
from agentlens.analytics.skills import generic_suggestions, get, suggest  # noqa: E402
from generate import build  # noqa: E402


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    paths = build(tmp_path_factory.mktemp("cc") / "projects")
    a = ClaudeCodeAdapter()
    sessions = [list(a.parse(SourceFile.of(p))) for p in paths]
    findings = run_all(sessions)
    profile = fingerprint_all(sessions)
    return sessions, findings, profile


# --- fingerprint ------------------------------------------------------------


def test_fingerprint_detects_language_and_test_command(world):
    _, _, profile = world
    assert profile.primary_language == "python"
    assert profile.test_command and "pytest" in profile.test_command


def test_fingerprint_reports_its_own_confidence(world):
    """A profile built from four events is a guess. The UI must be able to say
    so rather than presenting it with the same weight as a solid inference."""
    _, _, profile = world
    assert profile.confident is True

    thin = ProjectProfile(languages=["python"], sample_events=3)
    assert thin.confident is False


def test_fingerprint_needs_no_filesystem_access(world, monkeypatch):
    """The profile comes from stored events only. If it ever started reading
    the disk, the offline/privacy story would need re-justifying."""
    sessions, _, _ = world

    def explode(*a, **k):
        raise AssertionError("fingerprint must not touch the filesystem")

    monkeypatch.setattr(Path, "exists", explode)
    monkeypatch.setattr(Path, "open", explode)
    assert fingerprint_all(sessions).primary_language == "python"


# --- the hard rule ----------------------------------------------------------


def test_every_suggestion_names_its_trigger(world):
    _, findings, profile = world
    suggestions = suggest(findings, profile)
    assert suggestions
    for s in suggestions:
        assert s.triggered_by, f"{s.id} has no trigger — that makes it generic advice"


def test_no_findings_means_almost_no_suggestions():
    """With nothing detected there is nothing to justify, so the list must
    collapse — only fingerprint-derived items may survive."""
    profile = ProjectProfile(languages=["python"], sample_events=50)
    suggestions = suggest([], profile)
    assert all(s.triggered_by.startswith("fingerprint:") for s in suggestions)


def test_no_findings_and_no_profile_means_nothing_at_all():
    assert suggest([], ProjectProfile()) == []


def test_suggestions_cite_the_dollar_cost(world):
    _, findings, profile = world
    s = get(suggest(findings, profile), "post-edit-tests")
    assert s is not None
    assert "$" in s.why
    assert s.evidence_usd > 0


def test_suggestions_ranked_by_evidence(world):
    _, findings, profile = world
    costs = [s.evidence_usd for s in suggest(findings, profile)]
    assert costs == sorted(costs, reverse=True)


# --- generated artifacts ----------------------------------------------------


def test_hook_artifact_matches_the_documented_schema(world):
    """Verified against current Claude Code hooks docs. A plausible-looking but
    wrong hook produces a file that silently never fires, which is worse than
    generating nothing — the user believes they're covered."""
    _, findings, profile = world
    s = get(suggest(findings, profile), "post-edit-tests")
    payload = json.loads(s.artifact.content)

    assert "hooks" in payload
    entry = payload["hooks"]["PostToolUse"][0]
    assert entry["matcher"] == "Edit|Write"
    handler = entry["hooks"][0]
    assert handler["type"] == "command"
    assert profile.test_command in handler["command"]
    assert s.artifact.path == ".claude/settings.json"


def test_skill_artifact_has_valid_frontmatter(world):
    _, findings, profile = world
    s = get(suggest(findings, profile), "recurring-fix-skill")
    content = s.artifact.content
    assert content.startswith("---\n")
    assert "description:" in content.split("---")[1]
    assert s.artifact.path.startswith(".claude/skills/")
    assert s.artifact.path.endswith("/SKILL.md")


def test_command_artifact_goes_where_commands_live(world):
    _, findings, profile = world
    s = get(suggest(findings, profile), "debug-command")
    assert s.artifact.path == ".claude/commands/debug.md"
    assert s.artifact.content.startswith("---\n")


def test_generated_json_is_human_readable(world):
    """These files get opened and read. Escaped unicode looks like a bug."""
    _, findings, profile = world
    for s in suggest(findings, profile):
        if s.artifact and s.artifact.language == "json":
            assert "\\u" not in s.artifact.content


def test_unverified_packages_are_flagged_not_invented(world):
    """AgentLens must not name an MCP server it hasn't verified exists.
    Recommending an imaginary package is the same credibility failure as
    claiming full support for an unverified log format."""
    _, findings, profile = world
    s = get(suggest(findings, profile), "code-index-mcp")
    if s is None:
        pytest.skip("whole_file_reread not present in this fixture set")
    assert s.unverified_category is True
    assert "<mcp-server-command>" in s.artifact.content


def test_no_suggestion_hardcodes_a_package_name(world):
    """A blunt guard: if a future edit adds a specific npm/pypi package to a
    generated MCP config, it must come with verification — this test is the
    speed bump that forces that conversation."""
    _, findings, profile = world
    for s in suggest(findings, profile):
        if s.kind == "mcp" and not s.unverified_category:
            raise AssertionError(
                f"{s.id} names a concrete MCP package without being marked verified"
            )


# --- the generic bucket (Phase 4, section D) --------------------------------


def test_generic_section_never_sets_triggered_by():
    """The whole point of this bucket: it is not derived from the user's
    data, so it must never claim a receipt it doesn't have."""
    for s in generic_suggestions():
        assert s.triggered_by == ""


def test_generic_section_is_present_and_has_three_verified_tools():
    ids = {s.id for s in generic_suggestions()}
    assert ids == {"generic-ast-grep", "generic-codegraph", "generic-codebase-memory"}


def test_generic_section_only_names_verified_packages():
    """Every package here was checked to actually exist this session — see
    PLAN_PHASE4.md's "Verified this session" list. `code-review-graph` is
    deliberately absent: it's ambiguous (several unrelated packages share
    the name), so it never got a real command."""
    for s in generic_suggestions():
        assert "code-review-graph" not in (s.artifact.content if s.artifact else "")


def test_triggered_suggestions_never_include_generic_only_packages_unless_organic(world):
    """The three generic-bucket packages must not leak into the *triggered*
    list unless something in the user's own data actually triggered them —
    `discovery-tools` (metric-triggered) is the one legitimate exception,
    and it names its own trigger explicitly."""
    _, findings, profile = world
    for s in suggest(findings, profile):
        if s.id in {"generic-ast-grep", "generic-codegraph", "generic-codebase-memory"}:
            raise AssertionError(f"{s.id} leaked into the triggered suggestion list")


def test_both_sections_present_given_a_rich_fixture(world):
    _, findings, profile = world
    triggered = suggest(findings, profile)
    generic = generic_suggestions()
    assert triggered, "the rich fixture should trigger at least one suggestion"
    assert generic, "the generic bucket must always be present"
    assert all(s.triggered_by for s in triggered)
    assert all(not s.triggered_by for s in generic)


def test_discovery_share_triggers_a_metric_suggestion():
    """Section E ties the discovery-share metric into a triggered suggestion
    for the same Engineering-tab tile — same `Suggestion` object, second
    render location, no duplicate logic."""
    hot = suggest([], ProjectProfile(), metrics={"discovery_share_pct": 45.0})
    s = get(hot, "discovery-tools")
    assert s is not None
    assert s.triggered_by == "metric:discovery_share"
    assert "45%" in s.why

    cold = suggest([], ProjectProfile(), metrics={"discovery_share_pct": 5.0})
    assert get(cold, "discovery-tools") is None


# --- engineering metrics (Phase 4, section E) -------------------------------


def test_engineering_metrics_are_a_pure_rollup(world):
    """No new detector — these numbers come straight out of tool-call
    events and existing findings."""
    sessions, findings, _ = world
    m = engineering_metrics(sessions, findings)
    assert 0 <= m["discovery_share_pct"] <= 100
    assert m["files_opened"] >= 0
    assert m["loop_cost_usd"] >= 0


def test_engineering_metrics_empty_scope_is_all_zero():
    assert engineering_metrics([], []) == {
        "discovery_share_pct": 0.0,
        "total_agent_steps": 0,
        "searches": {"total": 0, "project_search": 0, "shell_grep": 0},
        "files_opened": 0,
        "files_opened_repeat": 0,
        "loop_cost_usd": 0.0,
    }
