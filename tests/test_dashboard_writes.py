"""The dashboard's two write endpoints: dismiss-a-finding and manual-session
import. Both are local-only (SQLite, never a file write, never off-machine —
see SECURITY.md's "The two endpoints that write" section) and both are
add-only by design. These tests cover the `Store` methods that back them and
`build_payload`'s use of them — not the raw HTTP handlers, which are thin
wrappers over the same `Store` calls tested here plus JSON parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fixtures"))

from agentlens.adapters import ClaudeCodeAdapter  # noqa: E402
from agentlens.adapters.base import SourceFile  # noqa: E402
from agentlens.analytics.detectors.base import Evidence, Finding  # noqa: E402
from agentlens.schema import Provider, Usage  # noqa: E402
from agentlens.store import Store  # noqa: E402
from agentlens.web.server import _apply_dismissals, _exclude_manual, build_payload  # noqa: E402
from generate import build  # noqa: E402

#: A real loop-finding session from the shared fixture set, same one
#: `test_statusline.py` uses — guaranteed to produce at least one finding
#: with real evidence to dismiss.
_LOOPING = "66666666-6666-4666-8666-666666666666"


@pytest.fixture()
def seeded_db(tmp_path) -> Path:
    paths = build(tmp_path / "projects")
    adapter = ClaudeCodeAdapter()
    db = tmp_path / "t.db"
    with Store(db) as s:
        for p in paths:
            s.add_events(adapter.parse(SourceFile.of(p)))
    return db


# --- dismiss: Store layer ---------------------------------------------------


def test_dismiss_evidence_is_idempotent(tmp_path):
    """Dismissing the same evidence twice (a double-click, a stale second
    tab re-POSTing) must not create two rows or error — `dismiss_key` is a
    stable hash precisely so `INSERT OR REPLACE` collapses repeats."""
    with Store(tmp_path / "t.db") as s:
        s.dismiss_evidence("loop", "sess-1", "2026-01-01T00:00:00", "deliberate")
        s.dismiss_evidence("loop", "sess-1", "2026-01-01T00:00:00", "deliberate")
        assert s.dismissed_count() == 1
        assert ("loop", "sess-1", "2026-01-01T00:00:00") in s.dismissed_keys()


def test_dismiss_evidence_distinct_keys_are_independent(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.dismiss_evidence("loop", "sess-1", "2026-01-01T00:00:00")
        s.dismiss_evidence("target_churn", "sess-1", "2026-01-01T00:00:00")
        s.dismiss_evidence("loop", "sess-2", "2026-01-01T00:00:00")
        assert s.dismissed_count() == 3


# --- dismiss: _apply_dismissals ---------------------------------------------


def _finding(evidence: list[Evidence]) -> Finding:
    return Finding(
        detector="loop",
        title="t",
        prescription="p",
        mechanism="m",
        wasted_usd=sum(e.cost_usd for e in evidence),
        wasted_tokens=sum(e.tokens for e in evidence),
        occurrences=len(evidence),
        evidence=evidence,
    )


def test_apply_dismissals_drops_only_the_matched_evidence():
    kept_ev = Evidence(session_id="s1", ts="t1", quote="q1", cost_usd=1.0, tokens=100)
    dropped_ev = Evidence(session_id="s2", ts="t2", quote="q2", cost_usd=2.0, tokens=200)
    f = _finding([kept_ev, dropped_ev])

    out = _apply_dismissals([f], {("loop", "s2", "t2")})

    assert len(out) == 1
    assert [e.session_id for e in out[0].evidence] == ["s1"]
    # Recomputed, not left stale at the pre-dismissal total.
    assert out[0].wasted_usd == pytest.approx(1.0)
    assert out[0].wasted_tokens == 100
    assert out[0].occurrences == 1


def test_apply_dismissals_drops_the_whole_finding_when_all_evidence_is_gone():
    ev = Evidence(session_id="s1", ts="t1", quote="q1", cost_usd=1.0, tokens=100)
    f = _finding([ev])

    out = _apply_dismissals([f], {("loop", "s1", "t1")})

    assert out == []


def test_apply_dismissals_is_a_noop_when_nothing_matches():
    ev = Evidence(session_id="s1", ts="t1", quote="q1", cost_usd=1.0, tokens=100)
    f = _finding([ev])

    out = _apply_dismissals([f], {("loop", "some-other-session", "t9")})

    assert out == [f]


# --- dismiss: end-to-end through build_payload ------------------------------


def test_dismissing_a_findings_only_evidence_removes_it_from_the_payload(seeded_db):
    payload = build_payload(str(seeded_db))
    loop_findings = [f for f in payload["findings"] if f["detector"] == "loop"]
    assert loop_findings, "fixture must produce at least one loop finding to dismiss"
    finding = loop_findings[0]
    assert payload["dismissed_count"] == 0

    with Store(str(seeded_db)) as s:
        for e in finding["evidence"]:
            s.dismiss_evidence("loop", e["session_id"], e["ts"], "not actually a problem")

    after = build_payload(str(seeded_db))
    assert after["dismissed_count"] == len(finding["evidence"])
    after_loop = [f for f in after["findings"] if f["detector"] == "loop"]
    # Either the finding is gone entirely, or every remaining piece of
    # evidence is something that wasn't dismissed above.
    dismissed_ts = {e["ts"] for e in finding["evidence"]}
    for f in after_loop:
        assert not any(e["ts"] in dismissed_ts for e in f["evidence"])


# --- manual sessions: Store layer -------------------------------------------


def test_add_manual_session_round_trips(tmp_path):
    with Store(tmp_path / "t.db") as s:
        s.add_manual_session(
            "manual-abc123", "2026-01-01T00:00:00", "cursor-frontend",
            "gpt-5.1", Usage(input=1000, output=200), 0.42,
        )
        sessions = s.manual_sessions()
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "manual-abc123"
    assert sessions[0]["label"] == "cursor-frontend"
    assert sessions[0]["cost_usd"] == pytest.approx(0.42)


def test_remove_manual_session_only_deletes_manual_provider_rows(seeded_db):
    """The safety rail: `remove_manual_session` must never be able to delete
    real ingested history, even if pointed at a real session id by mistake —
    it's scoped to `provider = 'manual'` in the SQL itself, not just by
    convention."""
    with Store(str(seeded_db)) as s:
        real_ids = s.session_ids()
        assert real_ids, "fixture must have real ingested sessions"
        real_id = real_ids[0]

        removed = s.remove_manual_session(real_id)
        assert removed == 0
        assert real_id in s.session_ids(), "a real session must survive an attempted removal"

        s.add_manual_session(
            "manual-xyz", "2026-01-01T00:00:00", "some-tool", None, Usage(), 0.0,
        )
        assert s.remove_manual_session("manual-xyz") == 1
        assert "manual-xyz" not in [r["session_id"] for r in s.manual_sessions()]


# --- manual sessions: excluded from detector/profile input -----------------


def test_exclude_manual_drops_only_manual_sessions():
    """Direct test of the filter `build_payload` relies on to keep manual
    entries out of detector/profile input — a single synthetic event can't
    trip a real detector either way, so a test that only checks "no finding
    appeared" would pass even if this filter were deleted entirely. This is
    the one that would actually catch that regression."""
    from agentlens.schema import Event, Role

    real = [Event(
        session_id="s1", provider=Provider.CLAUDE_CODE, seq=1,
        ts="2026-01-01T00:00:00", role=Role.ASSISTANT,
    )]
    manual = [Event(
        session_id="s2", provider=Provider.MANUAL, seq=1,
        ts="2026-01-01T00:00:00", role=Role.ASSISTANT,
    )]

    out = _exclude_manual([real, manual])

    assert out == [real]


# --- manual sessions: never produce a detector finding ----------------------


def test_manual_session_is_counted_in_totals_but_produces_no_finding(seeded_db):
    before = build_payload(str(seeded_db))
    before_findings = len(before["findings"])
    before_cost = before["totals"]["cost_usd"]

    with Store(str(seeded_db)) as s:
        # A large cost on purpose — if a manual session could leak into
        # detector input, a big enough number is likelier to visibly perturb
        # something (e.g. cost-based sorting) than a tiny one would.
        s.add_manual_session(
            "manual-big", "2026-01-01T00:00:00", "big-manual-tool",
            "gpt-5.1", Usage(input=500_000, output=50_000), 25.00,
        )

    after = build_payload(str(seeded_db))

    assert len(after["findings"]) == before_findings, (
        "a manual, turn-by-turn-data-free session must never add a finding"
    )
    assert after["totals"]["cost_usd"] == pytest.approx(before_cost + 25.00)
    assert any(s["provider"] == Provider.MANUAL.value for s in after["sessions"])
    assert "manual" in after["providers"]


# --- CLI: `agentlens sessions --remove` -------------------------------------


def test_cli_sessions_remove_deletes_a_manual_session(tmp_path):
    from agentlens.cli import main as cli_main

    db = tmp_path / "t.db"
    with Store(db) as s:
        s.add_manual_session(
            "manual-cli-1", "2026-01-01T00:00:00", "some-tool", None, Usage(), 1.0,
        )

    assert cli_main(["sessions", "--db", str(db), "--remove", "manual-cli-1"]) == 0
    with Store(db) as s:
        assert s.manual_sessions() == []


def test_cli_sessions_remove_refuses_unknown_id(tmp_path):
    from agentlens.cli import main as cli_main

    db = tmp_path / "t.db"
    Store(db).close()  # create an empty db so the command has something to open
    assert cli_main(["sessions", "--db", str(db), "--remove", "not-a-real-id"]) == 1
