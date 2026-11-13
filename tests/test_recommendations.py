import json

from agentlens.analytics.recommendations import recommend


def _finding(usd, sids=("a", "b", "c", "d", "e"), **kw):
    return {"detector": "loop", "title": "Retry loop", "mechanism": "Same error re-sent.",
            "advice": "Paste the error once and ask for a root cause.", "wasted_usd": usd,
            "wasted_tokens": 1000, "occurrences": len(sids),
            "evidence": [{"session_id": s, "quote": f"q{s}"} for s in sids], **kw}


OPP = {"kind": "reread", "title": "Unchanged files re-read", "recommendation": "Ask for a diff.",
       "occurrences": 4, "sessions": 6, "tokens": 50_000, "cost_usd": 0.0, "examples": ["a.py"]}
PAT = {"kind": "repeated_instruction", "title": "Use the repo layer", "problem": "Typed 3x.",
       "recommendation": "Move it to AGENTS.md.", "confidence": "low", "sessions": 3,
       "tokens": 40, "cost_usd": 0.5, "examples": ["Use the repo layer"]}


def test_merges_all_sources_ranked_by_money_then_tokens():
    recs = recommend([_finding(2.0)], [OPP], [PAT])
    assert [r["source"] for r in recs] == ["waste", "prompt", "context"]
    assert all({"problem", "evidence", "recommendation", "expected_impact", "confidence",
                "sessions", "tokens", "cost_usd"} <= r.keys() for r in recs)


def test_unpriced_saving_is_unknown_not_zero():
    r = recommend([], [OPP], [])[0]
    assert r["cost_usd"] is None and "cost unknown" in r["expected_impact"]


def test_no_tokens_no_money_says_insufficient_data():
    r = recommend([_finding(0.0, wasted_tokens=0)], [], [])[0]
    assert r["expected_impact"].startswith("Insufficient data")


def test_thin_evidence_lowers_confidence():
    assert recommend([_finding(1.0)], [], [])[0]["confidence"] == "medium"
    assert recommend([_finding(1.0, sids=("a",))], [], [])[0]["confidence"] == "low"


def test_limit_and_serializes():
    recs = recommend([_finding(float(i)) for i in range(20)], [], [], limit=5)
    assert len(recs) == 5 and recs[0]["cost_usd"] == 19.0
    json.dumps(recs)


def test_session_count_survives_the_five_item_evidence_cap():
    from agentlens.analytics.detectors.base import Evidence, Finding, merge
    fs = [Finding("x", "t", "p", "m", wasted_usd=1.0, occurrences=1,
                  evidence=[Evidence(session_id=f"s{i}", ts="", quote="q")]) for i in range(8)]
    f = merge(fs)[0]
    assert len(f.evidence) == 5 and f.sessions == 8
    assert recommend([f.to_dict()], [], [])[0]["sessions"] == 8
