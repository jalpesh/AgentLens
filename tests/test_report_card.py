import xml.dom.minidom

from agentlens.analytics.report_card import card_data, grade, render_svg

RECS = [
    {"source": "prompt", "title": "Use the SECRET-PROJECT repository layer", "cost_usd": 9.0,
     "sessions": 3, "recommendation": "Move it to AGENTS.md."},
    {"source": "waste", "title": "Huge command output flooding the context window",
     "cost_usd": 4.0, "sessions": 2, "recommendation": "Pipe it: | tail -40."},
]
TYPES = [{"task_type": "feature", "pct": 60.0, "cost_per_task": 2.5},
         {"task_type": "bug fix", "pct": 40.0, "cost_per_task": 1.25}]
TASKS = [{"cost_usd": c} for c in (1.0, 2.0, 5.0)]


def _data():
    return card_data({"cost_usd": 20.0, "sessions": 4}, 3.0, RECS, TYPES, TASKS,
                     [{"provider": "claude-code", "sessions": 4, "cost_usd": 20.0}],
                     [{"date": "2026-04-02"}, {"date": "2026-11-20"}])


def test_card_numbers_and_grade():
    d = _data()
    assert (d["spend"], d["avoidable_pct"], d["grade"], d["tasks"], d["median_task"]) == \
        (20.0, 15.0, "D", 3, 2.0)
    assert d["period"] == "Apr – Nov 2026" and d["agents"] == ["Claude Code"]
    assert [grade(p) for p in (0, 7, 12, 20, 40)] == ["A", "B", "C", "D", "F"]


def test_habit_never_comes_from_the_users_own_sentences():
    d = _data()
    assert d["habit"]["title"].startswith("Huge command output")
    assert "SECRET-PROJECT" not in render_svg(d)


def test_svg_is_valid_xml_and_hide_money_removes_dollars():
    d = _data()
    xml.dom.minidom.parseString(render_svg(d))
    hidden = render_svg(d, hide_money=True)
    xml.dom.minidom.parseString(hidden)
    assert "$" in render_svg(d) and "$" not in hidden


def test_empty_data_still_renders():
    d = card_data({"cost_usd": 0, "sessions": 0}, 0.0, [], [], [], [], [])
    xml.dom.minidom.parseString(render_svg(d))
    assert d["habit"] is None and d["median_task"] is None


def test_static_export_keeps_the_card_a_valid_svg(tmp_path):
    import json

    from agentlens.schema import Event, Provider, Role, Usage
    from agentlens.store import Store
    from agentlens.web.server import _MARKER, export_html  # noqa: F401
    db = tmp_path / "db.sqlite"
    with Store(db) as st:
        st.add_events([Event(session_id="s", provider=Provider.CLAUDE_CODE, seq=1,
                             ts="2026-11-01T10:00:00", role=Role.ASSISTANT,
                             model="claude-opus-5-5", usage=Usage(input=1000), cost_usd=0.01,
                             cwd="/home/dev/secret-proj")])
    html = export_html(str(tmp_path / "out.html"), db=str(db)).read_text(encoding="utf-8")
    line = next(ln for ln in html.splitlines() if ln.startswith("const AGENTLENS_STATIC_DATA"))
    data = json.loads(line.split("=", 1)[1].rstrip().rstrip(";"))
    svg = data["report_card"]["svg"]
    xml.dom.minidom.parseString(svg)
    assert "REDACTED" not in svg and "secret-proj" not in svg
