import json
import math

from agentlens.analytics.detectors.base import Evidence, Finding
from agentlens.analytics.tasks import (
    FAILING, NO_CHANGE, UNKNOWN, UNVALIDATED, VALIDATED, build_tasks, by_outcome,
    task_economics,
)
from agentlens.schema import Event, Provider, Role, ToolInfo, ToolKind, Usage


def _ev(sid, seq, ts, role=Role.ASSISTANT, provider=Provider.CODEX, repo="r1", **kw):
    return Event(session_id=sid, provider=provider, seq=seq, ts=ts, role=role,
                 repo_id=repo, git_branch="main", **kw)


def _session(sid, start_min, *, edit=False, test=None, provider=Provider.CODEX, repo="r1",
             tools=True):
    ts = lambda m: f"2026-03-01T10:{m:02d}:00"  # noqa: E731
    evs = [
        _ev(sid, 1, ts(start_min), Role.USER, provider, repo, text="do the thing"),
        _ev(sid, 2, ts(start_min + 1), provider=provider, repo=repo,
            usage=Usage(input=100, output=50), cost_usd=0.10),
    ]
    if tools:
        evs.append(_ev(sid, 3, ts(start_min + 2), Role.TOOL_CALL, provider, repo,
                       tool=ToolInfo(kind=ToolKind.READ, target_path="a.py")))
    if edit:
        evs.append(_ev(sid, 3, ts(start_min + 2), Role.TOOL_CALL, provider, repo,
                       tool=ToolInfo(kind=ToolKind.EDIT, target_path="a.py")))
    if test is not None:
        evs.append(_ev(sid, 4, ts(start_min + 3), Role.TOOL_CALL, provider, repo,
                       tool=ToolInfo(kind=ToolKind.TEST, exit_code=0 if test else 1)))
    return evs


def _by_label(measures):
    return {m.label: m for m in measures}


def test_empty_history_is_unavailable_not_zero():
    (m,) = task_economics(build_tasks([]))
    assert not m.available and m.value is None


def test_outcomes_are_classified_from_local_evidence():
    tasks = build_tasks([
        _session("ok", 0, edit=True, test=True),
        _session("bad", 0, edit=True, test=False, repo="r2"),
        _session("unv", 0, edit=True, repo="r3"),
        _session("read", 0, repo="r4"),
    ])
    assert {t.task_id: t.outcome for t in tasks} == {
        "ok": VALIDATED, "bad": FAILING, "unv": UNVALIDATED, "read": NO_CHANGE,
    }
    assert by_outcome(tasks) == {VALIDATED: 1, FAILING: 1, UNVALIDATED: 1, NO_CHANGE: 1, UNKNOWN: 0}
    m = _by_label(task_economics(tasks))
    # 2 completed of 3 that changed code; read-only task excluded.
    assert m["Completion rate"].value == 66.7
    assert m["Validation success"].value == 50.0
    assert not m["Rework rate"].available


def test_resumed_session_in_same_repo_branch_is_one_task():
    tasks = build_tasks([_session("a", 0, edit=True), _session("b", 10, edit=True, test=True)])
    assert len(tasks) == 1 and tasks[0].session_ids == ["a", "b"]
    assert tasks[0].outcome == VALIDATED


def test_different_agents_never_merge_into_one_task():
    tasks = build_tasks([_session("a", 0), _session("b", 5, provider=Provider.CLAUDE_CODE)])
    assert len(tasks) == 2


def test_loop_finding_breaks_first_pass_and_counts_in_loop_rate():
    sessions = [_session("x", 0, edit=True, test=True)]
    f = Finding(detector="loop", title="t", prescription="p", mechanism="m",
                evidence=[Evidence(session_id="x", ts="", quote="")])
    tasks = build_tasks(sessions, [f])
    assert tasks[0].completed and not tasks[0].first_pass
    m = _by_label(task_economics(tasks))
    assert m["Loop rate"].value == 100.0 and m["First-pass success"].value == 0.0


def test_no_completed_tasks_and_no_validation_stay_unavailable():
    m = _by_label(task_economics(build_tasks([_session("r", 0)])))
    for label in ("Completion rate", "AI cost / completed task", "First-pass success",
                  "Validation success"):
        assert not m[label].available, label


def test_missing_timestamps_and_manual_sessions_do_not_crash_or_count():
    evs = [_ev("nots", 1, "", Role.USER, text="hi")]
    manual = [_ev("man", 1, "2026-03-01T10:00:00", provider=Provider.MANUAL, cost_usd=5.0)]
    tasks = build_tasks([evs, manual])
    assert [t.task_id for t in tasks] == ["nots"]
    m = _by_label(task_economics(tasks))
    assert not m["Active time / task (min)"].available
    for x in m.values():
        if isinstance(x.value, float):
            assert math.isfinite(x.value)
    json.dumps([x.to_dict() for x in m.values()] + [t.to_dict() for t in tasks])


def test_small_samples_are_tempered():
    m = _by_label(task_economics(build_tasks([_session("ok", 0, edit=True, test=True)])))
    assert m["AI cost / task"].confidence.value != "high"


def test_no_tool_events_is_unknown_not_read_only():
    tasks = build_tasks([_session("blind", 0, tools=False), _session("ok", 0, edit=True,
                                                                     test=True, repo="r2")])
    assert {t.task_id: t.outcome for t in tasks} == {"blind": UNKNOWN, "ok": VALIDATED}
    m = _by_label(task_economics(tasks))
    assert m["Tasks that changed code"].sample_size == 1
    assert m["Completion rate"].value == 100.0


def test_test_result_classified_as_bash_is_paired_with_its_test_call():
    """Older Claude Code ingests: call is TEST, its result is BASH."""
    evs = _session("cc", 0, edit=True, provider=Provider.CLAUDE_CODE)
    evs += [
        _ev("cc", 5, "2026-03-01T10:04:00", Role.TOOL_CALL, Provider.CLAUDE_CODE,
            tool=ToolInfo(kind=ToolKind.TEST)),
        _ev("cc", 6, "2026-03-01T10:05:00", Role.TOOL_RESULT, Provider.CLAUDE_CODE,
            tool=ToolInfo(kind=ToolKind.BASH, exit_code=1)),
    ]
    (task,) = build_tasks([evs])
    assert task.outcome == FAILING and task.validation_failures == 1


def test_idle_gaps_do_not_count_as_task_time():
    evs = [_ev("s", 1, "2026-03-01T10:00:00", Role.USER, text="x"),
           _ev("s", 2, "2026-03-01T10:10:00"),
           _ev("s", 3, "2026-03-04T10:00:00", Role.USER, text="back after days")]
    (task,) = build_tasks([evs])
    assert task.active_min == 10.0


def test_cost_per_completed_task_ignores_unobservable_spend():
    blind = _session("blind", 0, tools=False)
    blind[1].cost_usd = 100.0
    tasks = build_tasks([blind, _session("ok", 0, edit=True, test=True, repo="r2")])
    m = _by_label(task_economics(tasks))
    assert m["AI cost / completed task"].value == 0.1
