from __future__ import annotations

from agents.planner import parse_planning_result
from phase2_demo import main
from tests.fixtures import VALID_PLANNER_JSON


def test_phase2_demo_success(monkeypatch, capsys) -> None:
    fake_result = parse_planning_result(VALID_PLANNER_JSON)
    monkeypatch.setattr("phase2_demo.plan_repo_task", lambda task: fake_result)

    exit_code = main(["Add CLI"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Running DevAgent Phase 2 planning demo" in captured.out
    assert "DevAgent Implementation Plan" in captured.out
    assert "Create cli.py" in captured.out


def test_phase2_demo_no_task_prints_usage_without_api_call(monkeypatch, capsys) -> None:
    def fail_if_called(task: str):
        raise AssertionError("plan_repo_task must not be called without a task")

    monkeypatch.setattr("phase2_demo.plan_repo_task", fail_if_called)

    exit_code = main([])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "Usage:" in captured.out
    assert "Running DevAgent" not in captured.out


def test_phase2_demo_failure(monkeypatch, capsys) -> None:
    def fake_plan_repo_task(task: str):
        raise RuntimeError("boom")

    monkeypatch.setattr("phase2_demo.plan_repo_task", fake_plan_repo_task)

    exit_code = main(["Add CLI"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Phase 2 demo failed" in captured.out
    assert "boom" in captured.out
