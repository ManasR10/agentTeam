from __future__ import annotations

from dataclasses import dataclass

import pytest

from agents import orchestrator
from agents.orchestrator import (
    DirtyWorktreeError,
    enforce_run_limits,
    execute_repo_task,
)
from agents.results import CodingResult, ReviewIssue, ReviewResult
from agents.state import PlanningResult
from tools.mutation_safety import MutationPolicy
from tools.results import CommandResult


def _plan(input_tokens: int = 10, output_tokens: int = 5) -> PlanningResult:
    return PlanningResult(
        task="t",
        repo_summary="s",
        relevant_files=[],
        implementation_plan=["do it"],
        files_likely_to_change=["a.py"],
        tests_to_add=[],
        risks=[],
        unknowns=[],
        raw_response="{}",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _coding(changed=("a.py",), input_tokens=100, output_tokens=40) -> CodingResult:
    return CodingResult(
        summary="did it",
        reported_changed_files=(),
        tests_requested=(),
        known_issues=(),
        actual_changed_files=tuple(changed),
        tool_calls=(),
        iterations=2,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _review(verdict: str) -> ReviewResult:
    if verdict == "approved":
        return ReviewResult("approved", "ok", (), "tests ok")
    return ReviewResult(
        "changes_requested",
        "needs work",
        (ReviewIssue("major", "a.py", "bug", "fix it"),),
        "tests weak",
    )


@dataclass
class _FakeRun:
    input_tokens: int = 30
    output_tokens: int = 10


def _cmd(name: str, exit_code: int) -> CommandResult:
    return CommandResult(name, ("python", name), exit_code, "out", "err",
                         False, 0.1, False)


@pytest.fixture()
def base(monkeypatch):
    """Patch all the moving parts with safe defaults; tests override as needed."""
    monkeypatch.setattr(orchestrator, "plan_repo_task", lambda task: _plan())
    monkeypatch.setattr(orchestrator, "ensure_ready_worktree", lambda s: None)
    monkeypatch.setattr(orchestrator, "run_verification",
                        lambda s: (_cmd("pytest", 0),))
    monkeypatch.setattr(orchestrator, "get_diff_text", lambda s: "the diff")
    monkeypatch.setattr(orchestrator, "enforce_run_limits", lambda p, c, s: None)
    return monkeypatch


def test_dry_run_stops_after_planning(base) -> None:
    called = {"implement": 0}

    def implement(*a, **k):
        called["implement"] += 1
        return _coding()

    base.setattr(orchestrator, "implement_repo_task", implement)

    result = execute_repo_task("t", apply=False)
    assert result.status == "planned"
    assert result.changed_files == ()
    assert called["implement"] == 0
    # planner tokens only
    assert result.input_tokens == 10
    assert result.output_tokens == 5


def test_apply_refuses_dirty_worktree(base) -> None:
    def dirty(_settings):
        raise DirtyWorktreeError("dirty")

    base.setattr(orchestrator, "ensure_ready_worktree", dirty)
    base.setattr(orchestrator, "implement_repo_task", lambda *a, **k: _coding())

    with pytest.raises(DirtyWorktreeError):
        execute_repo_task("t", apply=True)


def test_successful_implementation_reaches_completed(base) -> None:
    base.setattr(orchestrator, "implement_repo_task", lambda *a, **k: _coding())
    base.setattr(orchestrator, "run_reviewer",
                 lambda *a, **k: (_review("approved"), _FakeRun()))

    result = execute_repo_task("t", apply=True)
    assert result.status == "completed"
    assert result.changed_files == ("a.py",)
    assert len(result.reviews) == 1


def test_test_failure_is_supplied_to_reviewer(base) -> None:
    base.setattr(orchestrator, "run_verification",
                 lambda s: (_cmd("pytest", 1),))
    base.setattr(orchestrator, "implement_repo_task", lambda *a, **k: _coding())
    seen = {}

    def fake_reviewer(task, plan, diff, command_results, **kw):
        seen["command_results"] = command_results
        seen["diff"] = diff
        return _review("approved"), _FakeRun()

    base.setattr(orchestrator, "run_reviewer", fake_reviewer)

    execute_repo_task("t", apply=True)
    assert seen["command_results"][0].exit_code == 1  # the failing run is passed
    assert seen["diff"] == "the diff"


def test_changes_requested_triggers_repair_then_completes(base) -> None:
    calls = {"implement": 0, "review": 0}

    def implement(*a, **k):
        calls["implement"] += 1
        return _coding()

    def reviewer(*a, **k):
        calls["review"] += 1
        verdict = "approved" if calls["review"] >= 2 else "changes_requested"
        return _review(verdict), _FakeRun()

    base.setattr(orchestrator, "implement_repo_task", implement)
    base.setattr(orchestrator, "run_reviewer", reviewer)

    result = execute_repo_task("t", apply=True)
    assert result.status == "completed"
    assert calls["implement"] == 2  # initial + one repair
    assert calls["review"] == 2


def test_max_review_cycles_stops_with_changes_requested(base) -> None:
    base.setattr(orchestrator, "implement_repo_task", lambda *a, **k: _coding())
    base.setattr(orchestrator, "run_reviewer",
                 lambda *a, **k: (_review("changes_requested"), _FakeRun()))

    result = execute_repo_task("t", apply=True, max_review_cycles=2)
    assert result.status == "changes_requested"
    assert len(result.reviews) == 2


def test_coder_failure_produces_failed(base) -> None:
    from agents.coder import CodingParseError

    def boom(*a, **k):
        raise CodingParseError("truncated")

    base.setattr(orchestrator, "implement_repo_task", boom)
    result = execute_repo_task("t", apply=True)
    assert result.status == "failed"
    assert "Coder failed" in result.summary


def test_reviewer_failure_produces_failed(base) -> None:
    from agents.reviewer import ReviewParseError

    base.setattr(orchestrator, "implement_repo_task", lambda *a, **k: _coding())

    def boom(*a, **k):
        raise ReviewParseError("bad json")

    base.setattr(orchestrator, "run_reviewer", boom)
    result = execute_repo_task("t", apply=True)
    assert result.status == "failed"
    assert "Reviewer failed" in result.summary


def test_token_usage_accumulates_across_agents(base) -> None:
    base.setattr(orchestrator, "implement_repo_task",
                 lambda *a, **k: _coding(input_tokens=100, output_tokens=40))
    base.setattr(orchestrator, "run_reviewer",
                 lambda *a, **k: (_review("approved"), _FakeRun(30, 10)))

    result = execute_repo_task("t", apply=True)
    # plan(10/5) + coding(100/40) + reviewer(30/10)
    assert result.input_tokens == 140
    assert result.output_tokens == 55


def test_run_limit_files_exceeded_fails(base) -> None:
    base.setattr(
        orchestrator, "implement_repo_task",
        lambda *a, **k: _coding(changed=tuple(f"f{i}.py" for i in range(20))),
    )
    # Use the REAL enforce_run_limits this time (the imported original, not the
    # fixture's patched stub).
    base.setattr(orchestrator, "enforce_run_limits", enforce_run_limits)
    base.setattr(orchestrator, "run_reviewer",
                 lambda *a, **k: (_review("approved"), _FakeRun()))

    result = execute_repo_task("t", apply=True)
    assert result.status == "failed"
    assert "max_files_changed" in result.summary


def test_enforce_run_limits_unit(tmp_path) -> None:
    from config import get_settings
    from dataclasses import replace

    settings = replace(get_settings(), tool_workspace_root=tmp_path)
    policy = MutationPolicy(
        max_files_changed=2,
        max_file_write_chars=1000,
        max_total_write_chars=10,
        allow_create_files=True,
        allow_overwrite_files=True,
    )
    # Too many files.
    assert enforce_run_limits(policy, ["a", "b", "c"], settings) is not None
    # Too many total chars.
    (tmp_path / "big.py").write_text("x" * 50)
    reason = enforce_run_limits(policy, ["big.py"], settings)
    assert reason is not None and "max_total_write_chars" in reason
    # Within limits.
    (tmp_path / "small.py").write_text("x")
    assert enforce_run_limits(policy, ["small.py"], settings) is None
