from __future__ import annotations

import phase3_demo
from agents.formatting import format_implementation_result
from agents.results import ImplementationRunResult, ReviewIssue, ReviewResult
from agents.orchestrator import DirtyWorktreeError
from agents.state import PlanningResult
from tools.results import CommandResult


def _plan() -> PlanningResult:
    return PlanningResult("t", "s", [], ["step"], ["a.py"], [], [], [], "{}")


def _result(status: str, **kw) -> ImplementationRunResult:
    defaults = dict(
        task="t",
        status=status,
        plan=_plan(),
        changed_files=("a.py",),
        command_results=(CommandResult("pytest", ("pytest",), 0, "", "",
                                       False, 0.1, False),),
        reviews=(ReviewResult("approved", "ok", (), "tests ok"),),
        diff="diff --git a/a.py b/a.py\n+added line\n-removed line\n",
        summary="done",
        input_tokens=100,
        output_tokens=50,
    )
    defaults.update(kw)
    return ImplementationRunResult(**defaults)


def test_no_task_returns_usage_exit_2(capsys) -> None:
    assert phase3_demo.main([]) == 2


def test_dry_run_is_default(monkeypatch) -> None:
    seen = {}

    def fake(task, *, apply):
        seen["apply"] = apply
        return _result("planned")

    monkeypatch.setattr(phase3_demo, "execute_repo_task", fake)
    code = phase3_demo.main(["do", "the", "thing"])
    assert seen["apply"] is False
    assert code == 0


def test_apply_flag_sets_apply(monkeypatch) -> None:
    seen = {}

    def fake(task, *, apply):
        seen["apply"] = apply
        return _result("completed")

    monkeypatch.setattr(phase3_demo, "execute_repo_task", fake)
    assert phase3_demo.main(["task", "--apply"]) == 0
    assert seen["apply"] is True


def test_completed_exit_0(monkeypatch) -> None:
    monkeypatch.setattr(phase3_demo, "execute_repo_task",
                        lambda task, *, apply: _result("completed"))
    assert phase3_demo.main(["task", "--apply"]) == 0


def test_changes_requested_exit_1(monkeypatch) -> None:
    monkeypatch.setattr(phase3_demo, "execute_repo_task",
                        lambda task, *, apply: _result("changes_requested"))
    assert phase3_demo.main(["task", "--apply"]) == 1


def test_failed_exit_1(monkeypatch) -> None:
    monkeypatch.setattr(phase3_demo, "execute_repo_task",
                        lambda task, *, apply: _result("failed"))
    assert phase3_demo.main(["task", "--apply"]) == 1


def test_dirty_worktree_exit_3(monkeypatch) -> None:
    def boom(task, *, apply):
        raise DirtyWorktreeError("dirty")

    monkeypatch.setattr(phase3_demo, "execute_repo_task", boom)
    assert phase3_demo.main(["task", "--apply"]) == 3


def test_unexpected_error_exit_1(monkeypatch) -> None:
    def boom(task, *, apply):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(phase3_demo, "execute_repo_task", boom)
    assert phase3_demo.main(["task", "--apply"]) == 1


# --- formatter -------------------------------------------------------------


def test_format_implementation_result_sections() -> None:
    text = format_implementation_result(_result("completed"))
    for section in ["# DevAgent implementation", "## Status", "## Changed files",
                    "## Verification", "## Review", "## Diff summary",
                    "## Token usage"]:
        assert section in text
    assert "pytest: PASS" in text
    assert "Verdict: approved" in text
    assert "1 insertion(s)" in text
    assert "1 deletion(s)" in text


def test_format_hides_full_diff_by_default() -> None:
    text = format_implementation_result(_result("completed"))
    assert "```diff" not in text
    shown = format_implementation_result(_result("completed"), show_diff=True)
    assert "```diff" in shown
    assert "+added line" in shown


def test_format_shows_review_issues() -> None:
    review = ReviewResult(
        "changes_requested", "needs work",
        (ReviewIssue("major", "a.py", "bug", "fix it"),), "weak",
    )
    text = format_implementation_result(_result("changes_requested",
                                                reviews=(review,)))
    assert "Verdict: changes_requested" in text
    assert "[major] a.py: fix it" in text
