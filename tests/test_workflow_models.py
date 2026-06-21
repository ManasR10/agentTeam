from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.results import CodingResult, ReviewIssue, ReviewResult
from workflow.models import (
    SUPPORTED_SCHEMA_VERSIONS,
    ApprovalDecision,
    CodingAttempt,
    ReviewAttempt,
    RunStatus,
    WorkflowRun,
)


def _run(**overrides) -> WorkflowRun:
    base = dict(
        task="Add a count() method",
        workspace_root="/tmp/sample_repo",
        starting_git_head="abc123",
        starting_worktree_clean=True,
    )
    base.update(overrides)
    return WorkflowRun(**base)


def _coding_result() -> CodingResult:
    return CodingResult(
        summary="added count()",
        reported_changed_files=(),
        tests_requested=(),
        known_issues=(),
    )


def test_create_valid_run_defaults():
    run = _run()
    assert run.status is RunStatus.CREATED
    assert run.current_stage is None
    assert run.schema_version in SUPPORTED_SCHEMA_VERSIONS
    assert run.plan is None
    assert run.approval is None
    assert run.coding_attempts == []
    assert run.changed_files == []


def test_run_id_is_unique():
    assert _run().run_id != _run().run_id


def test_timestamps_are_utc():
    run = _run()
    assert run.created_at.tzinfo is timezone.utc
    assert run.updated_at.tzinfo is timezone.utc


def test_empty_task_rejected():
    with pytest.raises(ValueError):
        _run(task="   ")


def test_blank_workspace_rejected():
    with pytest.raises(ValueError):
        _run(workspace_root="")


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        _run(total_input_tokens=-1)
    with pytest.raises(ValueError):
        _run(total_output_tokens=-5)


def test_negative_review_cycle_rejected():
    with pytest.raises(ValueError):
        _run(review_cycle=-1)


def test_unsupported_schema_version_rejected():
    with pytest.raises(ValueError):
        _run(schema_version=0)
    with pytest.raises(ValueError):
        _run(schema_version=999)


def test_run_status_round_trips_through_value():
    for status in RunStatus:
        assert RunStatus(status.value) is status


def test_approval_decision_values():
    assert ApprovalDecision("approved") is ApprovalDecision.APPROVED
    assert ApprovalDecision("rejected") is ApprovalDecision.REJECTED


def test_coding_attempts_are_appended_not_overwritten():
    run = _run()
    now = datetime.now(timezone.utc)
    run.coding_attempts.append(
        CodingAttempt(1, _coding_result(), started_at=now, completed_at=now)
    )
    run.coding_attempts.append(
        CodingAttempt(2, _coding_result(), started_at=now, completed_at=now)
    )
    assert [a.attempt_number for a in run.coding_attempts] == [1, 2]


def test_review_attempts_preserve_history():
    run = _run()
    now = datetime.now(timezone.utc)
    approved = ReviewResult(
        verdict="approved", summary="ok", issues=(), tests_assessment="pass"
    )
    needs_work = ReviewResult(
        verdict="changes_requested",
        summary="fix it",
        issues=(
            ReviewIssue(
                severity="major",
                path="todo.py",
                description="missing edge case",
                required_change="handle empty list",
            ),
        ),
        tests_assessment="pass",
    )
    run.review_attempts.append(ReviewAttempt(1, needs_work, now, now))
    run.review_attempts.append(ReviewAttempt(2, approved, now, now))
    assert len(run.review_attempts) == 2
    assert run.review_attempts[0].result.verdict == "changes_requested"
    assert run.review_attempts[-1].result.verdict == "approved"
