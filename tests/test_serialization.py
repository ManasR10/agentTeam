from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from agents.results import (
    ChangedFileSummary,
    CodingResult,
    ReviewIssue,
    ReviewResult,
)
from agents.state import PlanningResult, RelevantFile
from tools.results import CommandResult
from workflow.models import (
    ApprovalDecision,
    ApprovalRecord,
    CodingAttempt,
    ReviewAttempt,
    RunStatus,
    VerificationRecord,
    WorkflowErrorRecord,
    WorkflowRun,
)
from workflow.serialization import (
    SerializationError,
    deserialize_workflow_run,
    serialize_workflow_run,
)

_NOW = datetime(2026, 6, 21, 12, 0, 0, tzinfo=timezone.utc)


def _plan() -> PlanningResult:
    return PlanningResult(
        task="add count()",
        repo_summary="tiny todo app",
        relevant_files=[RelevantFile(path="todo.py", reason="holds TodoList")],
        implementation_plan=["add a count method"],
        files_likely_to_change=["todo.py"],
        tests_to_add=["tests/test_todo.py::test_count"],
        risks=["none"],
        unknowns=[],
        raw_response='{"task": "add count()"}',
        tool_calls=["list_files(.)", "read_file(todo.py)"],
        iterations=2,
        input_tokens=100,
        output_tokens=40,
    )


def _coding_result() -> CodingResult:
    return CodingResult(
        summary="added count()",
        reported_changed_files=(ChangedFileSummary(path="todo.py", reason="add method"),),
        tests_requested=("tests/test_todo.py",),
        known_issues=(),
        actual_changed_files=("todo.py",),
        tool_calls=("replace_in_file(todo.py): ok",),
        iterations=3,
        input_tokens=300,
        output_tokens=80,
    )


def _approved_review() -> ReviewResult:
    return ReviewResult(
        verdict="approved",
        summary="looks good",
        issues=(),
        tests_assessment="all pass",
    )


def _command_result() -> CommandResult:
    return CommandResult(
        command_name="pytest",
        argv=("python", "-m", "pytest", "-q"),
        exit_code=0,
        stdout="4 passed",
        stderr="",
        timed_out=False,
        duration_seconds=1.23,
        output_truncated=False,
    )


def _minimal_run(**overrides) -> WorkflowRun:
    base = dict(
        task="add count()",
        workspace_root="/tmp/sample_todo",
        starting_git_head="abc123",
        starting_worktree_clean=True,
    )
    base.update(overrides)
    return WorkflowRun(**base)


def _completed_run() -> WorkflowRun:
    """A run that has been through the whole pipeline, for a rich round-trip."""
    return _minimal_run(
        status=RunStatus.COMPLETED,
        current_stage="review",
        created_at=_NOW,
        updated_at=_NOW,
        plan=_plan(),
        approval=ApprovalRecord(
            approval_id=uuid4(),
            run_id=uuid4(),
            decision=ApprovalDecision.APPROVED,
            plan_hash="deadbeef",
            comment="looks safe",
            created_at=_NOW,
        ),
        coding_attempts=[CodingAttempt(1, _coding_result(), _NOW, _NOW)],
        verification_runs=[
            VerificationRecord(1, (_command_result(),), True, _NOW, _NOW)
        ],
        review_attempts=[ReviewAttempt(1, _approved_review(), _NOW, _NOW)],
        changed_files=["todo.py"],
        review_cycle=1,
        total_input_tokens=400,
        total_output_tokens=120,
    )


def test_minimal_round_trip_is_equal():
    run = _minimal_run(created_at=_NOW, updated_at=_NOW)
    assert deserialize_workflow_run(serialize_workflow_run(run)) == run


def test_full_round_trip_is_equal():
    run = _completed_run()
    restored = deserialize_workflow_run(serialize_workflow_run(run))
    assert restored == run


def test_round_trip_preserves_nested_types():
    restored = deserialize_workflow_run(serialize_workflow_run(_completed_run()))
    # tuples stay tuples, lists stay lists
    assert isinstance(restored.coding_attempts[0].result.actual_changed_files, tuple)
    assert isinstance(restored.verification_runs[0].command_results[0].argv, tuple)
    assert isinstance(restored.changed_files, list)
    assert restored.created_at == _NOW
    assert restored.status is RunStatus.COMPLETED


def test_serialized_form_is_json_object_with_schema_version():
    data = json.loads(serialize_workflow_run(_minimal_run()))
    assert data["schema_version"] == 1
    assert data["status"] == "created"


def test_invalid_json_rejected():
    with pytest.raises(SerializationError):
        deserialize_workflow_run("{not valid json")


def test_non_object_json_rejected():
    with pytest.raises(SerializationError):
        deserialize_workflow_run("[1, 2, 3]")


def test_missing_run_id_rejected():
    data = json.loads(serialize_workflow_run(_minimal_run()))
    del data["run_id"]
    with pytest.raises(SerializationError, match="run_id"):
        deserialize_workflow_run(json.dumps(data))


def test_unsupported_schema_version_rejected():
    data = json.loads(serialize_workflow_run(_minimal_run()))
    data["schema_version"] = 999
    with pytest.raises(SerializationError, match="schema version"):
        deserialize_workflow_run(json.dumps(data))


def test_unknown_enum_rejected():
    data = json.loads(serialize_workflow_run(_minimal_run()))
    data["status"] = "frobnicate"
    with pytest.raises(SerializationError, match="status"):
        deserialize_workflow_run(json.dumps(data))


def test_corrupt_datetime_rejected():
    data = json.loads(serialize_workflow_run(_minimal_run()))
    data["created_at"] = "not-a-date"
    with pytest.raises(SerializationError, match="datetime"):
        deserialize_workflow_run(json.dumps(data))


def test_inconsistent_state_implementing_without_approval_rejected():
    # awaiting_plan_approval with a plan is valid to store; flipping it to
    # implementing makes the missing approval an inconsistency.
    run = _minimal_run(status=RunStatus.AWAITING_PLAN_APPROVAL, plan=_plan())
    data = json.loads(serialize_workflow_run(run))
    data["status"] = "implementing"
    with pytest.raises(SerializationError, match="approval"):
        deserialize_workflow_run(json.dumps(data))


def test_inconsistent_state_awaiting_approval_without_plan_rejected():
    run = _minimal_run()
    data = json.loads(serialize_workflow_run(run))
    data["status"] = "awaiting_plan_approval"  # but plan is None
    with pytest.raises(SerializationError, match="plan"):
        deserialize_workflow_run(json.dumps(data))


def test_completed_without_approved_review_rejected():
    run = _completed_run()
    data = json.loads(serialize_workflow_run(run))
    data["review_attempts"] = []  # completed but no approved review now
    with pytest.raises(SerializationError, match="approved review"):
        deserialize_workflow_run(json.dumps(data))
