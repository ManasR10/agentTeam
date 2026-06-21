from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from agents.results import (
    ChangedFileSummary,
    CodingResult,
    ReviewIssue,
    ReviewResult,
)
from agents.state import PlanningResult, RelevantFile
from tools.results import CommandResult
from workflow.errors import WorkflowError
from workflow.models import (
    SUPPORTED_SCHEMA_VERSIONS,
    ApprovalDecision,
    ApprovalRecord,
    CodingAttempt,
    ReviewAttempt,
    RunStatus,
    VerificationRecord,
    WorkflowErrorRecord,
    WorkflowRun,
)

# One place that turns a WorkflowRun into stored JSON and back. Nothing else in
# the codebase should hand-roll JSON for these types. No pickle: persisted state
# is plain validated JSON, never executable.
#
# Encoding is generic (mechanical: dataclass -> dict, datetime -> isoformat,
# etc.). Decoding is explicit per type, because rebuilding needs to know which
# list holds which dataclass and which fields are datetimes/UUIDs/enums. That
# asymmetry is deliberate — the explicit decoders are what make a corrupt field
# produce a clear error instead of a confusing one three layers down.


class SerializationError(WorkflowError):
    """Raised when a run cannot be serialized, or loaded state is unsafe."""


# --- Encoding --------------------------------------------------------------


def _to_jsonable(value: Any) -> Any:
    """Recursively convert a workflow value tree into JSON-serializable data."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):  # before str: StrEnum is a str subclass
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_jsonable(getattr(value, f.name)) for f in fields(value)}
    raise SerializationError(
        f"Cannot serialize value of type {type(value).__name__}"
    )


def serialize_workflow_run(run: WorkflowRun) -> str:
    """Render a WorkflowRun as indented JSON suitable for storage."""
    return json.dumps(_to_jsonable(run), indent=2)


# --- Decoding helpers ------------------------------------------------------


def _req(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise SerializationError(f"Missing required field: {key}")
    return data[key]


def _dt(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise SerializationError(f"Invalid datetime for {field_name}: {value!r}")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise SerializationError(
            f"Corrupt datetime for {field_name}: {value!r}"
        ) from exc


def _uuid(value: Any, field_name: str) -> UUID:
    try:
        return UUID(str(value))
    except (ValueError, AttributeError, TypeError) as exc:
        raise SerializationError(
            f"Invalid UUID for {field_name}: {value!r}"
        ) from exc


def _enum(enum_cls: type[Enum], value: Any, field_name: str) -> Any:
    try:
        return enum_cls(value)
    except ValueError as exc:
        raise SerializationError(f"Unknown {field_name}: {value!r}") from exc


# --- Decoding: nested agent/tool result types ------------------------------


def _relevant_file_from_dict(d: dict[str, Any]) -> RelevantFile:
    return RelevantFile(path=d["path"], reason=d["reason"])


def _planning_result_from_dict(d: dict[str, Any]) -> PlanningResult:
    return PlanningResult(
        task=d["task"],
        repo_summary=d["repo_summary"],
        relevant_files=[_relevant_file_from_dict(x) for x in d["relevant_files"]],
        implementation_plan=list(d["implementation_plan"]),
        files_likely_to_change=list(d["files_likely_to_change"]),
        tests_to_add=list(d["tests_to_add"]),
        risks=list(d["risks"]),
        unknowns=list(d["unknowns"]),
        raw_response=d["raw_response"],
        tool_calls=list(d["tool_calls"]),
        iterations=d["iterations"],
        input_tokens=d["input_tokens"],
        output_tokens=d["output_tokens"],
    )


def _changed_file_summary_from_dict(d: dict[str, Any]) -> ChangedFileSummary:
    return ChangedFileSummary(path=d["path"], reason=d["reason"])


def _coding_result_from_dict(d: dict[str, Any]) -> CodingResult:
    return CodingResult(
        summary=d["summary"],
        reported_changed_files=tuple(
            _changed_file_summary_from_dict(x) for x in d["reported_changed_files"]
        ),
        tests_requested=tuple(d["tests_requested"]),
        known_issues=tuple(d["known_issues"]),
        actual_changed_files=tuple(d["actual_changed_files"]),
        tool_calls=tuple(d["tool_calls"]),
        iterations=d["iterations"],
        input_tokens=d["input_tokens"],
        output_tokens=d["output_tokens"],
    )


def _review_issue_from_dict(d: dict[str, Any]) -> ReviewIssue:
    return ReviewIssue(
        severity=d["severity"],
        path=d["path"],
        description=d["description"],
        required_change=d["required_change"],
    )


def _review_result_from_dict(d: dict[str, Any]) -> ReviewResult:
    return ReviewResult(
        verdict=d["verdict"],
        summary=d["summary"],
        issues=tuple(_review_issue_from_dict(x) for x in d["issues"]),
        tests_assessment=d["tests_assessment"],
    )


def _command_result_from_dict(d: dict[str, Any]) -> CommandResult:
    return CommandResult(
        command_name=d["command_name"],
        argv=tuple(d["argv"]),
        exit_code=d["exit_code"],
        stdout=d["stdout"],
        stderr=d["stderr"],
        timed_out=d["timed_out"],
        duration_seconds=d["duration_seconds"],
        output_truncated=d["output_truncated"],
    )


# --- Decoding: workflow-owned types ----------------------------------------


def _approval_from_dict(d: dict[str, Any]) -> ApprovalRecord:
    return ApprovalRecord(
        approval_id=_uuid(d["approval_id"], "approval.approval_id"),
        run_id=_uuid(d["run_id"], "approval.run_id"),
        decision=_enum(ApprovalDecision, d["decision"], "approval.decision"),
        plan_hash=d["plan_hash"],
        comment=d["comment"],
        created_at=_dt(d["created_at"], "approval.created_at"),
    )


def _error_from_dict(d: dict[str, Any]) -> WorkflowErrorRecord:
    return WorkflowErrorRecord(
        error_type=d["error_type"],
        message=d["message"],
        stage=d["stage"],
        timestamp=_dt(d["timestamp"], "last_error.timestamp"),
        retryable=d["retryable"],
    )


def _coding_attempt_from_dict(d: dict[str, Any]) -> CodingAttempt:
    return CodingAttempt(
        attempt_number=d["attempt_number"],
        result=_coding_result_from_dict(d["result"]),
        started_at=_dt(d["started_at"], "coding_attempt.started_at"),
        completed_at=_dt(d["completed_at"], "coding_attempt.completed_at"),
    )


def _verification_from_dict(d: dict[str, Any]) -> VerificationRecord:
    return VerificationRecord(
        attempt_number=d["attempt_number"],
        command_results=tuple(
            _command_result_from_dict(x) for x in d["command_results"]
        ),
        passed=d["passed"],
        started_at=_dt(d["started_at"], "verification.started_at"),
        completed_at=_dt(d["completed_at"], "verification.completed_at"),
    )


def _review_attempt_from_dict(d: dict[str, Any]) -> ReviewAttempt:
    return ReviewAttempt(
        attempt_number=d["attempt_number"],
        result=_review_result_from_dict(d["result"]),
        started_at=_dt(d["started_at"], "review_attempt.started_at"),
        completed_at=_dt(d["completed_at"], "review_attempt.completed_at"),
    )


# --- Cross-field consistency (checked on load) -----------------------------

# Statuses that can only exist once a plan was produced and should still hold it.
_PLAN_REQUIRED = frozenset(
    {
        RunStatus.AWAITING_PLAN_APPROVAL,
        RunStatus.PLAN_APPROVED,
        RunStatus.PLAN_REJECTED,
        RunStatus.IMPLEMENTING,
        RunStatus.VERIFYING,
        RunStatus.REVIEWING,
        RunStatus.REPAIRING,
        RunStatus.COMPLETED,
        RunStatus.CHANGES_REQUESTED,
    }
)

# Statuses that can only exist once a human approved the plan.
_APPROVAL_REQUIRED = frozenset(
    {
        RunStatus.PLAN_APPROVED,
        RunStatus.IMPLEMENTING,
        RunStatus.VERIFYING,
        RunStatus.REVIEWING,
        RunStatus.REPAIRING,
        RunStatus.COMPLETED,
        RunStatus.CHANGES_REQUESTED,
    }
)


def _validate_stage_consistency(run: WorkflowRun) -> None:
    """Reject persisted state whose status contradicts its contents.

    The field-level invariants live in WorkflowRun.__post_init__; this is the
    cross-field layer that only makes sense once the whole run is assembled.
    """
    status = run.status
    if status in _PLAN_REQUIRED and run.plan is None:
        raise SerializationError(
            f"Inconsistent state: status {status.value} requires a plan."
        )
    if status in _APPROVAL_REQUIRED:
        if run.approval is None:
            raise SerializationError(
                f"Inconsistent state: status {status.value} requires an approval."
            )
        if run.approval.decision is not ApprovalDecision.APPROVED:
            raise SerializationError(
                f"Inconsistent state: status {status.value} requires an "
                "approved plan."
            )
    if status is RunStatus.PLAN_REJECTED:
        if run.approval is None or run.approval.decision is not ApprovalDecision.REJECTED:
            raise SerializationError(
                "Inconsistent state: plan_rejected requires a rejection record."
            )
    if status is RunStatus.COMPLETED and not any(
        attempt.result.verdict == "approved" for attempt in run.review_attempts
    ):
        raise SerializationError(
            "Inconsistent state: completed requires an approved review."
        )


# --- Top-level decode ------------------------------------------------------


def _workflow_run_from_dict(data: dict[str, Any]) -> WorkflowRun:
    version = data.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SerializationError(
            f"Unsupported workflow schema version: {version!r}"
        )

    plan = data.get("plan")
    approval = data.get("approval")
    last_error = data.get("last_error")
    try:
        run = WorkflowRun(
            task=_req(data, "task"),
            workspace_root=_req(data, "workspace_root"),
            starting_git_head=_req(data, "starting_git_head"),
            starting_worktree_clean=_req(data, "starting_worktree_clean"),
            run_id=_uuid(_req(data, "run_id"), "run_id"),
            status=_enum(RunStatus, _req(data, "status"), "status"),
            current_stage=data.get("current_stage"),
            active_stage=data.get("active_stage"),
            schema_version=version,
            created_at=_dt(_req(data, "created_at"), "created_at"),
            updated_at=_dt(_req(data, "updated_at"), "updated_at"),
            plan=_planning_result_from_dict(plan) if plan is not None else None,
            approval=_approval_from_dict(approval) if approval is not None else None,
            coding_attempts=[
                _coding_attempt_from_dict(x)
                for x in data.get("coding_attempts", [])
            ],
            verification_runs=[
                _verification_from_dict(x)
                for x in data.get("verification_runs", [])
            ],
            review_attempts=[
                _review_attempt_from_dict(x)
                for x in data.get("review_attempts", [])
            ],
            changed_files=list(data.get("changed_files", [])),
            review_cycle=data.get("review_cycle", 0),
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            last_error=_error_from_dict(last_error) if last_error is not None else None,
        )
    except (KeyError, TypeError) as exc:
        raise SerializationError(f"Malformed workflow state: {exc}") from exc
    except ValueError as exc:
        # A field-level invariant from WorkflowRun.__post_init__ fired.
        raise SerializationError(f"Invalid workflow state: {exc}") from exc

    _validate_stage_consistency(run)
    return run


def deserialize_workflow_run(raw: str) -> WorkflowRun:
    """Load and validate a WorkflowRun from stored JSON.

    Raises SerializationError on anything wrong: bad JSON, missing fields, an
    unknown enum or schema version, a corrupt timestamp, or a status that
    contradicts the rest of the run.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SerializationError(f"Invalid workflow JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SerializationError("Workflow JSON must be a JSON object.")
    return _workflow_run_from_dict(data)
