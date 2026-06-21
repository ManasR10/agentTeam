from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from uuid import uuid4

from agents.state import PlanningResult
from workflow.errors import WorkflowError
from workflow.events import EventType, redact_event_payload
from workflow.models import ApprovalDecision, ApprovalRecord, RunStatus, WorkflowRun
from workflow.serialization import _to_jsonable
from workflow.transitions import transition_run

if TYPE_CHECKING:
    from storage.base import RunStore

# Approval is a real, persisted boundary: planning stops, the plan is shown, and
# a human approves THIS plan before any coder tool exists. Approval and
# execution are separate actions — approve_plan only moves the run to
# PLAN_APPROVED; a later resume runs the coder.


class ApprovalError(WorkflowError):
    """Base class for approval-gate failures."""


class ApprovalNotAllowedError(ApprovalError):
    """Raised when a run isn't in a state where a plan can be decided."""


class ApprovalMismatchError(ApprovalError):
    """Raised when the stored approval doesn't match the current plan.

    This is the guard that stops an approval of Plan A from silently carrying
    over to a Plan B that changed afterwards.
    """


def calculate_plan_hash(plan: PlanningResult) -> str:
    """A deterministic SHA-256 over the plan's canonical JSON form.

    Uses the same value tree as storage (sorted keys, no whitespace) so the hash
    is stable across processes and depends only on the plan's content.
    """
    canonical = json.dumps(
        _to_jsonable(plan),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_approval_matches_plan(run: WorkflowRun) -> None:
    """Raise unless the run carries an approval for its current plan.

    Called at the start of implementation: if the plan was edited after
    approval, the hashes diverge and we refuse rather than build the wrong plan.
    """
    if run.approval is None:
        raise ApprovalMismatchError("Run has no approval on record.")
    if run.plan is None:
        raise ApprovalMismatchError("Run has an approval but no plan.")
    current = calculate_plan_hash(run.plan)
    if current != run.approval.plan_hash:
        raise ApprovalMismatchError(
            "Plan changed since approval; re-approval (or replanning) required."
        )


def _require_decidable(run: WorkflowRun) -> None:
    if run.status is not RunStatus.AWAITING_PLAN_APPROVAL:
        raise ApprovalNotAllowedError(
            f"Run is {run.status.value}; only a plan awaiting approval can be "
            "approved or rejected."
        )
    if run.plan is None:
        raise ApprovalNotAllowedError("Run has no plan to decide on.")


def _decide(
    store: RunStore,
    run_id: str,
    decision: ApprovalDecision,
    new_status: RunStatus,
    event_type: EventType,
    comment: str | None,
) -> WorkflowRun:
    run = store.load_run(run_id)
    _require_decidable(run)

    approval = ApprovalRecord(
        approval_id=uuid4(),
        run_id=run.run_id,
        decision=decision,
        plan_hash=calculate_plan_hash(run.plan),
        comment=comment,
        created_at=datetime.now(timezone.utc),
    )
    run.approval = approval
    transition_run(run, new_status, current_stage=new_status.value)
    # One transaction: run state + approvals row + audit event land together, so
    # a crash can't leave an approved run without its audit trail.
    store.commit_approval(
        run,
        approval,
        event_type=event_type,
        event_payload=redact_event_payload(
            {"plan_hash": approval.plan_hash, "comment": comment}
        ),
    )
    return run


def approve_plan(
    store: RunStore,
    run_id: str,
    *,
    comment: str | None = None,
) -> WorkflowRun:
    """Approve a run's plan and move it to PLAN_APPROVED. Does not run the coder."""
    return _decide(
        store,
        run_id,
        ApprovalDecision.APPROVED,
        RunStatus.PLAN_APPROVED,
        EventType.APPROVAL_GRANTED,
        comment,
    )


def reject_plan(
    store: RunStore,
    run_id: str,
    *,
    comment: str | None = None,
) -> WorkflowRun:
    """Reject a run's plan and move it to PLAN_REJECTED (terminal)."""
    return _decide(
        store,
        run_id,
        ApprovalDecision.REJECTED,
        RunStatus.PLAN_REJECTED,
        EventType.APPROVAL_REJECTED,
        comment,
    )
