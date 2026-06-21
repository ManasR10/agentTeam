from __future__ import annotations

from datetime import datetime, timezone

from workflow.errors import InvalidRunTransitionError
from workflow.models import RunStatus, WorkflowRun

# The only legal status moves. Every RunStatus is a key here, so an unknown
# state can't slip through with a permissive default. Two design points:
#
#   - Approval and execution are separate: AWAITING_PLAN_APPROVAL goes to
#     PLAN_APPROVED (a human decision), and only PLAN_APPROVED goes to
#     IMPLEMENTING (a later resume). You can never jump approval->coding in one
#     step, which is what keeps "no mutation before approval" enforceable.
#   - ROLLED_BACK is reachable from the finished states (COMPLETED, FAILED,
#     CHANGES_REQUESTED). Rollback is an out-of-band recovery action, not part of
#     the forward flow — see TERMINAL_STATUSES below.
ALLOWED_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.CREATED: frozenset(
        {RunStatus.PLANNING, RunStatus.CANCELLED, RunStatus.FAILED}
    ),
    RunStatus.PLANNING: frozenset(
        {RunStatus.AWAITING_PLAN_APPROVAL, RunStatus.FAILED}
    ),
    RunStatus.AWAITING_PLAN_APPROVAL: frozenset(
        {
            RunStatus.PLAN_APPROVED,
            RunStatus.PLAN_REJECTED,
            RunStatus.CANCELLED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.PLAN_APPROVED: frozenset(
        {RunStatus.IMPLEMENTING, RunStatus.CANCELLED, RunStatus.FAILED}
    ),
    RunStatus.IMPLEMENTING: frozenset(
        {RunStatus.VERIFYING, RunStatus.FAILED}
    ),
    RunStatus.VERIFYING: frozenset(
        {
            RunStatus.REVIEWING,
            RunStatus.REPAIRING,
            RunStatus.CHANGES_REQUESTED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.REVIEWING: frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.REPAIRING,
            RunStatus.CHANGES_REQUESTED,
            RunStatus.FAILED,
        }
    ),
    RunStatus.REPAIRING: frozenset(
        {RunStatus.VERIFYING, RunStatus.FAILED}
    ),
    RunStatus.CHANGES_REQUESTED: frozenset({RunStatus.ROLLED_BACK}),
    RunStatus.FAILED: frozenset({RunStatus.ROLLED_BACK}),
    RunStatus.COMPLETED: frozenset({RunStatus.ROLLED_BACK}),
    RunStatus.PLAN_REJECTED: frozenset(),
    RunStatus.CANCELLED: frozenset(),
    RunStatus.ROLLED_BACK: frozenset(),
}

# A run in a terminal status has reached a decided outcome: resume will not
# advance it. This is about the forward workflow, not rollback — a COMPLETED run
# is terminal yet can still be rolled back via the recovery edge above.
TERMINAL_STATUSES: frozenset[RunStatus] = frozenset(
    {
        RunStatus.COMPLETED,
        RunStatus.PLAN_REJECTED,
        RunStatus.CANCELLED,
        RunStatus.ROLLED_BACK,
    }
)


def is_terminal(status: RunStatus) -> bool:
    """True if a run in this status is finished and will not be resumed."""
    return status in TERMINAL_STATUSES


def transition_run(
    run: WorkflowRun,
    new_status: RunStatus,
    *,
    current_stage: str | None = None,
) -> WorkflowRun:
    """Move `run` to `new_status`, refusing any move not in the table.

    This is the one place a run's status changes — orchestration code must never
    assign `run.status` directly. The run is mutated in place and returned for
    convenience. `current_stage`, when given, is updated alongside the status;
    leaving it None keeps the existing stage.
    """
    allowed = ALLOWED_TRANSITIONS.get(run.status, frozenset())
    if new_status not in allowed:
        raise InvalidRunTransitionError(run.status, new_status)

    run.status = new_status
    if current_stage is not None:
        run.current_stage = current_stage
    run.updated_at = datetime.now(timezone.utc)
    return run
