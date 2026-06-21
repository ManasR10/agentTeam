from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

if TYPE_CHECKING:
    # Type-only imports: a WorkflowRun holds these but never constructs them, so
    # keeping them out of the runtime import graph avoids a cycle with agents/.
    from agents.results import CodingResult, ReviewResult
    from agents.state import PlanningResult
    from tools.results import CommandResult

# Bumped only when the persisted shape changes in a way old code can't read.
# Loading anything outside this set is an error, never a silent guess.
SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = frozenset({1})


def _utcnow() -> datetime:
    """Timezone-aware UTC now. All workflow timestamps are UTC so they survive
    serialization and compare correctly across machines."""
    return datetime.now(timezone.utc)


class RunStatus(StrEnum):
    """Where a run is in its lifecycle.

    Every value names a concrete stage, not a vague 'running'/'done'. The split
    between AWAITING_PLAN_APPROVAL, PLAN_APPROVED and IMPLEMENTING is what lets
    approval and execution be separate, persisted actions: a human approves a
    plan (-> PLAN_APPROVED), and only a later resume starts the coder
    (-> IMPLEMENTING).
    """

    CREATED = "created"
    PLANNING = "planning"
    AWAITING_PLAN_APPROVAL = "awaiting_plan_approval"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"
    IMPLEMENTING = "implementing"
    VERIFYING = "verifying"
    REVIEWING = "reviewing"
    REPAIRING = "repairing"
    COMPLETED = "completed"
    CHANGES_REQUESTED = "changes_requested"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


class ApprovalDecision(StrEnum):
    """A human's decision on a generated plan."""

    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class WorkflowErrorRecord:
    """A structured record of why a stage failed.

    `retryable` lets resume distinguish a transient failure (API timeout) from a
    permanent one (parser rejected the model's JSON), without re-parsing the
    message string.
    """

    error_type: str
    message: str
    stage: str
    timestamp: datetime
    retryable: bool


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """A persisted approval/rejection, bound to the exact plan it covers.

    `plan_hash` is the deterministic hash of the plan as approved. Before the
    coder runs, the stored plan is re-hashed and compared, so an approval can
    never silently carry over to a plan that changed afterwards.
    """

    approval_id: UUID
    run_id: UUID
    decision: ApprovalDecision
    plan_hash: str
    comment: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class CodingAttempt:
    """One coder (or repair) attempt, preserved as history.

    Attempt 1 is the initial implementation; later numbers are repair attempts.
    These are appended, never overwritten, so the full sequence stays auditable.
    """

    attempt_number: int
    result: CodingResult
    started_at: datetime
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """The result of one orchestrator-run verification pass (pytest, compile).

    `passed` is the orchestrator's own judgement from the command results — the
    coder is never trusted to report whether tests passed.
    """

    attempt_number: int
    command_results: tuple[CommandResult, ...]
    passed: bool
    started_at: datetime
    completed_at: datetime


@dataclass(frozen=True, slots=True)
class ReviewAttempt:
    """One reviewer verdict, preserved as history alongside the coding attempt
    it judged."""

    attempt_number: int
    result: ReviewResult
    started_at: datetime
    completed_at: datetime


@dataclass(slots=True)
class WorkflowRun:
    """The durable record of a single task run.

    Mutable on purpose (like the Phase 3 AgentState it evolves from): stages
    advance the status and append attempts in place, and the service persists
    the run after each change. History fields are append-only by convention —
    nothing here overwrites a prior attempt or review.

    The baseline fields (`starting_git_head`, `starting_worktree_clean`) are
    captured before planning so a later resume can prove the repository hasn't
    drifted out from under an approved plan.
    """

    task: str
    workspace_root: str
    starting_git_head: str
    starting_worktree_clean: bool

    run_id: UUID = field(default_factory=uuid4)
    status: RunStatus = RunStatus.CREATED
    current_stage: str | None = None
    # Set right before a write stage (implementation/repair) invokes the coder,
    # cleared once its attempt is recorded. If a run is reloaded with this still
    # set, the coder may have written before a crash, so resume must reconcile
    # the diff rather than re-run the coder (which could double-apply edits).
    active_stage: str | None = None
    schema_version: int = SCHEMA_VERSION
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    plan: PlanningResult | None = None
    approval: ApprovalRecord | None = None
    coding_attempts: list[CodingAttempt] = field(default_factory=list)
    verification_runs: list[VerificationRecord] = field(default_factory=list)
    review_attempts: list[ReviewAttempt] = field(default_factory=list)

    # Authoritative paths git reports as changed (not the model's own claims;
    # those live inside each attempt's CodingResult). This is the list rollback
    # and run-limit enforcement act on.
    changed_files: list[str] = field(default_factory=list)
    review_cycle: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    last_error: WorkflowErrorRecord | None = None

    def __post_init__(self) -> None:
        # Field-level invariants that hold for any run at any stage. Cross-field
        # stage consistency (e.g. "implementing implies an approval exists") is
        # checked on load in the serialization layer, since a run part-way
        # through construction need not satisfy it yet.
        if not self.task.strip():
            raise ValueError("WorkflowRun.task cannot be empty")
        if not self.workspace_root.strip():
            raise ValueError("WorkflowRun.workspace_root cannot be empty")
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported workflow schema version: {self.schema_version}"
            )
        if self.review_cycle < 0:
            raise ValueError("WorkflowRun.review_cycle cannot be negative")
        if self.total_input_tokens < 0 or self.total_output_tokens < 0:
            raise ValueError("WorkflowRun token counts cannot be negative")
