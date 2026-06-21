from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from workflow.events import EventType, WorkflowEvent
    from workflow.models import ApprovalRecord, RunStatus, WorkflowRun


class StorageError(RuntimeError):
    """Raised when the store cannot complete an operation it should have."""


class RunNotFoundError(LookupError):
    """Raised when a run id is not present in the store."""


class RunAlreadyExistsError(RuntimeError):
    """Raised when creating a run whose id already exists."""


@dataclass(frozen=True, slots=True)
class WorkflowRunSummary:
    """A lightweight row for listing runs without loading full state.

    Read straight from the denormalised columns on `workflow_runs`, so `list`
    stays cheap even when each run's `state_json` is large.
    """

    run_id: str
    task: str
    status: RunStatus
    current_stage: str | None
    created_at: datetime
    updated_at: datetime


class RunStore(Protocol):
    """What the workflow service needs from durable storage.

    Kept narrow on purpose: the service depends on this Protocol, not on SQLite,
    so a different backend (or an in-memory fake in tests) can be dropped in.
    """

    def create_run(self, run: WorkflowRun) -> None:
        """Insert a brand-new run. Raises RunAlreadyExistsError on a clash."""
        ...

    def save_run(self, run: WorkflowRun) -> None:
        """Persist changes to an existing run. Raises RunNotFoundError if it is
        not already stored."""
        ...

    def load_run(self, run_id: str) -> WorkflowRun:
        """Load and validate a run. Raises RunNotFoundError if absent."""
        ...

    def list_runs(
        self,
        *,
        limit: int = 20,
        status: RunStatus | None = None,
    ) -> list[WorkflowRunSummary]:
        """Most-recent-first summaries, optionally filtered by status."""
        ...

    def append_event(
        self,
        run_id: str | UUID,
        event_type: EventType,
        *,
        stage: str | None = None,
        agent_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        """Append one event, assigning the next per-run sequence number."""
        ...

    def list_events(self, run_id: str | UUID) -> list[WorkflowEvent]:
        """All events for a run, in sequence order."""
        ...

    def record_approval(self, approval: ApprovalRecord) -> None:
        """Persist an approval decision to the queryable approvals table."""
        ...

    def commit_approval(
        self,
        run: WorkflowRun,
        approval: ApprovalRecord,
        *,
        event_type: EventType,
        event_payload: dict[str, Any],
    ) -> None:
        """Save the run, approval row, and audit event in one transaction."""
        ...
