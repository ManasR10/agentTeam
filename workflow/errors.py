from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from workflow.models import RunStatus


class WorkflowError(RuntimeError):
    """Base class for workflow-engine errors.

    Storage, approval, and resume each define their own subclasses near the code
    that raises them; this is the common root so a caller can catch the whole
    family when it needs to.
    """


class InvalidRunTransitionError(WorkflowError):
    """Raised when code tries to move a run between two incompatible statuses.

    Carrying the from/to statuses (not just a message) lets callers branch on
    what was attempted without parsing the string.
    """

    def __init__(self, from_status: RunStatus, to_status: RunStatus) -> None:
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(
            f"Cannot transition run from {from_status.value} "
            f"to {to_status.value}."
        )


class StagePreconditionError(WorkflowError):
    """Raised when a stage is invoked from a status it does not handle.

    This is a programming/dispatch error (resume should route by status), not a
    normal failure, so it is not turned into a FAILED run.
    """


class StaleRepositoryError(WorkflowError):
    """Raised when the repository moved between planning and implementation.

    Applying an approved plan to a changed repository risks a wrong edit, so the
    run is failed and the user is asked to replan rather than proceed blindly.
    """


class RunNotResumableError(WorkflowError):
    """Raised when resume is asked to advance a run that is finished or waiting
    on a human."""
