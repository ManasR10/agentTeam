from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, Protocol

from agents.orchestrator import enforce_run_limits
from agents.results import CodingResult, ReviewResult
from agents.state import PlanningResult
from tools.git_tools import get_changed_paths, get_current_head, get_diff_text
from tools.mutation_safety import MutationPolicy
from tools.results import CommandResult
from workflow.approvals import ensure_approval_matches_plan
from workflow.errors import StagePreconditionError, StaleRepositoryError
from workflow.events import EventType, record_event
from workflow.models import (
    CodingAttempt,
    ReviewAttempt,
    RunStatus,
    VerificationRecord,
    WorkflowErrorRecord,
    WorkflowRun,
)
from workflow.transitions import transition_run

if TYPE_CHECKING:
    from config import Settings
    from storage.base import RunStore

# Each stage follows the same discipline: SAVE the starting state before doing
# any work, then save the result after. If the process dies mid-stage, the
# database shows the stage it was IN (e.g. implementing), not the one it was
# about to enter — that's what makes recovery honest.
#
# The transition INTO a stage's active status is performed by the *previous*
# stage, so the persisted state always reads "about to do X". Planning and
# implementation are the exceptions: they start from a paused status (created /
# plan_approved) and transition themselves.


# Injected work. The deterministic parts (git, limits) are imported directly;
# only these LLM-/subprocess-facing callables are injected, so tests run offline.
Planner = Callable[[str], PlanningResult]
# The coder receives the whole run so the service can build a run-scoped tool
# executor (snapshots + audit) for it; feedback is the repair prompt or None.
Coder = Callable[["WorkflowRun", "str | None"], CodingResult]
Verifier = Callable[[], "tuple[CommandResult, ...]"]


class Reviewer(Protocol):
    def __call__(
        self,
        task: str,
        plan: PlanningResult,
        diff: str,
        command_results: tuple[CommandResult, ...],
        changed_files: tuple[str, ...],
        coder_summary: str,
    ) -> tuple[ReviewResult, int, int]:
        ...


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _require(run: WorkflowRun, expected: RunStatus, stage: str) -> None:
    if run.status is not expected:
        raise StagePreconditionError(
            f"{stage} stage requires status {expected.value}, "
            f"got {run.status.value}."
        )


def _require_in(
    run: WorkflowRun, expected: tuple[RunStatus, ...], stage: str
) -> None:
    if run.status not in expected:
        allowed = ", ".join(s.value for s in expected)
        raise StagePreconditionError(
            f"{stage} stage requires one of [{allowed}], got {run.status.value}."
        )


def _fail(
    run: WorkflowRun,
    store: RunStore,
    stage: str,
    *,
    error_type: str,
    message: str,
    retryable: bool = False,
) -> WorkflowRun:
    run.last_error = WorkflowErrorRecord(
        error_type=error_type,
        message=message,
        stage=stage,
        timestamp=_now(),
        retryable=retryable,
    )
    run.active_stage = None
    transition_run(run, RunStatus.FAILED, current_stage=stage)
    store.save_run(run)
    record_event(
        store,
        str(run.run_id),
        EventType.STAGE_FAILED,
        stage=stage,
        payload={"error_type": error_type, "message": message},
    )
    return run


def _started(run: WorkflowRun, store: RunStore, stage: str) -> None:
    record_event(store, str(run.run_id), EventType.STAGE_STARTED, stage=stage)


def _completed(run: WorkflowRun, store: RunStore, stage: str) -> None:
    record_event(store, str(run.run_id), EventType.STAGE_COMPLETED, stage=stage)


# --- Planning --------------------------------------------------------------


def run_planning_stage(
    run: WorkflowRun,
    *,
    store: RunStore,
    planner: Planner,
) -> WorkflowRun:
    _require(run, RunStatus.CREATED, "planning")
    transition_run(run, RunStatus.PLANNING, current_stage="planning")
    store.save_run(run)
    _started(run, store, "planning")

    try:
        plan = planner(run.task)
    except Exception as exc:  # noqa: BLE001 - failure is recorded, not raised
        return _fail(
            run, store, "planning",
            error_type=type(exc).__name__, message=str(exc),
        )

    run.plan = plan
    run.total_input_tokens += plan.input_tokens
    run.total_output_tokens += plan.output_tokens
    transition_run(
        run, RunStatus.AWAITING_PLAN_APPROVAL, current_stage="awaiting_plan_approval"
    )
    store.save_run(run)
    record_event(
        store, str(run.run_id), EventType.PLAN_CREATED, stage="planning",
        payload={
            "steps": len(plan.implementation_plan),
            "files_likely_to_change": list(plan.files_likely_to_change),
        },
    )
    record_event(store, str(run.run_id), EventType.APPROVAL_REQUESTED, stage="planning")
    return run


# --- Implementation --------------------------------------------------------


def _record_coding(
    run: WorkflowRun, store: RunStore, coding: CodingResult, settings: Settings,
    policy: MutationPolicy, stage: str,
) -> WorkflowRun | None:
    """Attach a coding attempt and enforce limits. Returns a failed run if the
    run-level mutation limits are exceeded, else None to continue."""
    attempt_number = len(run.coding_attempts) + 1
    run.coding_attempts.append(
        CodingAttempt(attempt_number, coding, _now(), _now())
    )
    run.changed_files = list(coding.actual_changed_files)
    run.total_input_tokens += coding.input_tokens
    run.total_output_tokens += coding.output_tokens

    reason = enforce_run_limits(policy, coding.actual_changed_files, settings)
    if reason is not None:
        return _fail(
            run, store, stage, error_type="RunLimitExceeded", message=reason,
        )
    # file.changed events are emitted per-write by the recording executor (with
    # hashes/sizes); we don't re-emit a thinner duplicate here.
    return None


def run_implementation_stage(
    run: WorkflowRun,
    *,
    store: RunStore,
    coder: Coder,
    settings: Settings,
    policy: MutationPolicy,
) -> WorkflowRun:
    _require(run, RunStatus.PLAN_APPROVED, "implementation")

    # The plan was approved for THIS repository state. If it drifted, refuse.
    try:
        ensure_approval_matches_plan(run)
        _ensure_repo_baseline(run, settings)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            run, store, "implementing",
            error_type=type(exc).__name__, message=str(exc),
        )

    transition_run(run, RunStatus.IMPLEMENTING, current_stage="implementing")
    # Marker: from here the coder may write before its attempt is persisted, so
    # a crash must be recovered (not re-run). Saved before the coder is invoked.
    run.active_stage = "implementing"
    store.save_run(run)
    _started(run, store, "implementing")

    try:
        coding = coder(run, None)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            run, store, "implementing",
            error_type=type(exc).__name__, message=str(exc),
        )

    failed = _record_coding(run, store, coding, settings, policy, "implementing")
    if failed is not None:
        return failed

    run.active_stage = None
    transition_run(run, RunStatus.VERIFYING, current_stage="verifying")
    store.save_run(run)
    _completed(run, store, "implementing")
    return run


def _ensure_repo_baseline(run: WorkflowRun, settings: Settings) -> None:
    current = get_current_head(settings)
    if current != run.starting_git_head:
        raise StaleRepositoryError(
            f"HEAD moved since planning ({run.starting_git_head} -> {current}); "
            "replan required."
        )
    if get_changed_paths(settings):
        raise StaleRepositoryError(
            "Worktree has uncommitted changes since planning; replan required."
        )


# --- Verification ----------------------------------------------------------


def run_verification_stage(
    run: WorkflowRun,
    *,
    store: RunStore,
    verifier: Verifier,
    max_attempts: int,
) -> WorkflowRun:
    _require(run, RunStatus.VERIFYING, "verification")
    _started(run, store, "verification")

    try:
        command_results = verifier()
    except Exception as exc:  # noqa: BLE001 - a crashing verifier still fails durably
        return _fail(
            run, store, "verification",
            error_type=type(exc).__name__, message=str(exc),
        )
    passed = all(
        (not r.timed_out and r.exit_code == 0) for r in command_results
    ) and bool(command_results)
    run.verification_runs.append(
        VerificationRecord(
            attempt_number=len(run.coding_attempts),
            command_results=tuple(command_results),
            passed=passed,
            started_at=_now(),
            completed_at=_now(),
        )
    )
    record_event(
        store, str(run.run_id), EventType.COMMAND_COMPLETED, stage="verification",
        payload={"passed": passed, "commands": [r.command_name for r in command_results]},
    )

    if passed:
        transition_run(run, RunStatus.REVIEWING, current_stage="reviewing")
        store.save_run(run)
        _completed(run, store, "verification")
        return run

    # Failed verification: repair if we still have an attempt left, else stop.
    if len(run.coding_attempts) < max_attempts:
        transition_run(run, RunStatus.REPAIRING, current_stage="repairing")
    else:
        transition_run(
            run, RunStatus.CHANGES_REQUESTED, current_stage="changes_requested"
        )
    store.save_run(run)
    _completed(run, store, "verification")
    return run


# --- Review ----------------------------------------------------------------


def run_review_stage(
    run: WorkflowRun,
    *,
    store: RunStore,
    reviewer: Reviewer,
    settings: Settings,
    max_attempts: int,
) -> WorkflowRun:
    _require(run, RunStatus.REVIEWING, "review")
    _started(run, store, "review")

    diff = get_diff_text(settings)
    latest = run.verification_runs[-1] if run.verification_runs else None
    command_results = latest.command_results if latest else ()
    coder_summary = run.coding_attempts[-1].result.summary if run.coding_attempts else ""

    try:
        review, in_tokens, out_tokens = reviewer(
            run.task,
            run.plan,
            diff,
            command_results,
            tuple(run.changed_files),
            coder_summary,
        )
    except Exception as exc:  # noqa: BLE001
        return _fail(
            run, store, "review",
            error_type=type(exc).__name__, message=str(exc),
        )

    run.total_input_tokens += in_tokens
    run.total_output_tokens += out_tokens
    run.review_attempts.append(
        ReviewAttempt(len(run.coding_attempts), review, _now(), _now())
    )
    run.review_cycle = len(run.review_attempts)
    record_event(
        store, str(run.run_id), EventType.REVIEW_COMPLETED, stage="review",
        payload={"verdict": review.verdict, "issues": len(review.issues)},
    )

    if review.verdict == "approved":
        transition_run(run, RunStatus.COMPLETED, current_stage="completed")
        store.save_run(run)
        record_event(store, str(run.run_id), EventType.RUN_COMPLETED, stage="review")
        return run

    if len(run.coding_attempts) < max_attempts:
        transition_run(run, RunStatus.REPAIRING, current_stage="repairing")
    else:
        transition_run(
            run, RunStatus.CHANGES_REQUESTED, current_stage="changes_requested"
        )
    store.save_run(run)
    _completed(run, store, "review")
    return run


# --- Repair ----------------------------------------------------------------


def _format_repair_feedback(run: WorkflowRun) -> str:
    """What to tell the coder on a repair attempt.

    Prefer the reviewer's issues; if we got here from a failed verification with
    no review yet, hand back the failing command output instead.
    """
    if run.review_attempts:
        review = run.review_attempts[-1].result
        lines = [f"Reviewer summary: {review.summary}"]
        for issue in review.issues:
            location = issue.path or "(general)"
            lines.append(
                f"- [{issue.severity}] {location}: {issue.description} "
                f"-> {issue.required_change}"
            )
        return "\n".join(lines)

    if run.verification_runs:
        latest = run.verification_runs[-1]
        lines = ["Verification failed. Fix the failures below."]
        for result in latest.command_results:
            tail = (result.stderr or result.stdout or "").strip()
            lines.append(f"[{result.command_name}] {tail[-1000:]}")
        return "\n".join(lines)

    return "The previous attempt did not pass verification; please fix it."


def recover_interrupted_write_stage(
    run: WorkflowRun,
    *,
    store: RunStore,
    settings: Settings,
    stage: str = "implementing",
) -> WorkflowRun:
    """Recover a run that crashed mid-write (implementation or repair).

    The coder may have written files before its attempt was persisted, so we do
    NOT re-run it (that could double-apply edits or fail on stale content).
    Instead we attribute the current diff to the run's recorded file.changed
    events: if everything is accounted for, we record the attempt from the audit
    trail and move on to verification; if anything is unattributed, we fail and
    leave it for manual resolution or rollback.
    """
    _require_in(run, (RunStatus.IMPLEMENTING, RunStatus.REPAIRING), "recover")
    attributed = {
        event.payload.get("path")
        for event in store.list_events(str(run.run_id))
        if event.event_type is EventType.FILE_CHANGED
    }
    current = set(get_changed_paths(settings))
    unattributed = current - attributed
    if unattributed:
        return _fail(
            run, store, stage,
            error_type="UnattributedChanges",
            message=(
                "Repository has changes not attributed to this run: "
                f"{sorted(unattributed)}. Resolve or roll back manually."
            ),
        )

    # Use what git ACTUALLY reports changed now, not the union of historical
    # events: a file touched then reverted to baseline still has an old event
    # but is no longer in the diff, so it must not be listed as changed.
    changed = sorted(current)
    if changed:
        run.changed_files = changed
    # Record the attempt from the audit trail so recovered runs keep a history
    # (and verification/review get a real attempt number, never 0).
    run.coding_attempts.append(
        CodingAttempt(
            attempt_number=len(run.coding_attempts) + 1,
            result=CodingResult(
                summary=f"Recovered from an interrupted {stage}.",
                reported_changed_files=(),
                tests_requested=(),
                known_issues=(),
                actual_changed_files=tuple(changed),
            ),
            started_at=_now(),
            completed_at=_now(),
        )
    )
    run.active_stage = None
    transition_run(run, RunStatus.VERIFYING, current_stage="verifying")
    store.save_run(run)
    record_event(
        store, str(run.run_id), EventType.STAGE_COMPLETED,
        stage=stage, payload={"recovered": True},
    )
    return run


def run_repair_stage(
    run: WorkflowRun,
    *,
    store: RunStore,
    coder: Coder,
    settings: Settings,
    policy: MutationPolicy,
) -> WorkflowRun:
    _require(run, RunStatus.REPAIRING, "repair")
    # Marker saved before the coder runs: a crash mid-repair must be recovered,
    # never re-run (which could double-apply edits or fail on stale content).
    run.active_stage = "repair"
    store.save_run(run)
    record_event(store, str(run.run_id), EventType.REPAIR_STARTED, stage="repair")

    feedback = _format_repair_feedback(run)
    try:
        coding = coder(run, feedback)
    except Exception as exc:  # noqa: BLE001
        return _fail(
            run, store, "repair",
            error_type=type(exc).__name__, message=str(exc),
        )

    failed = _record_coding(run, store, coding, settings, policy, "repair")
    if failed is not None:
        return failed

    run.active_stage = None
    transition_run(run, RunStatus.VERIFYING, current_stage="verifying")
    store.save_run(run)
    _completed(run, store, "repair")
    return run
