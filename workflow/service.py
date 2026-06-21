from __future__ import annotations

from typing import TYPE_CHECKING

from agents.coder import implement_repo_task
from agents.orchestrator import ensure_ready_worktree, run_verification
from agents.planner import plan_repo_task
from agents.reviewer import run_reviewer
from config import Settings, get_settings
from tools.git_tools import get_current_head
from tools.mutation_safety import build_mutation_policy
from tools.registry import CODER_TOOL_NAMES, make_tool_executor
from workflow.approvals import approve_plan, reject_plan
from workflow.errors import RunNotResumableError
from workflow.events import EventType, record_event
from workflow.models import RunStatus, WorkflowRun
from workflow.rollback import SnapshotStore, make_recording_executor
from workflow.stages import (
    Coder,
    Planner,
    Reviewer,
    Verifier,
    recover_interrupted_write_stage,
    run_implementation_stage,
    run_planning_stage,
    run_repair_stage,
    run_review_stage,
    run_verification_stage,
)

if TYPE_CHECKING:
    from storage.base import RunStore

# Initial implementation + up to this many total coding attempts. 2 means one
# repair attempt, matching the Phase 3 orchestrator's effective behaviour.
DEFAULT_MAX_ATTEMPTS = 2

# Statuses that resume drives forward without further human input.
_DRIVABLE = frozenset(
    {
        RunStatus.PLAN_APPROVED,
        RunStatus.VERIFYING,
        RunStatus.REVIEWING,
        RunStatus.REPAIRING,
    }
)


class WorkflowService:
    """The application API: start, approve/reject, and resume durable runs.

    The CLI talks to this, not to the agents directly. The LLM-/subprocess-facing
    work is injected (planner/coder/verifier/reviewer) so the whole service can
    be driven offline in tests with deterministic fakes; the defaults wire up the
    real Phase 3 agents.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        settings: Settings | None = None,
        planner: Planner | None = None,
        coder: Coder | None = None,
        verifier: Verifier | None = None,
        reviewer: Reviewer | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self._settings = self.settings
        self._policy = build_mutation_policy(self._settings)
        self._max_attempts = max_attempts
        self._planner = planner or (lambda task: plan_repo_task(task))
        self._coder = coder or self._default_coder
        self._verifier = verifier or (lambda: run_verification(self._settings))
        self._reviewer = reviewer or self._default_reviewer

    def _default_coder(self, run: WorkflowRun, feedback: str | None) -> object:
        # Wrap the coder's tools so every write is snapshotted (for rollback) and
        # audited, scoped to this run.
        snapshot_store = SnapshotStore(
            self._settings.devagent_data_dir, str(run.run_id)
        )
        executor = make_recording_executor(
            make_tool_executor(CODER_TOOL_NAMES),
            run_id=str(run.run_id),
            store=self.store,
            snapshot_store=snapshot_store,
            workspace_root=self._settings.tool_workspace_root,
            stage="repair" if feedback is not None else "implementing",
        )
        return implement_repo_task(
            run.task, run.plan, review_feedback=feedback, tool_executor=executor
        )

    def _default_reviewer(
        self, task, plan, diff, command_results, changed_files, coder_summary
    ):
        review, run_result = run_reviewer(
            task,
            plan,
            diff,
            command_results,
            changed_files=changed_files,
            coder_summary=coder_summary,
        )
        return review, run_result.input_tokens, run_result.output_tokens

    def start_run(self, task: str) -> WorkflowRun:
        """Create a run, capture the git baseline, and produce a plan.

        Stops at AWAITING_PLAN_APPROVAL — no coder runs. Refuses unless the
        workspace is a clean git root (raises NotAGitRepoError/DirtyWorktreeError).
        """
        ensure_ready_worktree(self._settings)
        run = WorkflowRun(
            task=task,
            workspace_root=str(self._settings.tool_workspace_root),
            starting_git_head=get_current_head(self._settings) or "",
            starting_worktree_clean=True,
        )
        self.store.create_run(run)
        record_event(
            self.store, str(run.run_id), EventType.RUN_CREATED,
            payload={"task": task},
        )
        return run_planning_stage(run, store=self.store, planner=self._planner)

    def approve_run(self, run_id: str, *, comment: str | None = None) -> WorkflowRun:
        """Approve the plan (-> PLAN_APPROVED). Does not start coding."""
        return approve_plan(self.store, run_id, comment=comment)

    def reject_run(self, run_id: str, *, comment: str | None = None) -> WorkflowRun:
        """Reject the plan (-> PLAN_REJECTED, terminal)."""
        return reject_plan(self.store, run_id, comment=comment)

    def resume_run(self, run_id: str) -> WorkflowRun:
        """Advance a run to the next human gate or terminal state."""
        run = self.store.load_run(run_id)
        if run.status is RunStatus.CREATED:
            return run_planning_stage(run, store=self.store, planner=self._planner)
        if run.status is RunStatus.AWAITING_PLAN_APPROVAL:
            return run  # waiting on a human decision
        if run.status is RunStatus.IMPLEMENTING:
            # IMPLEMENTING is only ever seen after a crash mid-stage.
            run = recover_interrupted_write_stage(
                run, store=self.store, settings=self._settings, stage="implementing"
            )
            return self._drive(run) if run.status in _DRIVABLE else run
        if run.status is RunStatus.REPAIRING and run.active_stage == "repair":
            # Crashed after the repair coder may have written: reconcile, don't
            # re-run. A REPAIRING run without the marker just hasn't started its
            # coder, so it falls through to _drive (which runs the repair stage).
            run = recover_interrupted_write_stage(
                run, store=self.store, settings=self._settings, stage="repair"
            )
            return self._drive(run) if run.status in _DRIVABLE else run
        if run.status in _DRIVABLE:
            return self._drive(run)
        raise RunNotResumableError(
            f"Run in status {run.status.value} cannot be resumed."
        )

    def _drive(self, run: WorkflowRun) -> WorkflowRun:
        """Run stages back-to-back until a terminal status or human gate."""
        while run.status in _DRIVABLE:
            if run.status is RunStatus.PLAN_APPROVED:
                run = run_implementation_stage(
                    run, store=self.store, coder=self._coder,
                    settings=self._settings, policy=self._policy,
                )
            elif run.status is RunStatus.VERIFYING:
                run = run_verification_stage(
                    run, store=self.store, verifier=self._verifier,
                    max_attempts=self._max_attempts,
                )
            elif run.status is RunStatus.REVIEWING:
                run = run_review_stage(
                    run, store=self.store, reviewer=self._reviewer,
                    settings=self._settings, max_attempts=self._max_attempts,
                )
            elif run.status is RunStatus.REPAIRING:
                run = run_repair_stage(
                    run, store=self.store, coder=self._coder,
                    settings=self._settings, policy=self._policy,
                )
        return run
