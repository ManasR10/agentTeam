from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from agents.results import CodingResult, ReviewResult
from agents.state import PlanningResult
from tools.results import CommandResult
from workflow.approvals import calculate_plan_hash
from workflow.events import EventType
from workflow.errors import RunNotResumableError
from workflow.models import (
    ApprovalDecision,
    ApprovalRecord,
    CodingAttempt,
    ReviewAttempt,
    RunStatus,
    WorkflowRun,
)
from workflow.service import WorkflowService
from workflow.stages import recover_interrupted_write_stage

_NOW = datetime(2026, 6, 21, tzinfo=timezone.utc)


def _plan() -> PlanningResult:
    return PlanningResult(
        task="add count()",
        repo_summary="todo",
        relevant_files=[],
        implementation_plan=["add count"],
        files_likely_to_change=["todo.py"],
        tests_to_add=[],
        risks=[],
        unknowns=[],
        raw_response="{}",
    )


def _coding() -> CodingResult:
    return CodingResult(
        summary="did it",
        reported_changed_files=(),
        tests_requested=(),
        known_issues=(),
        actual_changed_files=("todo.py",),
    )


def _review(verdict="approved") -> ReviewResult:
    return ReviewResult(verdict=verdict, summary="s", issues=(), tests_assessment="p")


def _passing():
    return CommandResult(
        command_name="pytest", argv=("pytest",), exit_code=0, stdout="ok",
        stderr="", timed_out=False, duration_seconds=0.1, output_truncated=False,
    )


def _run(repo_settings, *, status, head=None, coding=0, reviews=()):
    plan = _plan()
    run = WorkflowRun(
        task="add count()",
        workspace_root=str(repo_settings.tool_workspace_root),
        starting_git_head=head or "",
        starting_worktree_clean=True,
        status=status,
        plan=plan,
        changed_files=["todo.py"] if coding else [],
    )
    if status not in (RunStatus.CREATED,):
        run.approval = ApprovalRecord(
            approval_id=uuid4(), run_id=run.run_id,
            decision=ApprovalDecision.APPROVED,
            plan_hash=calculate_plan_hash(plan), comment=None, created_at=_NOW,
        )
    for i in range(coding):
        run.coding_attempts.append(CodingAttempt(i + 1, _coding(), _NOW, _NOW))
    for i, verdict in enumerate(reviews):
        run.review_attempts.append(ReviewAttempt(i + 1, _review(verdict), _NOW, _NOW))
    return run


class _Spy:
    def __init__(self, reviews):
        self.coder_calls = 0
        self._reviews = list(reviews)

    def coder(self, run, feedback):
        self.coder_calls += 1
        return _coding()

    def reviewer(self, *a):
        return self._reviews.pop(0), 0, 0


def _service(store, repo_settings, spy, **kw):
    return WorkflowService(
        store,
        settings=repo_settings,
        planner=lambda task: _plan(),
        coder=spy.coder,
        verifier=lambda: (_passing(),),
        reviewer=spy.reviewer,
        **kw,
    )


@pytest.fixture()
def store(repo_settings):
    from storage.sqlite_store import SqliteRunStore

    s = SqliteRunStore(repo_settings.devagent_database_path)
    yield s
    s.close()


def test_resume_created_runs_planning(store, repo_settings):
    run = _run(repo_settings, status=RunStatus.CREATED)
    store.create_run(run)
    spy = _Spy([])
    resumed = _service(store, repo_settings, spy).resume_run(str(run.run_id))
    assert resumed.status is RunStatus.AWAITING_PLAN_APPROVAL
    assert spy.coder_calls == 0


def test_resume_awaiting_approval_does_nothing(store, repo_settings):
    run = _run(repo_settings, status=RunStatus.AWAITING_PLAN_APPROVAL)
    store.create_run(run)
    spy = _Spy([])
    resumed = _service(store, repo_settings, spy).resume_run(str(run.run_id))
    assert resumed.status is RunStatus.AWAITING_PLAN_APPROVAL
    assert spy.coder_calls == 0


def test_resume_verifying_does_not_add_coding_attempt(store, repo_settings):
    run = _run(repo_settings, status=RunStatus.VERIFYING, coding=1)
    store.create_run(run)
    spy = _Spy([_review("approved")])
    done = _service(store, repo_settings, spy).resume_run(str(run.run_id))
    assert done.status is RunStatus.COMPLETED
    assert spy.coder_calls == 0  # resuming verification must not re-run the coder
    assert len(done.coding_attempts) == 1


def test_repair_cycle_survives_restart(repo_settings):
    from storage.sqlite_store import SqliteRunStore

    run = _run(
        repo_settings, status=RunStatus.REPAIRING, coding=1, reviews=("changes_requested",)
    )
    first = SqliteRunStore(repo_settings.devagent_database_path)
    first.create_run(run)
    first.close()

    second = SqliteRunStore(repo_settings.devagent_database_path)
    try:
        spy = _Spy([_review("approved")])
        done = _service(second, repo_settings, spy).resume_run(str(run.run_id))
        assert done.status is RunStatus.COMPLETED
        # Attempt numbering continued from the restored state, not from 1.
        assert spy.coder_calls == 1
        assert [a.attempt_number for a in done.coding_attempts] == [1, 2]
    finally:
        second.close()


def test_resume_completed_run_refuses(store, repo_settings):
    run = _run(repo_settings, status=RunStatus.COMPLETED, coding=1, reviews=("approved",))
    store.create_run(run)
    spy = _Spy([])
    with pytest.raises(RunNotResumableError):
        _service(store, repo_settings, spy).resume_run(str(run.run_id))


def test_interrupted_implementation_with_known_changes_proceeds(store, repo_settings):
    head = _head(repo_settings)
    run = _run(repo_settings, status=RunStatus.IMPLEMENTING, head=head)
    store.create_run(run)
    # Record that this run touched todo.py, then make that exact change.
    store.append_event(
        str(run.run_id), EventType.FILE_CHANGED, payload={"path": "todo.py"}
    )
    (repo_settings.tool_workspace_root / "todo.py").write_text(
        "class TodoList:\n    pass\n", encoding="utf-8"
    )
    spy = _Spy([_review("approved")])
    done = _service(store, repo_settings, spy).resume_run(str(run.run_id))
    assert done.status is RunStatus.COMPLETED
    assert spy.coder_calls == 0  # recovery never re-runs the coder
    # Recovery reconstructs a coding attempt so history isn't empty and the
    # attempt number is never 0.
    assert len(done.coding_attempts) == 1
    assert done.coding_attempts[0].attempt_number == 1
    assert "todo.py" in done.coding_attempts[0].result.actual_changed_files
    assert done.verification_runs[-1].attempt_number == 1


def test_verifier_crash_is_persisted_as_failed(store, repo_settings):
    run = _run(repo_settings, status=RunStatus.VERIFYING, coding=1)
    store.create_run(run)
    spy = _Spy([])

    def boom():
        raise RuntimeError("verify boom")

    svc = _service(store, repo_settings, spy)
    svc._verifier = boom
    done = svc.resume_run(str(run.run_id))
    assert done.status is RunStatus.FAILED
    assert done.last_error is not None
    assert done.last_error.stage == "verification"
    # And it's durable, not just in memory.
    assert store.load_run(str(run.run_id)).status is RunStatus.FAILED


def test_interrupted_implementation_with_unattributed_changes_fails(store, repo_settings):
    head = _head(repo_settings)
    run = _run(repo_settings, status=RunStatus.IMPLEMENTING, head=head)
    store.create_run(run)
    # A change the run never recorded — refuse and require manual resolution.
    (repo_settings.tool_workspace_root / "rogue.py").write_text("x = 1\n", encoding="utf-8")
    recovered = recover_interrupted_write_stage(
        run, store=store, settings=repo_settings, stage="implementing"
    )
    assert recovered.status is RunStatus.FAILED
    assert recovered.last_error.error_type == "UnattributedChanges"


def test_recovery_uses_current_diff_not_reverted_files(store, repo_settings):
    # ghost.py has a file.changed event but was reverted (never on disk), so it
    # is not in the git diff and must not be reported as changed.
    head = _head(repo_settings)
    run = _run(repo_settings, status=RunStatus.IMPLEMENTING, head=head)
    store.create_run(run)
    store.append_event(str(run.run_id), EventType.FILE_CHANGED, payload={"path": "todo.py"})
    store.append_event(str(run.run_id), EventType.FILE_CHANGED, payload={"path": "ghost.py"})
    (repo_settings.tool_workspace_root / "todo.py").write_text(
        "class TodoList:\n    pass\n", encoding="utf-8"
    )
    spy = _Spy([_review("approved")])
    done = _service(store, repo_settings, spy).resume_run(str(run.run_id))
    assert done.status is RunStatus.COMPLETED
    assert done.changed_files == ["todo.py"]
    assert done.coding_attempts[0].result.actual_changed_files == ("todo.py",)


def test_repairing_without_marker_runs_the_repair_coder(store, repo_settings):
    # No active_stage marker => the repair coder hasn't started; resume runs it.
    run = _run(
        repo_settings, status=RunStatus.REPAIRING, coding=1, reviews=("changes_requested",)
    )
    store.create_run(run)
    spy = _Spy([_review("approved")])
    done = _service(store, repo_settings, spy).resume_run(str(run.run_id))
    assert done.status is RunStatus.COMPLETED
    assert spy.coder_calls == 1  # the repair coder did run
    assert len(done.coding_attempts) == 2


def test_interrupted_repair_does_not_rerun_coder(store, repo_settings):
    # Marker set => the repair coder may have written before the crash; resume
    # must reconcile from the audit trail, not re-run the coder.
    run = _run(
        repo_settings, status=RunStatus.REPAIRING, coding=1, reviews=("changes_requested",)
    )
    run.active_stage = "repair"
    store.create_run(run)
    store.append_event(
        str(run.run_id), EventType.FILE_CHANGED, payload={"path": "todo.py"}
    )
    (repo_settings.tool_workspace_root / "todo.py").write_text(
        "class TodoList:\n    pass\n", encoding="utf-8"
    )
    spy = _Spy([_review("approved")])
    done = _service(store, repo_settings, spy).resume_run(str(run.run_id))
    assert done.status is RunStatus.COMPLETED
    assert spy.coder_calls == 0  # repair coder NOT re-run
    assert len(done.coding_attempts) == 2  # original + recovered
    assert "repair" in done.coding_attempts[-1].result.summary
    assert done.active_stage is None


def test_interrupted_repair_with_unattributed_change_fails(store, repo_settings):
    run = _run(
        repo_settings, status=RunStatus.REPAIRING, coding=1, reviews=("changes_requested",)
    )
    run.active_stage = "repair"
    store.create_run(run)
    (repo_settings.tool_workspace_root / "rogue.py").write_text("x = 1\n", encoding="utf-8")
    spy = _Spy([])
    done = _service(store, repo_settings, spy).resume_run(str(run.run_id))
    assert done.status is RunStatus.FAILED
    assert done.last_error.error_type == "UnattributedChanges"
    assert spy.coder_calls == 0


def _head(repo_settings) -> str:
    from tools.git_tools import get_current_head

    return get_current_head(repo_settings) or ""
