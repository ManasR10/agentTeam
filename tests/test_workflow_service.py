from __future__ import annotations

import pytest

from agents.results import CodingResult, ReviewResult
from agents.state import PlanningResult
from storage.sqlite_store import SqliteRunStore
from tools.results import CommandResult
from workflow.errors import RunNotResumableError
from workflow.models import RunStatus
from workflow.service import WorkflowService


def _plan(task: str = "add count()") -> PlanningResult:
    return PlanningResult(
        task=task,
        repo_summary="tiny todo app",
        relevant_files=[],
        implementation_plan=["add count method"],
        files_likely_to_change=["todo.py"],
        tests_to_add=[],
        risks=[],
        unknowns=[],
        raw_response="{}",
        input_tokens=10,
        output_tokens=5,
    )


def _coding(summary: str = "did it") -> CodingResult:
    return CodingResult(
        summary=summary,
        reported_changed_files=(),
        tests_requested=(),
        known_issues=(),
        actual_changed_files=("todo.py",),
        tool_calls=("replace_in_file(todo.py): ok",),
        input_tokens=20,
        output_tokens=8,
    )


def _passing_command() -> CommandResult:
    return CommandResult(
        command_name="pytest", argv=("pytest",), exit_code=0, stdout="ok",
        stderr="", timed_out=False, duration_seconds=0.1, output_truncated=False,
    )


def _failing_command() -> CommandResult:
    return CommandResult(
        command_name="pytest", argv=("pytest",), exit_code=1, stdout="",
        stderr="boom", timed_out=False, duration_seconds=0.1, output_truncated=False,
    )


def _approved():
    return ReviewResult(verdict="approved", summary="lgtm", issues=(), tests_assessment="pass")


def _changes():
    return ReviewResult(
        verdict="changes_requested", summary="fix", issues=(), tests_assessment="pass"
    )


class _Spy:
    """Records how many times the coder ran, and serves scripted reviews."""

    def __init__(self, reviews):
        self.coder_calls = 0
        self._reviews = list(reviews)

    def coder(self, run, feedback):
        self.coder_calls += 1
        return _coding(f"attempt {self.coder_calls}")

    def reviewer(self, task, plan, diff, command_results, changed_files, summary):
        return self._reviews.pop(0), 1, 1


def _service(repo_settings, *, spy, verifier=None, **kw):
    store = SqliteRunStore(repo_settings.devagent_database_path)
    return store, WorkflowService(
        store,
        settings=repo_settings,
        planner=lambda task: _plan(task),
        coder=spy.coder,
        verifier=verifier or (lambda: (_passing_command(),)),
        reviewer=spy.reviewer,
        **kw,
    )


def test_start_plans_only_and_waits(repo_settings):
    spy = _Spy([_approved()])
    store, svc = _service(repo_settings, spy=spy)
    try:
        run = svc.start_run("add count()")
        assert run.status is RunStatus.AWAITING_PLAN_APPROVAL
        assert run.plan is not None
        assert spy.coder_calls == 0  # coder must not run before approval
        assert run.coding_attempts == []
    finally:
        store.close()


def test_approve_then_resume_completes(repo_settings):
    spy = _Spy([_approved()])
    store, svc = _service(repo_settings, spy=spy)
    try:
        run = svc.start_run("add count()")
        svc.approve_run(str(run.run_id))
        done = svc.resume_run(str(run.run_id))
        assert done.status is RunStatus.COMPLETED
        assert spy.coder_calls == 1
        assert len(done.coding_attempts) == 1
        assert len(done.verification_runs) == 1
        assert len(done.review_attempts) == 1
    finally:
        store.close()


def test_resume_without_approval_waits(repo_settings):
    spy = _Spy([_approved()])
    store, svc = _service(repo_settings, spy=spy)
    try:
        run = svc.start_run("add count()")
        resumed = svc.resume_run(str(run.run_id))
        assert resumed.status is RunStatus.AWAITING_PLAN_APPROVAL
        assert spy.coder_calls == 0
    finally:
        store.close()


def test_rejected_run_cannot_resume(repo_settings):
    spy = _Spy([_approved()])
    store, svc = _service(repo_settings, spy=spy)
    try:
        run = svc.start_run("add count()")
        svc.reject_run(str(run.run_id), comment="no")
        with pytest.raises(RunNotResumableError):
            svc.resume_run(str(run.run_id))
    finally:
        store.close()


def test_completed_run_cannot_resume(repo_settings):
    spy = _Spy([_approved()])
    store, svc = _service(repo_settings, spy=spy)
    try:
        run = svc.start_run("add count()")
        svc.approve_run(str(run.run_id))
        svc.resume_run(str(run.run_id))
        with pytest.raises(RunNotResumableError):
            svc.resume_run(str(run.run_id))
    finally:
        store.close()


def test_review_requesting_changes_triggers_one_repair(repo_settings):
    # First review asks for changes, second approves -> exactly two coder runs.
    spy = _Spy([_changes(), _approved()])
    store, svc = _service(repo_settings, spy=spy)
    try:
        run = svc.start_run("add count()")
        svc.approve_run(str(run.run_id))
        done = svc.resume_run(str(run.run_id))
        assert done.status is RunStatus.COMPLETED
        assert spy.coder_calls == 2
        assert len(done.coding_attempts) == 2  # history preserved
    finally:
        store.close()


def test_failing_verification_triggers_repair(repo_settings):
    # Verification fails once then passes; reviewer approves the repaired code.
    results = iter([(_failing_command(),), (_passing_command(),)])
    spy = _Spy([_approved()])
    store, svc = _service(repo_settings, spy=spy, verifier=lambda: next(results))
    try:
        run = svc.start_run("add count()")
        svc.approve_run(str(run.run_id))
        done = svc.resume_run(str(run.run_id))
        assert done.status is RunStatus.COMPLETED
        assert spy.coder_calls == 2
    finally:
        store.close()


def test_exhausted_repairs_end_in_changes_requested(repo_settings):
    spy = _Spy([_changes(), _changes()])
    store, svc = _service(repo_settings, spy=spy, max_attempts=2)
    try:
        run = svc.start_run("add count()")
        svc.approve_run(str(run.run_id))
        done = svc.resume_run(str(run.run_id))
        assert done.status is RunStatus.CHANGES_REQUESTED
        assert spy.coder_calls == 2  # initial + one repair, then give up
    finally:
        store.close()


def test_planner_failure_is_persisted(repo_settings):
    store = SqliteRunStore(repo_settings.devagent_database_path)
    svc = WorkflowService(
        store,
        settings=repo_settings,
        planner=lambda task: (_ for _ in ()).throw(RuntimeError("planner boom")),
        coder=lambda run, feedback: _coding(),
        verifier=lambda: (_passing_command(),),
        reviewer=lambda *a: (_approved(), 0, 0),
    )
    try:
        run = svc.start_run("add count()")
        assert run.status is RunStatus.FAILED
        assert run.last_error is not None
        assert run.last_error.stage == "planning"
        # And it survived to the database.
        assert store.load_run(str(run.run_id)).status is RunStatus.FAILED
    finally:
        store.close()
