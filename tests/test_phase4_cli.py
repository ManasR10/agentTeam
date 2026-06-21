from __future__ import annotations

import pytest

import devagent
from agents.results import CodingResult, ReviewResult
from agents.state import PlanningResult
from storage.sqlite_store import SqliteRunStore
from tools.results import CommandResult
from workflow.service import WorkflowService


def _plan() -> PlanningResult:
    return PlanningResult(
        task="add count()", repo_summary="todo", relevant_files=[],
        implementation_plan=["add count method"], files_likely_to_change=["todo.py"],
        tests_to_add=[], risks=[], unknowns=[], raw_response="{}",
    )


def _coding() -> CodingResult:
    return CodingResult(
        summary="did it", reported_changed_files=(), tests_requested=(),
        known_issues=(), actual_changed_files=("todo.py",),
    )


def _passing():
    return CommandResult(
        command_name="pytest", argv=("pytest",), exit_code=0, stdout="ok",
        stderr="", timed_out=False, duration_seconds=0.1, output_truncated=False,
    )


def _approved():
    return ReviewResult(verdict="approved", summary="ok", issues=(), tests_assessment="p")


@pytest.fixture()
def svc(repo_settings):
    store = SqliteRunStore(repo_settings.devagent_database_path)
    service = WorkflowService(
        store,
        settings=repo_settings,
        planner=lambda task: _plan(),
        coder=lambda run, feedback: _coding(),
        verifier=lambda: (_passing(),),
        reviewer=lambda *a: (_approved(), 0, 0),
    )
    yield service
    store.close()


def _start(svc) -> str:
    run = svc.start_run("add count()")
    return str(run.run_id)


def test_no_command_is_usage_error(capsys):
    assert devagent.main([]) == devagent.EXIT_USAGE


def test_start_succeeds(svc, capsys):
    code = devagent.main(["start", "add", "count()"], service=svc)
    out = capsys.readouterr().out
    assert code == devagent.EXIT_OK
    assert "Run ID:" in out
    assert "awaiting_plan_approval" in out
    assert "No files have been modified." in out


def test_show_unknown_run(svc):
    assert devagent.main(["show", "nope"], service=svc) == devagent.EXIT_NOT_FOUND


def test_approve_then_resume_completes(svc, capsys):
    run_id = _start(svc)
    assert devagent.main(["approve", run_id], service=svc) == devagent.EXIT_OK
    code = devagent.main(["resume", run_id], service=svc)
    assert code == devagent.EXIT_OK
    assert "completed" in capsys.readouterr().out


def test_approve_wrong_status_is_refused(svc):
    run_id = _start(svc)
    devagent.main(["approve", run_id], service=svc)
    # Approving an already-approved run is refused.
    assert devagent.main(["approve", run_id], service=svc) == devagent.EXIT_REFUSED


def test_reject_succeeds(svc, capsys):
    run_id = _start(svc)
    assert devagent.main(["reject", run_id, "--comment", "no"], service=svc) == devagent.EXIT_OK
    assert "plan_rejected" in capsys.readouterr().out


def test_resume_before_approval_requires_approval(svc):
    run_id = _start(svc)
    assert devagent.main(["resume", run_id], service=svc) == devagent.EXIT_APPROVAL


def test_resume_rejected_run_not_resumable(svc):
    run_id = _start(svc)
    devagent.main(["reject", run_id], service=svc)
    assert devagent.main(["resume", run_id], service=svc) == devagent.EXIT_NOT_RESUMABLE


def test_list_and_events(svc, capsys):
    run_id = _start(svc)
    assert devagent.main(["list"], service=svc) == devagent.EXIT_OK
    assert run_id in capsys.readouterr().out
    assert devagent.main(["events", run_id], service=svc) == devagent.EXIT_OK
    assert "run.created" in capsys.readouterr().out


def test_show_reports_status(svc, capsys):
    run_id = _start(svc)
    assert devagent.main(["show", run_id], service=svc) == devagent.EXIT_OK
    assert "awaiting_plan_approval" in capsys.readouterr().out


def test_rollback_requires_confirmation(svc, monkeypatch, capsys):
    run_id = _start(svc)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    code = devagent.main(["rollback", run_id], service=svc)
    assert code == devagent.EXIT_OK
    assert "Aborted." in capsys.readouterr().out
