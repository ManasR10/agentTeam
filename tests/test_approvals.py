from __future__ import annotations

from dataclasses import replace

import pytest

from agents.state import PlanningResult
from storage.sqlite_store import SqliteRunStore
from workflow.approvals import (
    ApprovalMismatchError,
    ApprovalNotAllowedError,
    approve_plan,
    calculate_plan_hash,
    ensure_approval_matches_plan,
    reject_plan,
)
from workflow.models import ApprovalDecision, RunStatus, WorkflowRun


def _plan(summary: str = "tiny todo app") -> PlanningResult:
    return PlanningResult(
        task="add count()",
        repo_summary=summary,
        relevant_files=[],
        implementation_plan=["add count method"],
        files_likely_to_change=["todo.py"],
        tests_to_add=[],
        risks=[],
        unknowns=[],
        raw_response="{}",
    )


def _awaiting_run() -> WorkflowRun:
    return WorkflowRun(
        task="add count()",
        workspace_root="/tmp/repo",
        starting_git_head="abc123",
        starting_worktree_clean=True,
        status=RunStatus.AWAITING_PLAN_APPROVAL,
        plan=_plan(),
    )


@pytest.fixture()
def store(tmp_path):
    s = SqliteRunStore(tmp_path / "devagent.db")
    yield s
    s.close()


def test_approve_moves_to_plan_approved(store):
    run = _awaiting_run()
    store.create_run(run)
    approved = approve_plan(store, str(run.run_id))
    assert approved.status is RunStatus.PLAN_APPROVED
    assert approved.approval is not None
    assert approved.approval.decision is ApprovalDecision.APPROVED


def test_reject_moves_to_plan_rejected(store):
    run = _awaiting_run()
    store.create_run(run)
    rejected = reject_plan(store, str(run.run_id), comment="too broad")
    assert rejected.status is RunStatus.PLAN_REJECTED
    assert rejected.approval.decision is ApprovalDecision.REJECTED
    assert rejected.approval.comment == "too broad"


def test_approval_binds_to_plan_hash(store):
    run = _awaiting_run()
    store.create_run(run)
    approved = approve_plan(store, str(run.run_id))
    assert approved.approval.plan_hash == calculate_plan_hash(run.plan)


def test_cannot_approve_run_that_is_not_awaiting(store):
    # A freshly created run is in CREATED, not awaiting approval.
    run = WorkflowRun(
        task="x",
        workspace_root="/tmp/repo",
        starting_git_head="abc",
        starting_worktree_clean=True,
    )
    store.create_run(run)
    with pytest.raises(ApprovalNotAllowedError):
        approve_plan(store, str(run.run_id))


def test_cannot_approve_twice(store):
    run = _awaiting_run()
    store.create_run(run)
    approve_plan(store, str(run.run_id))
    with pytest.raises(ApprovalNotAllowedError):
        approve_plan(store, str(run.run_id))


def test_cannot_approve_after_rejection(store):
    run = _awaiting_run()
    store.create_run(run)
    reject_plan(store, str(run.run_id))
    with pytest.raises(ApprovalNotAllowedError):
        approve_plan(store, str(run.run_id))


def test_modified_plan_invalidates_approval(store):
    run = _awaiting_run()
    store.create_run(run)
    approved = approve_plan(store, str(run.run_id))
    # Tamper with the plan after approval; the bound hash no longer matches.
    approved.plan = replace(approved.plan, implementation_plan=["something else"])
    with pytest.raises(ApprovalMismatchError):
        ensure_approval_matches_plan(approved)


def test_unchanged_plan_passes_approval_check(store):
    run = _awaiting_run()
    store.create_run(run)
    approved = approve_plan(store, str(run.run_id))
    ensure_approval_matches_plan(approved)  # should not raise


def test_plan_hash_is_deterministic():
    assert calculate_plan_hash(_plan()) == calculate_plan_hash(_plan())
    assert calculate_plan_hash(_plan("a")) != calculate_plan_hash(_plan("b"))


def test_approval_survives_restart(tmp_path):
    db = tmp_path / "devagent.db"
    run = _awaiting_run()
    first = SqliteRunStore(db)
    first.create_run(run)
    approve_plan(first, str(run.run_id))
    first.close()

    second = SqliteRunStore(db)
    try:
        reloaded = second.load_run(str(run.run_id))
        assert reloaded.status is RunStatus.PLAN_APPROVED
        assert reloaded.approval is not None
        ensure_approval_matches_plan(reloaded)
    finally:
        second.close()


def test_approval_row_recorded_in_table(store):
    run = _awaiting_run()
    store.create_run(run)
    approve_plan(store, str(run.run_id))
    rows = store._conn.execute(
        "SELECT decision FROM approvals WHERE run_id = ?", (str(run.run_id),)
    ).fetchall()
    assert [r["decision"] for r in rows] == ["approved"]


def test_approval_writes_state_table_and_event_together(store):
    # The run state, approvals row, and audit event are committed in one
    # transaction, so all three are present after an approval.
    from workflow.events import EventType

    run = _awaiting_run()
    store.create_run(run)
    approve_plan(store, str(run.run_id))

    assert store.load_run(str(run.run_id)).status is RunStatus.PLAN_APPROVED
    approvals = store._conn.execute(
        "SELECT 1 FROM approvals WHERE run_id = ?", (str(run.run_id),)
    ).fetchall()
    assert len(approvals) == 1
    event_types = [e.event_type for e in store.list_events(str(run.run_id))]
    assert EventType.APPROVAL_GRANTED in event_types
