from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from workflow.errors import InvalidRunTransitionError
from workflow.models import RunStatus, WorkflowRun
from workflow.transitions import (
    ALLOWED_TRANSITIONS,
    TERMINAL_STATUSES,
    is_terminal,
    transition_run,
)


def _run(status: RunStatus = RunStatus.CREATED) -> WorkflowRun:
    run = WorkflowRun(
        task="do a thing",
        workspace_root="/tmp/repo",
        starting_git_head="abc123",
        starting_worktree_clean=True,
    )
    run.status = status
    return run


def test_created_to_planning_allowed():
    run = transition_run(_run(), RunStatus.PLANNING, current_stage="planning")
    assert run.status is RunStatus.PLANNING
    assert run.current_stage == "planning"


def test_planning_to_awaiting_approval_allowed():
    run = transition_run(_run(RunStatus.PLANNING), RunStatus.AWAITING_PLAN_APPROVAL)
    assert run.status is RunStatus.AWAITING_PLAN_APPROVAL


def test_awaiting_approval_to_plan_approved_allowed():
    run = transition_run(
        _run(RunStatus.AWAITING_PLAN_APPROVAL), RunStatus.PLAN_APPROVED
    )
    assert run.status is RunStatus.PLAN_APPROVED


def test_awaiting_approval_to_plan_rejected_allowed():
    run = transition_run(
        _run(RunStatus.AWAITING_PLAN_APPROVAL), RunStatus.PLAN_REJECTED
    )
    assert run.status is RunStatus.PLAN_REJECTED


def test_plan_approved_to_implementing_allowed():
    run = transition_run(_run(RunStatus.PLAN_APPROVED), RunStatus.IMPLEMENTING)
    assert run.status is RunStatus.IMPLEMENTING


def test_awaiting_approval_cannot_skip_straight_to_implementing():
    # The whole point of the approval gate: coding can't start without an
    # explicit PLAN_APPROVED step in between.
    with pytest.raises(InvalidRunTransitionError):
        transition_run(_run(RunStatus.AWAITING_PLAN_APPROVAL), RunStatus.IMPLEMENTING)


def test_plan_rejected_to_implementing_blocked():
    with pytest.raises(InvalidRunTransitionError):
        transition_run(_run(RunStatus.PLAN_REJECTED), RunStatus.IMPLEMENTING)


def test_completed_to_implementing_blocked():
    with pytest.raises(InvalidRunTransitionError):
        transition_run(_run(RunStatus.COMPLETED), RunStatus.IMPLEMENTING)


def test_invalid_transition_message_names_both_states():
    with pytest.raises(InvalidRunTransitionError) as exc_info:
        transition_run(_run(RunStatus.PLAN_REJECTED), RunStatus.IMPLEMENTING)
    message = str(exc_info.value)
    assert "plan_rejected" in message
    assert "implementing" in message
    assert exc_info.value.from_status is RunStatus.PLAN_REJECTED
    assert exc_info.value.to_status is RunStatus.IMPLEMENTING


def test_transition_updates_timestamp():
    run = _run()
    run.updated_at = datetime.now(timezone.utc) - timedelta(hours=1)
    before = run.updated_at
    transition_run(run, RunStatus.PLANNING)
    assert run.updated_at > before


def test_current_stage_unchanged_when_not_given():
    run = _run()
    run.current_stage = "planning"
    transition_run(run, RunStatus.PLANNING)
    assert run.current_stage == "planning"


def test_completed_can_roll_back():
    run = transition_run(_run(RunStatus.COMPLETED), RunStatus.ROLLED_BACK)
    assert run.status is RunStatus.ROLLED_BACK


def test_terminal_detection():
    assert is_terminal(RunStatus.COMPLETED)
    assert is_terminal(RunStatus.CANCELLED)
    assert not is_terminal(RunStatus.PLANNING)
    assert not is_terminal(RunStatus.FAILED)  # not terminal: can still roll back


def test_terminal_dead_ends_have_no_outgoing_transitions():
    for status in (RunStatus.PLAN_REJECTED, RunStatus.CANCELLED, RunStatus.ROLLED_BACK):
        assert ALLOWED_TRANSITIONS[status] == frozenset()


def test_every_status_is_in_the_table():
    assert set(ALLOWED_TRANSITIONS) == set(RunStatus)


def test_transition_targets_are_valid_statuses():
    for targets in ALLOWED_TRANSITIONS.values():
        for target in targets:
            assert isinstance(target, RunStatus)


def test_terminal_set_matches_dead_ends_plus_completed():
    # COMPLETED is terminal but keeps a rollback edge; the other terminals are
    # true dead ends. Guard the intended membership so it can't drift silently.
    assert TERMINAL_STATUSES == frozenset(
        {
            RunStatus.COMPLETED,
            RunStatus.PLAN_REJECTED,
            RunStatus.CANCELLED,
            RunStatus.ROLLED_BACK,
        }
    )
