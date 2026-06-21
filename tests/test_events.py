from __future__ import annotations

import sqlite3
from uuid import uuid4

import pytest

from storage.sqlite_store import SqliteRunStore
from workflow.events import (
    REDACTED,
    EventType,
    record_event,
    redact_event_payload,
)
from workflow.models import WorkflowRun


def _run() -> WorkflowRun:
    return WorkflowRun(
        task="do a thing",
        workspace_root="/tmp/repo",
        starting_git_head="abc123",
        starting_worktree_clean=True,
    )


@pytest.fixture()
def store_with_run(tmp_path):
    store = SqliteRunStore(tmp_path / "devagent.db")
    run = _run()
    store.create_run(run)  # events FK-reference an existing run
    yield store, str(run.run_id)
    store.close()


def test_events_append_in_order_starting_at_one(store_with_run):
    store, run_id = store_with_run
    record_event(store, run_id, EventType.RUN_CREATED)
    record_event(store, run_id, EventType.STAGE_STARTED, stage="planning")
    record_event(store, run_id, EventType.PLAN_CREATED, stage="planning")
    events = store.list_events(run_id)
    assert [e.sequence_number for e in events] == [1, 2, 3]
    assert [e.event_type for e in events] == [
        EventType.RUN_CREATED,
        EventType.STAGE_STARTED,
        EventType.PLAN_CREATED,
    ]


def test_events_carry_stage_and_agent(store_with_run):
    store, run_id = store_with_run
    event = record_event(
        store,
        run_id,
        EventType.TOOL_COMPLETED,
        stage="implementing",
        agent_name="coder",
        payload={"tool": "replace_in_file", "path": "todo.py"},
    )
    assert event.stage == "implementing"
    assert event.agent_name == "coder"
    assert event.payload["path"] == "todo.py"


def test_duplicate_sequence_number_rejected(store_with_run):
    store, run_id = store_with_run
    record_event(store, run_id, EventType.RUN_CREATED)
    # Force a second event into sequence 1 — the UNIQUE constraint must refuse.
    with pytest.raises(sqlite3.IntegrityError):
        store._conn.execute(
            """
            INSERT INTO workflow_events (
                event_id, run_id, sequence_number, event_type, payload_json,
                created_at
            ) VALUES (?, ?, 1, 'run.created', '{}', '2026-01-01T00:00:00')
            """,
            (str(uuid4()), run_id),
        )


def test_secret_keys_are_redacted():
    payload = {
        "api_key": "sk-ant-abc",
        "Authorization": "Bearer xyz",
        "nested": {"password": "hunter2", "ok": "keep me"},
    }
    safe = redact_event_payload(payload)
    assert safe["api_key"] == REDACTED
    assert safe["Authorization"] == REDACTED
    assert safe["nested"]["password"] == REDACTED
    assert safe["nested"]["ok"] == "keep me"


def test_secret_looking_values_are_redacted():
    payload = {"note": "the key is sk-ant-supersecret123 do not share"}
    safe = redact_event_payload(payload)
    assert "sk-ant-supersecret123" not in safe["note"]
    assert REDACTED in safe["note"]


def test_bearer_and_private_key_values_redacted():
    payload = {
        "header": "Authorization: Bearer abc.def.ghi",
        "pem": "-----BEGIN PRIVATE KEY-----\nMIIE...",
    }
    safe = redact_event_payload(payload)
    assert REDACTED in safe["header"]
    assert REDACTED in safe["pem"]


def test_record_event_redacts_before_storing(store_with_run):
    store, run_id = store_with_run
    record_event(
        store,
        run_id,
        EventType.TOOL_COMPLETED,
        payload={"api_key": "sk-ant-leak"},
    )
    stored = store.list_events(run_id)[0]
    assert stored.payload["api_key"] == REDACTED


def test_events_listed_per_run(tmp_path):
    store = SqliteRunStore(tmp_path / "devagent.db")
    a, b = _run(), _run()
    store.create_run(a)
    store.create_run(b)
    record_event(store, str(a.run_id), EventType.RUN_CREATED)
    record_event(store, str(b.run_id), EventType.RUN_CREATED)
    record_event(store, str(b.run_id), EventType.STAGE_STARTED)
    try:
        assert len(store.list_events(str(a.run_id))) == 1
        assert len(store.list_events(str(b.run_id))) == 2
    finally:
        store.close()


def test_events_survive_reopen(tmp_path):
    db = tmp_path / "devagent.db"
    run = _run()
    first = SqliteRunStore(db)
    first.create_run(run)
    record_event(first, str(run.run_id), EventType.RUN_CREATED)
    first.close()

    second = SqliteRunStore(db)
    try:
        events = second.list_events(str(run.run_id))
        assert len(events) == 1
        assert events[0].event_type is EventType.RUN_CREATED
    finally:
        second.close()
