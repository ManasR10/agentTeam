from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from storage.base import RunAlreadyExistsError, RunNotFoundError
from storage.sqlite_store import SqliteRunStore
from workflow.models import RunStatus, WorkflowRun
from workflow.serialization import SerializationError


def _run(task: str = "do a thing", **overrides) -> WorkflowRun:
    base = dict(
        task=task,
        workspace_root="/tmp/repo",
        starting_git_head="abc123",
        starting_worktree_clean=True,
    )
    base.update(overrides)
    return WorkflowRun(**base)


@pytest.fixture()
def store(tmp_path):
    db = tmp_path / "nested" / "devagent.db"
    s = SqliteRunStore(db)
    yield s
    s.close()


def test_database_file_is_created(tmp_path):
    db = tmp_path / "data" / "devagent.db"
    SqliteRunStore(db).close()
    assert db.exists()


def test_create_and_load_round_trip(store):
    run = _run()
    store.create_run(run)
    assert store.load_run(str(run.run_id)) == run


def test_duplicate_create_rejected(store):
    run = _run()
    store.create_run(run)
    with pytest.raises(RunAlreadyExistsError):
        store.create_run(run)


def test_load_unknown_run_raises(store):
    with pytest.raises(RunNotFoundError):
        store.load_run("does-not-exist")


def test_save_updates_existing_run(store):
    run = _run()
    store.create_run(run)
    run.status = RunStatus.PLANNING
    run.current_stage = "planning"
    store.save_run(run)
    loaded = store.load_run(str(run.run_id))
    assert loaded.status is RunStatus.PLANNING
    assert loaded.current_stage == "planning"


def test_save_unknown_run_raises(store):
    with pytest.raises(RunNotFoundError):
        store.save_run(_run())


def test_list_runs_is_most_recent_first(store):
    older = _run("older", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    newer = _run("newer", created_at=datetime(2026, 6, 1, tzinfo=timezone.utc))
    store.create_run(older)
    store.create_run(newer)
    tasks = [s.task for s in store.list_runs()]
    assert tasks == ["newer", "older"]


def test_list_runs_respects_limit(store):
    for i in range(5):
        store.create_run(
            _run(f"task {i}", created_at=datetime(2026, 1, 1, i + 1, tzinfo=timezone.utc))
        )
    assert len(store.list_runs(limit=2)) == 2


def test_list_runs_filters_by_status(store):
    a = _run("planning one", status=RunStatus.PLANNING)
    b = _run("created one")
    store.create_run(a)
    store.create_run(b)
    planning = store.list_runs(status=RunStatus.PLANNING)
    assert [s.task for s in planning] == ["planning one"]


def test_invalid_state_in_db_is_rejected(store, tmp_path):
    run = _run()
    store.create_run(run)
    # Corrupt the stored JSON directly, then prove load refuses it.
    store._conn.execute(
        "UPDATE workflow_runs SET state_json = ? WHERE run_id = ?",
        ("{not valid json", str(run.run_id)),
    )
    store._conn.commit()
    with pytest.raises(SerializationError):
        store.load_run(str(run.run_id))


def test_failed_create_does_not_corrupt_existing_run(store):
    run = _run()
    store.create_run(run)
    with pytest.raises(RunAlreadyExistsError):
        store.create_run(run)
    # The original is still intact and loadable.
    assert store.load_run(str(run.run_id)) == run


def test_run_survives_reopening_the_database(tmp_path):
    db = tmp_path / "devagent.db"
    run = _run()
    first = SqliteRunStore(db)
    first.create_run(run)
    first.close()

    second = SqliteRunStore(db)
    try:
        assert second.load_run(str(run.run_id)) == run
    finally:
        second.close()


def test_foreign_keys_enabled(store):
    assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
