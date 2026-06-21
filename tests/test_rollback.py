from __future__ import annotations

from dataclasses import replace

import pytest

from config import get_settings
from storage.sqlite_store import SqliteRunStore
from tools.file_tools import calculate_sha256
from tools.safety import PathSafetyError
from tools.schemas import ToolResult
from workflow.events import EventType
from workflow.models import RunStatus, WorkflowRun
from workflow.rollback import (
    RollbackError,
    SnapshotStore,
    make_recording_executor,
    rollback_run,
)


@pytest.fixture()
def settings(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return replace(
        get_settings(),
        tool_workspace_root=workspace,
        devagent_data_dir=tmp_path / "data",
        devagent_database_path=tmp_path / "data" / "devagent.db",
    )


@pytest.fixture()
def store(settings):
    s = SqliteRunStore(settings.devagent_database_path)
    yield s
    s.close()


def _failed_run(settings) -> WorkflowRun:
    return WorkflowRun(
        task="t",
        workspace_root=str(settings.tool_workspace_root),
        starting_git_head="abc",
        starting_worktree_clean=True,
        status=RunStatus.FAILED,
    )


def _snap_store(settings, run) -> SnapshotStore:
    return SnapshotStore(settings.devagent_data_dir, str(run.run_id))


def test_rollback_restores_a_modified_file(settings, store):
    ws = settings.tool_workspace_root
    (ws / "file.txt").write_text("original", encoding="utf-8")
    run = _failed_run(settings)
    store.create_run(run)

    snaps = _snap_store(settings, run)
    snaps.capture(ws, "file.txt")
    (ws / "file.txt").write_text("changed by agent", encoding="utf-8")
    snaps.record_after("file.txt", calculate_sha256("changed by agent"))

    rollback_run(str(run.run_id), store, settings)
    assert (ws / "file.txt").read_text(encoding="utf-8") == "original"
    assert store.load_run(str(run.run_id)).status is RunStatus.ROLLED_BACK


def test_rollback_removes_an_agent_created_file(settings, store):
    ws = settings.tool_workspace_root
    run = _failed_run(settings)
    store.create_run(run)

    snaps = _snap_store(settings, run)
    snaps.capture(ws, "new.py")  # captured before it exists
    (ws / "new.py").write_text("print('hi')\n", encoding="utf-8")
    snaps.record_after("new.py", calculate_sha256("print('hi')\n"))

    rollback_run(str(run.run_id), store, settings)
    assert not (ws / "new.py").exists()


def test_rollback_leaves_unrelated_files_untouched(settings, store):
    ws = settings.tool_workspace_root
    (ws / "file.txt").write_text("original", encoding="utf-8")
    (ws / "unrelated.txt").write_text("user data", encoding="utf-8")
    run = _failed_run(settings)
    store.create_run(run)

    snaps = _snap_store(settings, run)
    snaps.capture(ws, "file.txt")
    (ws / "file.txt").write_text("agent", encoding="utf-8")
    snaps.record_after("file.txt", calculate_sha256("agent"))

    rollback_run(str(run.run_id), store, settings)
    assert (ws / "unrelated.txt").read_text(encoding="utf-8") == "user data"


def test_rollback_refuses_when_user_edited_after_agent(settings, store):
    ws = settings.tool_workspace_root
    (ws / "file.txt").write_text("original", encoding="utf-8")
    run = _failed_run(settings)
    store.create_run(run)

    snaps = _snap_store(settings, run)
    snaps.capture(ws, "file.txt")
    snaps.record_after("file.txt", calculate_sha256("what the agent wrote"))
    # But the file on disk is something else — a human touched it afterwards.
    (ws / "file.txt").write_text("human edit", encoding="utf-8")

    with pytest.raises(RollbackError, match="modified after"):
        rollback_run(str(run.run_id), store, settings)
    # Nothing was restored — the file is left as the user left it.
    assert (ws / "file.txt").read_text(encoding="utf-8") == "human edit"


def test_rollback_refuses_protected_path(settings, store):
    run = _failed_run(settings)
    store.create_run(run)
    snaps = _snap_store(settings, run)
    # Hand-write a manifest entry for a protected path.
    snaps._save({".env": {"existed_before": True, "before_sha256": "x", "backup": "x.backup", "last_after_sha256": None}})
    with pytest.raises(RollbackError, match="protected"):
        rollback_run(str(run.run_id), store, settings)


def test_rollback_rejects_path_escape(settings, store):
    run = _failed_run(settings)
    store.create_run(run)
    snaps = _snap_store(settings, run)
    snaps._save({"../escape.txt": {"existed_before": False, "before_sha256": None, "backup": None, "last_after_sha256": None}})
    with pytest.raises(PathSafetyError):
        rollback_run(str(run.run_id), store, settings)


def test_second_rollback_is_rejected(settings, store):
    ws = settings.tool_workspace_root
    (ws / "file.txt").write_text("original", encoding="utf-8")
    run = _failed_run(settings)
    store.create_run(run)
    snaps = _snap_store(settings, run)
    snaps.capture(ws, "file.txt")
    (ws / "file.txt").write_text("agent", encoding="utf-8")
    snaps.record_after("file.txt", calculate_sha256("agent"))

    rollback_run(str(run.run_id), store, settings)
    with pytest.raises(RollbackError, match="already"):
        rollback_run(str(run.run_id), store, settings)


def test_rollback_event_is_recorded(settings, store):
    ws = settings.tool_workspace_root
    (ws / "file.txt").write_text("original", encoding="utf-8")
    run = _failed_run(settings)
    store.create_run(run)
    snaps = _snap_store(settings, run)
    snaps.capture(ws, "file.txt")
    (ws / "file.txt").write_text("agent", encoding="utf-8")
    snaps.record_after("file.txt", calculate_sha256("agent"))

    rollback_run(str(run.run_id), store, settings)
    types = [e.event_type for e in store.list_events(str(run.run_id))]
    assert EventType.RUN_ROLLED_BACK in types


def test_cannot_roll_back_a_running_run(settings, store):
    run = WorkflowRun(
        task="t", workspace_root=str(settings.tool_workspace_root),
        starting_git_head="abc", starting_worktree_clean=True,
        status=RunStatus.CREATED,
    )
    store.create_run(run)
    with pytest.raises(RollbackError):
        rollback_run(str(run.run_id), store, settings)


# --- recording executor ----------------------------------------------------


def test_recording_executor_snapshots_and_emits_event(settings, store):
    ws = settings.tool_workspace_root
    (ws / "todo.py").write_text("orig", encoding="utf-8")
    run = _failed_run(settings)
    store.create_run(run)
    snaps = _snap_store(settings, run)

    def fake_base(name, tool_input):
        # Pretend the write happened; return the metadata a real write tool would.
        return ToolResult(
            ok=True, content="ok",
            metadata={
                "path": "todo.py", "operation": "replace",
                "before_sha256": calculate_sha256("orig"),
                "after_sha256": "deadbeef",
                "chars_before": 4, "chars_after": 10,
            },
        )

    executor = make_recording_executor(
        fake_base, run_id=str(run.run_id), store=store,
        snapshot_store=snaps, workspace_root=ws,
    )
    result = executor("replace_in_file", {"path": "todo.py", "old_text": "o", "new_text": "x"})
    assert result.ok

    snapshots = {s.path: s for s in snaps.snapshots()}
    assert "todo.py" in snapshots
    assert snapshots["todo.py"].existed_before is True
    assert snapshots["todo.py"].last_after_sha256 == "deadbeef"

    file_events = [
        e for e in store.list_events(str(run.run_id))
        if e.event_type is EventType.FILE_CHANGED
    ]
    assert len(file_events) == 1
    assert file_events[0].payload["path"] == "todo.py"


def test_capture_skips_binary_files_without_crashing(settings):
    ws = settings.tool_workspace_root
    (ws / "logo.png").write_bytes(b"\xff\xfe\x00\x01")
    run = _failed_run(settings)
    snaps = _snap_store(settings, run)
    snaps.capture(ws, "logo.png")  # must not raise on non-UTF8
    # Skipped, so the write tool (which refuses binaries) is free to do its thing.
    assert snaps.snapshots() == []


def test_rollback_refuses_when_agent_file_deleted_afterwards(settings, store):
    ws = settings.tool_workspace_root
    (ws / "file.txt").write_text("original", encoding="utf-8")
    run = _failed_run(settings)
    store.create_run(run)
    snaps = _snap_store(settings, run)
    snaps.capture(ws, "file.txt")
    snaps.record_after("file.txt", calculate_sha256("agent wrote this"))
    # A human deletes the file after the agent's change.
    (ws / "file.txt").unlink()
    with pytest.raises(RollbackError, match="deleted after"):
        rollback_run(str(run.run_id), store, settings)


def test_recording_executor_labels_repair_stage(settings, store):
    ws = settings.tool_workspace_root
    (ws / "todo.py").write_text("orig", encoding="utf-8")
    run = _failed_run(settings)
    store.create_run(run)
    snaps = _snap_store(settings, run)

    def fake_base(name, tool_input):
        return ToolResult(
            ok=True, content="ok",
            metadata={"path": "todo.py", "operation": "rewrite",
                      "after_sha256": "abc", "before_sha256": "x",
                      "chars_before": 4, "chars_after": 5},
        )

    executor = make_recording_executor(
        fake_base, run_id=str(run.run_id), store=store, snapshot_store=snaps,
        workspace_root=ws, stage="repair",
    )
    executor("write_file", {"path": "todo.py", "content": "x", "expected_sha256": "x"})
    events = [e for e in store.list_events(str(run.run_id)) if e.event_type is EventType.FILE_CHANGED]
    assert len(events) == 1  # exactly one file.changed per write, no duplicate
    assert events[0].stage == "repair"


def test_recording_executor_passes_through_reads(settings, store):
    run = _failed_run(settings)
    store.create_run(run)
    snaps = _snap_store(settings, run)

    def fake_base(name, tool_input):
        return ToolResult(ok=True, content="file contents")

    executor = make_recording_executor(
        fake_base, run_id=str(run.run_id), store=store,
        snapshot_store=snaps, workspace_root=settings.tool_workspace_root,
    )
    result = executor("read_file", {"path": "todo.py"})
    assert result.content == "file contents"
    assert snaps.snapshots() == []  # reads never snapshot
    assert store.list_events(str(run.run_id)) == []
