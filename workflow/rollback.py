from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from tools.file_tools import calculate_sha256
from tools.mutation_safety import is_protected_mutation_path
from tools.safety import resolve_inside_workspace
from tools.schemas import ToolResult
from tools.write_tools import _atomic_write
from workflow.errors import WorkflowError
from workflow.events import EventType, record_event
from workflow.models import RunStatus
from workflow.transitions import transition_run

if TYPE_CHECKING:
    from config import Settings
    from storage.base import RunStore

# Rollback only ever touches files THIS run changed, tracked via snapshots taken
# before the first edit of each file. It never runs `git reset --hard`, which
# would also wipe unrelated developer work. File deletion lives here, in trusted
# orchestration code, and is never exposed to the model.

_WRITE_TOOLS = frozenset({"create_file", "replace_in_file", "write_file"})

_ROLLBACKABLE = frozenset(
    {RunStatus.FAILED, RunStatus.CHANGES_REQUESTED, RunStatus.COMPLETED}
)


class RollbackError(WorkflowError):
    """Raised when a run's changes cannot be safely rolled back."""


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    """The state of one file before this run first modified it.

    `last_after_sha256` is the hash of the agent's most recent write, used to
    detect whether a human edited the file afterwards (which blocks rollback).
    """

    path: str
    existed_before: bool
    before_sha256: str | None
    last_after_sha256: str | None


class SnapshotStore:
    """Per-run pre-edit backups, kept on disk under the data dir.

    Backups live at `<data_dir>/runs/<run_id>/backups/`, named by a hash of the
    relative path (two directories can hold the same filename). The manifest maps
    each relative path to its snapshot.
    """

    def __init__(self, data_dir: Path | str, run_id: str) -> None:
        self._root = Path(data_dir) / "runs" / str(run_id)
        self._backups = self._root / "backups"
        self._manifest = self._root / "snapshots.json"

    def _load(self) -> dict[str, dict[str, Any]]:
        if self._manifest.exists():
            return json.loads(self._manifest.read_text(encoding="utf-8"))
        return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self._root.mkdir(parents=True, exist_ok=True)
        self._manifest.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def capture(self, workspace_root: Path | str, relative_path: str) -> None:
        """Snapshot a file before its first edit. First write wins, so a repair
        attempt never overwrites the original pre-run baseline."""
        data = self._load()
        if relative_path in data:
            return
        target = Path(workspace_root) / relative_path
        if target.exists() and target.is_file():
            try:
                content = target.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                # Not text we can snapshot. The write tools only mutate text
                # files, so they'll refuse this one anyway — skip rather than
                # crash, and let the tool return its own safe refusal.
                return
            self._backups.mkdir(parents=True, exist_ok=True)
            backup_name = (
                hashlib.sha256(relative_path.encode("utf-8")).hexdigest()
                + ".backup"
            )
            (self._backups / backup_name).write_text(content, encoding="utf-8")
            data[relative_path] = {
                "existed_before": True,
                "before_sha256": calculate_sha256(content),
                "backup": backup_name,
                "last_after_sha256": None,
            }
        else:
            data[relative_path] = {
                "existed_before": False,
                "before_sha256": None,
                "backup": None,
                "last_after_sha256": None,
            }
        self._save(data)

    def record_after(self, relative_path: str, after_sha256: str) -> None:
        data = self._load()
        if relative_path in data:
            data[relative_path]["last_after_sha256"] = after_sha256
            self._save(data)

    def snapshots(self) -> list[FileSnapshot]:
        return [
            FileSnapshot(
                path=path,
                existed_before=entry["existed_before"],
                before_sha256=entry["before_sha256"],
                last_after_sha256=entry.get("last_after_sha256"),
            )
            for path, entry in self._load().items()
        ]

    def read_backup(self, relative_path: str) -> str:
        entry = self._load()[relative_path]
        return (self._backups / entry["backup"]).read_text(encoding="utf-8")


ToolExecutor = Callable[[str, dict[str, Any]], ToolResult]


def make_recording_executor(
    base_executor: ToolExecutor,
    *,
    run_id: str,
    store: RunStore,
    snapshot_store: SnapshotStore,
    workspace_root: Path | str,
    stage: str = "implementing",
) -> ToolExecutor:
    """Wrap a tool executor so writes are snapshotted and audited.

    Before a write tool runs, the target file is snapshotted (so rollback has a
    pre-edit copy). After a successful write, the agent's new hash is recorded
    and a file.changed event is emitted from the tool's own metadata. Read/other
    tools pass straight through. The low-level write tools stay storage-free —
    all run context lives here. `stage` labels the emitted events so repair
    writes aren't logged as implementation writes.
    """

    def _executor(name: str, tool_input: dict[str, Any]) -> ToolResult:
        relative: str | None = None
        if name in _WRITE_TOOLS:
            relative = _safe_relative(tool_input.get("path"), workspace_root)
            if relative is not None:
                snapshot_store.capture(workspace_root, relative)

        result = base_executor(name, tool_input)

        if name in _WRITE_TOOLS and result.ok and result.metadata:
            meta = result.metadata
            after = meta.get("after_sha256")
            if relative is not None and isinstance(after, str):
                snapshot_store.record_after(relative, after)
            record_event(
                store,
                run_id,
                EventType.FILE_CHANGED,
                stage=stage,
                agent_name="coder",
                payload={
                    "path": meta.get("path", relative),
                    "operation": meta.get("operation"),
                    "before_sha256": meta.get("before_sha256"),
                    "after_sha256": after,
                    "chars_before": meta.get("chars_before"),
                    "chars_after": meta.get("chars_after"),
                },
            )
        return result

    return _executor


def _safe_relative(path: Any, workspace_root: Path | str) -> str | None:
    if not isinstance(path, str):
        return None
    try:
        resolved = resolve_inside_workspace(path, workspace_root=Path(workspace_root))
    except Exception:  # noqa: BLE001 - a bad path just means "don't snapshot"
        return None
    return str(resolved.relative_to(Path(workspace_root).resolve()))


def rollback_run(
    run_id: str,
    store: RunStore,
    settings: Settings,
) -> None:
    """Restore every file this run changed, then mark the run ROLLED_BACK.

    Two passes: validate all snapshots first (path safety, protected paths, and
    no post-agent user edits), then apply. If validation fails nothing is
    touched, so a refusal leaves the repository exactly as it was.
    """
    run = store.load_run(run_id)
    if run.status is RunStatus.ROLLED_BACK:
        raise RollbackError("Run has already been rolled back.")
    if run.status not in _ROLLBACKABLE:
        raise RollbackError(
            f"Cannot roll back a run in status {run.status.value}."
        )

    workspace = settings.tool_workspace_root
    snapshot_store = SnapshotStore(settings.devagent_data_dir, run_id)
    planned: list[tuple[FileSnapshot, Path]] = []
    for snapshot in snapshot_store.snapshots():
        # resolve_inside_workspace raises on any path escape.
        resolved = resolve_inside_workspace(snapshot.path, workspace_root=workspace)
        relative = resolved.relative_to(workspace.resolve())
        if is_protected_mutation_path(relative):
            raise RollbackError(
                f"Refusing to roll back a protected path: {snapshot.path}"
            )
        _ensure_not_user_modified(resolved, snapshot)
        planned.append((snapshot, resolved))

    restored: list[str] = []
    removed: list[str] = []
    for snapshot, resolved in planned:
        if snapshot.existed_before:
            _atomic_write(resolved, snapshot_store.read_backup(snapshot.path))
            restored.append(snapshot.path)
        elif resolved.exists():
            resolved.unlink()
            removed.append(snapshot.path)

    transition_run(run, RunStatus.ROLLED_BACK, current_stage="rolled_back")
    store.save_run(run)
    record_event(
        store, run_id, EventType.RUN_ROLLED_BACK, stage="rollback",
        payload={"restored": restored, "removed": removed},
    )


def _ensure_not_user_modified(resolved: Path, snapshot: FileSnapshot) -> None:
    """Block rollback if a human touched the file after the agent's last write."""
    if snapshot.last_after_sha256 is None:
        return
    if not resolved.exists():
        # The agent left a file here and it's now gone — a deletion we didn't
        # make. Refuse rather than silently recreate it.
        raise RollbackError(
            f"Refusing to roll back {snapshot.path}: it was deleted after the "
            "agent's change (unattributed edit)."
        )
    current = calculate_sha256(resolved.read_text(encoding="utf-8"))
    if current != snapshot.last_after_sha256:
        raise RollbackError(
            f"Refusing to roll back {snapshot.path}: it was modified after the "
            "agent's change (unattributed edit)."
        )
