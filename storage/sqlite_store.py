from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from storage.base import (
    RunAlreadyExistsError,
    RunNotFoundError,
    WorkflowRunSummary,
)
from storage.migrations import apply_schema
from workflow.events import EventType, WorkflowEvent
from workflow.models import ApprovalRecord, RunStatus, WorkflowRun
from workflow.serialization import (
    deserialize_workflow_run,
    serialize_workflow_run,
)


class SqliteRunStore:
    """A `RunStore` backed by a single SQLite database file.

    One connection is held for the store's lifetime (the CLI is single-process),
    with foreign keys on and WAL journalling for durability. Every write runs in
    a transaction via `with self._conn`, so a failure mid-write leaves the
    previous state intact rather than half-applied.
    """

    def __init__(self, database_path: Path | str) -> None:
        self._path = Path(database_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # The data dir lives inside the target repo. Drop a self-ignoring
        # .gitignore so git never reports our database as an uncommitted change
        # (which would otherwise fail the clean-worktree check) regardless of the
        # repo's own ignore rules.
        gitignore = self._path.parent / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n", encoding="utf-8")
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        # Pragmas are connection-level, set once up front.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        apply_schema(self._conn)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SqliteRunStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def create_run(self, run: WorkflowRun) -> None:
        try:
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO workflow_runs (
                        run_id, task, workspace_root, status, current_stage,
                        state_json, schema_version, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    self._row(run),
                )
        except sqlite3.IntegrityError as exc:
            raise RunAlreadyExistsError(
                f"Run already exists: {run.run_id}"
            ) from exc

    def save_run(self, run: WorkflowRun) -> None:
        with self._conn:
            if self._update_run_row(run) == 0:
                # No row updated: the run was never created. The transaction
                # rolls back on this raise (nothing was written anyway).
                raise RunNotFoundError(f"Run not found: {run.run_id}")

    def load_run(self, run_id: str) -> WorkflowRun:
        row = self._conn.execute(
            "SELECT state_json FROM workflow_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise RunNotFoundError(f"Run not found: {run_id}")
        # Corrupt/inconsistent stored state raises SerializationError here.
        return deserialize_workflow_run(row["state_json"])

    def list_runs(
        self,
        *,
        limit: int = 20,
        status: RunStatus | None = None,
    ) -> list[WorkflowRunSummary]:
        columns = (
            "run_id, task, status, current_stage, created_at, updated_at"
        )
        if status is not None:
            rows = self._conn.execute(
                f"SELECT {columns} FROM workflow_runs "
                "WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status.value, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                f"SELECT {columns} FROM workflow_runs "
                "ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._summary(row) for row in rows]

    def append_event(
        self,
        run_id: str | UUID,
        event_type: EventType,
        *,
        stage: str | None = None,
        agent_name: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        with self._conn:
            return self._insert_event_row(
                run_id, event_type, stage, agent_name, payload or {}
            )

    def record_approval(self, approval: ApprovalRecord) -> None:
        with self._conn:
            self._insert_approval_row(approval)

    def commit_approval(
        self,
        run: WorkflowRun,
        approval: ApprovalRecord,
        *,
        event_type: EventType,
        event_payload: dict[str, Any],
    ) -> None:
        """Persist an approval decision atomically.

        The run state, the queryable approvals row, and the audit event are
        written in ONE transaction, so a crash can't leave the run approved with
        the approval table or audit log missing (or vice versa).
        """
        with self._conn:
            if self._update_run_row(run) == 0:
                raise RunNotFoundError(f"Run not found: {run.run_id}")
            self._insert_approval_row(approval)
            self._insert_event_row(
                run.run_id, event_type, "approval", None, event_payload
            )

    def list_events(self, run_id: str | UUID) -> list[WorkflowEvent]:
        rows = self._conn.execute(
            "SELECT * FROM workflow_events WHERE run_id = ? "
            "ORDER BY sequence_number ASC",
            (str(run_id),),
        ).fetchall()
        return [self._event(row) for row in rows]

    # --- helpers (run inside a caller's transaction) -----------------------

    def _update_run_row(self, run: WorkflowRun) -> int:
        cursor = self._conn.execute(
            """
            UPDATE workflow_runs SET
                task = ?, workspace_root = ?, status = ?, current_stage = ?,
                state_json = ?, schema_version = ?, created_at = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (*self._row(run)[1:], str(run.run_id)),
        )
        return cursor.rowcount

    def _insert_approval_row(self, approval: ApprovalRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO approvals (
                approval_id, run_id, decision, plan_hash, comment, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                str(approval.approval_id),
                str(approval.run_id),
                approval.decision.value,
                approval.plan_hash,
                approval.comment,
                approval.created_at.isoformat(),
            ),
        )

    def _insert_event_row(
        self,
        run_id: str | UUID,
        event_type: EventType,
        stage: str | None,
        agent_name: str | None,
        payload: dict[str, Any],
    ) -> WorkflowEvent:
        # Pick the next slot and insert together, so two appends can't read the
        # same MAX and collide; the UNIQUE constraint is the backstop.
        event_id = uuid4()
        created_at = datetime.now(timezone.utc)
        row = self._conn.execute(
            "SELECT COALESCE(MAX(sequence_number), 0) + 1 AS next "
            "FROM workflow_events WHERE run_id = ?",
            (str(run_id),),
        ).fetchone()
        sequence_number = row["next"]
        self._conn.execute(
            """
            INSERT INTO workflow_events (
                event_id, run_id, sequence_number, event_type, stage,
                agent_name, payload_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(event_id),
                str(run_id),
                sequence_number,
                event_type.value,
                stage,
                agent_name,
                json.dumps(payload),
                created_at.isoformat(),
            ),
        )
        return WorkflowEvent(
            event_id=event_id,
            run_id=UUID(str(run_id)),
            sequence_number=sequence_number,
            event_type=event_type,
            stage=stage,
            agent_name=agent_name,
            payload=payload,
            created_at=created_at,
        )

    @staticmethod
    def _event(row: sqlite3.Row) -> WorkflowEvent:
        return WorkflowEvent(
            event_id=UUID(row["event_id"]),
            run_id=UUID(row["run_id"]),
            sequence_number=row["sequence_number"],
            event_type=EventType(row["event_type"]),
            stage=row["stage"],
            agent_name=row["agent_name"],
            payload=json.loads(row["payload_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row(run: WorkflowRun) -> tuple[object, ...]:
        return (
            str(run.run_id),
            run.task,
            run.workspace_root,
            run.status.value,
            run.current_stage,
            serialize_workflow_run(run),
            run.schema_version,
            run.created_at.isoformat(),
            run.updated_at.isoformat(),
        )

    @staticmethod
    def _summary(row: sqlite3.Row) -> WorkflowRunSummary:
        return WorkflowRunSummary(
            run_id=row["run_id"],
            task=row["task"],
            status=RunStatus(row["status"]),
            current_stage=row["current_stage"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )
