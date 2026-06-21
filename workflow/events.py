from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any
from uuid import UUID

if TYPE_CHECKING:
    from storage.base import RunStore

# A WorkflowRun snapshot answers "what is the state now?". This event log answers
# "how did it get there?". Events are append-only: there is deliberately no
# function to update or delete one, so the history can't be quietly rewritten.


class EventType(StrEnum):
    RUN_CREATED = "run.created"
    STAGE_STARTED = "stage.started"
    STAGE_COMPLETED = "stage.completed"
    STAGE_FAILED = "stage.failed"
    PLAN_CREATED = "plan.created"
    APPROVAL_REQUESTED = "approval.requested"
    APPROVAL_GRANTED = "approval.granted"
    APPROVAL_REJECTED = "approval.rejected"
    TOOL_REQUESTED = "tool.requested"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    FILE_CHANGED = "file.changed"
    COMMAND_STARTED = "command.started"
    COMMAND_COMPLETED = "command.completed"
    REVIEW_COMPLETED = "review.completed"
    REPAIR_STARTED = "repair.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_ROLLED_BACK = "run.rolled_back"


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    """One recorded fact about a run, in the order it happened.

    `sequence_number` is assigned by storage (1, 2, 3, ... per run) so ordering
    survives reload and two events can't share a slot.
    """

    event_id: UUID
    run_id: UUID
    sequence_number: int
    event_type: EventType
    stage: str | None
    agent_name: str | None
    payload: dict[str, Any]
    created_at: datetime


# --- Redaction -------------------------------------------------------------

REDACTED = "[REDACTED]"

# A value is dropped if its KEY looks secret (case-insensitive substring match)...
_SECRET_KEY_HINTS = (
    "api_key",
    "authorization",
    "token",
    "password",
    "secret",
    "cookie",
    "private_key",
)

# ...or if its string VALUE matches a known credential shape.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-ant-[A-Za-z0-9_\-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"\bBearer\s+\S+"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _key_is_secret(key: str) -> bool:
    lowered = key.lower()
    return any(hint in lowered for hint in _SECRET_KEY_HINTS)


def _redact_string(value: str) -> str:
    redacted = value
    for pattern in _SECRET_VALUE_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)
    return redacted


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: (REDACTED if _key_is_secret(key) else _redact(item))
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _redact_string(value)
    return value


def redact_event_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of `payload` with secret-looking keys and values masked.

    This is best-effort defence: callers should also avoid putting whole file
    contents or raw API responses in payloads in the first place (store hashes,
    paths, sizes, and short previews instead).
    """
    return _redact(payload)


def record_event(
    store: RunStore,
    run_id: str,
    event_type: EventType,
    *,
    stage: str | None = None,
    agent_name: str | None = None,
    payload: dict[str, Any] | None = None,
) -> WorkflowEvent:
    """Redact, then append one event. The store assigns the sequence number."""
    safe_payload = redact_event_payload(payload or {})
    return store.append_event(
        run_id,
        event_type,
        stage=stage,
        agent_name=agent_name,
        payload=safe_payload,
    )
