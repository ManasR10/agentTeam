from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Low-level execution result records produced by Phase 3 tools. These live in
# the tools layer (never importing anything from agents/) so the agent layer
# can depend on them, but not the reverse.


@dataclass(frozen=True, slots=True)
class FileChange:
    """
    A single applied file mutation, derived from a write tool's metadata.

    The orchestrator converts a tool's human-readable ToolResult into this
    structured record instead of parsing the tool's content string.
    """

    path: str
    operation: Literal["create", "replace", "rewrite"]
    before_sha256: str | None
    after_sha256: str
    chars_before: int
    chars_after: int


@dataclass(frozen=True, slots=True)
class CommandResult:
    """
    Outcome of one restricted command (tests, checks, git) run by Python.

    `exit_code` is None only when `timed_out` is True. `timed_out` is reported
    as a normal failed result rather than being allowed to hang the agent.
    """

    command_name: str
    argv: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float
    output_truncated: bool
