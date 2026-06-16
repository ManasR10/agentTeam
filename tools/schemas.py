from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    """
    Standard result returned by every DevAgent tool.

    The tool loop will convert this into an Anthropic tool_result block.

    Attributes:
        ok:
            True if the tool completed its requested operation.
        content:
            Human/model readable result text (file content, error message,
            file listing, ...).
        metadata:
            Optional structured extra information (path, char counts, ...).
    """

    ok: bool
    content: str
    metadata: dict[str, Any] | None = None


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot complete its requested operation."""
