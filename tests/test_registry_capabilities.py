from __future__ import annotations

import pytest

from tools.registry import (
    PLANNER_TOOL_NAMES,
    RegisteredTool,
    execute_tool,
    get_tool_definitions,
    make_tool_executor,
)
from tools.schemas import ToolResult


def test_get_tool_definitions_returns_only_requested() -> None:
    defs = get_tool_definitions({"read_file"})
    names = {d["name"] for d in defs}
    assert names == {"read_file"}
    # Each definition is a valid Anthropic tool schema.
    for d in defs:
        assert set(d) == {"name", "description", "input_schema"}


def test_planner_profile_is_read_only() -> None:
    defs = get_tool_definitions(PLANNER_TOOL_NAMES)
    names = {d["name"] for d in defs}
    assert names == {"read_file", "list_files"}


def test_get_tool_definitions_rejects_unknown_name() -> None:
    with pytest.raises(ValueError):
        get_tool_definitions({"read_file", "not_a_tool"})


def test_registered_but_unapproved_tool_is_denied() -> None:
    """A real tool not in the run's profile is refused by dispatch."""
    result = execute_tool(
        "read_file",
        {"path": "README.md"},
        allowed_tools={"list_files"},
    )
    assert result.ok is False
    assert "not permitted" in result.content.lower()
    assert result.metadata == {"tool": "read_file", "denied": True}


def test_executor_closure_enforces_profile() -> None:
    executor = make_tool_executor({"list_files"})
    denied = executor("read_file", {"path": "README.md"})
    assert denied.ok is False
    assert "not permitted" in denied.content.lower()

    allowed = executor("list_files", {"path": "."})
    assert allowed.ok is True


def test_unknown_tool_still_denied_before_lookup() -> None:
    """An unknown name outside the profile is denied, not 'unknown'."""
    result = execute_tool(
        "definitely_not_a_tool",
        {},
        allowed_tools={"read_file"},
    )
    assert result.ok is False
    assert "not permitted" in result.content.lower()


def test_execute_tool_without_allowlist_permits_registered() -> None:
    """Backward-compatible default: None means all registered tools allowed."""
    result = execute_tool("list_files", {"path": "."})
    assert result.ok is True


def test_registered_tool_descriptor_shape() -> None:
    defs = get_tool_definitions(PLANNER_TOOL_NAMES)
    assert isinstance(defs, list)
    # RegisteredTool is the source-of-truth record; construct one to confirm
    # the public shape is stable for future write/command/git tools.
    sentinel = RegisteredTool(
        name="x",
        description="d",
        input_schema={"type": "object"},
        function=lambda: ToolResult(ok=True, content=""),
        category="read",
    )
    assert sentinel.category == "read"
