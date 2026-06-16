from __future__ import annotations

from tools.registry import execute_tool


def test_execute_unknown_tool_returns_error() -> None:
    """An unregistered tool name must fail safely, not raise."""
    result = execute_tool("definitely_not_a_real_tool", {})
    assert result.ok is False
    assert "unknown tool" in result.content.lower()
    assert result.metadata == {"tool": "definitely_not_a_real_tool"}


def test_execute_tool_with_invalid_input_returns_error() -> None:
    """Bad arguments hit the TypeError guard and come back as ok=False."""
    # read_file() accepts `path`; an unexpected keyword raises TypeError,
    # which execute_tool() must convert into a ToolResult.
    result = execute_tool("read_file", {"not_a_real_arg": "value"})
    assert result.ok is False
    assert "invalid input" in result.content.lower()
    assert result.metadata is not None
    assert result.metadata["tool"] == "read_file"


def test_execute_tool_missing_required_argument_returns_error() -> None:
    """Omitting a required argument is also invalid input, not a crash."""
    result = execute_tool("read_file", {})
    assert result.ok is False
    assert "invalid input" in result.content.lower()
