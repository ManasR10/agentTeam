from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tools.file_tools import list_files, read_file
from tools.schemas import ToolResult

ToolFunction = Callable[..., ToolResult]

# Tool definitions exactly as Claude receives them. Each entry follows the
# Anthropic tool schema: name, description, and a JSON Schema input_schema.
ANTHROPIC_TOOLS: list[dict[str, Any]] = [
    {
        "name": "read_file",
        "description": (
            "Read a text file from the project workspace. "
            "Use this when you need to inspect source code, README files, "
            "configuration files, or tests. Do not use it for binary files."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path to the file inside the workspace. "
                        "Example: README.md or tools/file_tools.py"
                    ),
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_files",
        "description": (
            "List files under a directory in the project workspace. "
            "Use this before reading files if you do not know the structure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative directory path inside the workspace. "
                        "Defaults to the project root."
                    ),
                    "default": ".",
                }
            },
            "required": [],
            "additionalProperties": False,
        },
    },
]

# Maps a tool name to the Python function that actually performs the work.
TOOL_REGISTRY: dict[str, ToolFunction] = {
    "read_file": read_file,
    "list_files": list_files,
}


def execute_tool(name: str, tool_input: dict[str, Any]) -> ToolResult:
    """
    Execute a registered tool by name.

    The model only *requests* a tool. This function is where Python actually
    decides whether the tool exists and how to call it, turning every failure
    mode into a safe ToolResult instead of an exception.
    """
    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return ToolResult(
            ok=False,
            content=f"Unknown tool requested: {name}",
            metadata={"tool": name},
        )
    try:
        return tool(**tool_input)
    except TypeError as exc:
        # Wrong / missing arguments for the tool function.
        return ToolResult(
            ok=False,
            content=f"Invalid input for tool {name}: {exc}",
            metadata={"tool": name, "input": tool_input},
        )
    except Exception as exc:
        # Last-resort guard: a tool bug must never crash the whole loop.
        return ToolResult(
            ok=False,
            content=(
                f"Tool {name} failed unexpectedly: "
                f"{type(exc).__name__}: {exc}"
            ),
            metadata={"tool": name, "input": tool_input},
        )
