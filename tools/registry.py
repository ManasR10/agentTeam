from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Literal

from tools.file_tools import list_files, read_file
from tools.schemas import ToolResult
from tools.write_tools import create_file, replace_in_file, write_file

ToolFunction = Callable[..., ToolResult]

ToolCategory = Literal["read", "write", "command", "git"]


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """
    A tool's schema, implementation, and capability category in one record.

    Bundling the Anthropic schema with the Python function and a category lets
    the registry hand each agent exactly the capabilities its role needs, and
    lets dispatch enforce that allow-list independently of what was advertised.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    function: ToolFunction
    category: ToolCategory


_READ_FILE = RegisteredTool(
    name="read_file",
    description=(
        "Read a text file from the project workspace. "
        "Use this when you need to inspect source code, README files, "
        "configuration files, or tests. Do not use it for binary files."
    ),
    input_schema={
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
    function=read_file,
    category="read",
)

_LIST_FILES = RegisteredTool(
    name="list_files",
    description=(
        "List files under a directory in the project workspace. "
        "Use this before reading files if you do not know the structure."
    ),
    input_schema={
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
    function=list_files,
    category="read",
)


_CREATE_FILE = RegisteredTool(
    name="create_file",
    description=(
        "Create a NEW text file inside the workspace. Fails if the file "
        "already exists or its parent directory is missing. Use this only for "
        "genuinely new files; use replace_in_file to edit existing ones."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the new file in the workspace.",
            },
            "content": {
                "type": "string",
                "description": "Full text content of the new file.",
            },
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    },
    function=create_file,
    category="write",
)

_REPLACE_IN_FILE = RegisteredTool(
    name="replace_in_file",
    description=(
        "Replace an exact substring in an existing text file. The edit is "
        "refused unless old_text occurs exactly expected_replacements times, so "
        "it cannot accidentally change the wrong or a stale version. Prefer "
        "this over write_file for most edits."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the file to edit.",
            },
            "old_text": {
                "type": "string",
                "description": "Exact existing text to replace.",
            },
            "new_text": {
                "type": "string",
                "description": "Replacement text.",
            },
            "expected_replacements": {
                "type": "integer",
                "description": "How many occurrences old_text must match.",
                "default": 1,
            },
        },
        "required": ["path", "old_text", "new_text"],
        "additionalProperties": False,
    },
    function=replace_in_file,
    category="write",
)

_WRITE_FILE = RegisteredTool(
    name="write_file",
    description=(
        "Fully rewrite an existing text file. Requires expected_sha256 from the "
        "last read of the file; the write is rejected if the file changed since "
        "(stale write) or is too large to have been read whole. Use only when a "
        "complete rewrite is justified — otherwise prefer replace_in_file."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path of the file to rewrite.",
            },
            "content": {
                "type": "string",
                "description": "Full new text content of the file.",
            },
            "expected_sha256": {
                "type": "string",
                "description": "SHA-256 the file had when last read.",
            },
        },
        "required": ["path", "content", "expected_sha256"],
        "additionalProperties": False,
    },
    function=write_file,
    category="write",
)


# The single source of truth for every tool the agent layer may use. New
# Phase 3 tools (command/git) register themselves here.
TOOL_REGISTRY: dict[str, RegisteredTool] = {
    _READ_FILE.name: _READ_FILE,
    _LIST_FILES.name: _LIST_FILES,
    _CREATE_FILE.name: _CREATE_FILE,
    _REPLACE_IN_FILE.name: _REPLACE_IN_FILE,
    _WRITE_FILE.name: _WRITE_FILE,
}


def register_tool(tool: RegisteredTool) -> None:
    """Register a tool, rejecting duplicate names so collisions fail loudly."""
    if tool.name in TOOL_REGISTRY:
        raise ValueError(f"Tool already registered: {tool.name}")
    TOOL_REGISTRY[tool.name] = tool


# Capability profiles. Each agent role gets only the names it needs; this is
# capability-based security, not just prompt instructions. Profiles are
# validated against the registry at call time, so a name here that is not yet
# registered surfaces immediately rather than silently doing nothing.
PLANNER_TOOL_NAMES: frozenset[str] = frozenset({"read_file", "list_files"})

# The coder may inspect and mutate. Command/git capabilities are added in P3.4.
CODER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_file",
        "list_files",
        "create_file",
        "replace_in_file",
        "write_file",
    }
)

# The reviewer is read-only. Git inspection tools are added in P3.4. It must
# never receive any write capability.
REVIEWER_TOOL_NAMES: frozenset[str] = frozenset({"read_file", "list_files"})


def get_tool_definitions(names: Collection[str]) -> list[dict[str, Any]]:
    """
    Return the Anthropic tool schemas for `names`, in a stable sorted order.

    Raises:
        ValueError: If a requested name is not registered, so a typo in a
            capability profile fails fast instead of silently dropping a tool.
    """
    definitions: list[dict[str, Any]] = []
    for name in sorted(names):
        tool = TOOL_REGISTRY.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool requested in profile: {name}")
        definitions.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
        )
    return definitions


# Backwards-compatible export: the full set of definitions, as the Phase 1/2
# loop used to receive them. Phase 3 callers should prefer get_tool_definitions
# with a capability profile instead.
ANTHROPIC_TOOLS: list[dict[str, Any]] = get_tool_definitions(TOOL_REGISTRY.keys())


def execute_tool(
    name: str,
    tool_input: dict[str, Any],
    *,
    allowed_tools: Collection[str] | None = None,
) -> ToolResult:
    """
    Execute a registered tool by name, enforcing a capability allow-list.

    The model only *requests* a tool. This function decides whether the tool
    exists, whether this agent run is permitted to use it, and how to call it,
    turning every failure mode into a safe ToolResult instead of an exception.

    `allowed_tools` is defence in depth: even if a tool somehow appears in a
    request without being advertised to the model, it is denied unless it is in
    this run's profile. None means "all registered tools are permitted" (the
    Phase 1/2 behaviour).
    """
    if allowed_tools is not None and name not in allowed_tools:
        return ToolResult(
            ok=False,
            content=(
                f"Tool {name!r} is not permitted for this agent run."
            ),
            metadata={"tool": name, "denied": True},
        )

    tool = TOOL_REGISTRY.get(name)
    if tool is None:
        return ToolResult(
            ok=False,
            content=f"Unknown tool requested: {name}",
            metadata={"tool": name},
        )
    try:
        return tool.function(**tool_input)
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


def make_tool_executor(
    allowed_tools: Collection[str],
) -> Callable[[str, dict[str, Any]], ToolResult]:
    """
    Build an executor closed over a capability profile.

    The tool loop calls the returned function as `executor(name, input)` and
    never sees the allow-list, so the permission decision lives in exactly one
    place.
    """
    allowed = frozenset(allowed_tools)

    def _executor(name: str, tool_input: dict[str, Any]) -> ToolResult:
        return execute_tool(name, tool_input, allowed_tools=allowed)

    return _executor
