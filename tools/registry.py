from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass
from typing import Any, Literal

from tools.command_tools import run_check, run_tests
from tools.file_tools import list_files, read_file
from tools.git_tools import git_diff, git_status
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


_RUN_TESTS = RegisteredTool(
    name="run_tests",
    description=(
        "Run pytest on specific test paths under tests/. The executable and "
        "command are fixed by Python; you only choose which test files to run "
        "and an allow-listed set of flags (-q, -x, -v)."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Test paths under tests/ (e.g. tests/test_x.py).",
            },
            "extra_args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional pytest flags from {-q, -x, -v}.",
            },
        },
        "required": [],
        "additionalProperties": False,
    },
    function=run_tests,
    category="command",
)

_RUN_CHECK = RegisteredTool(
    name="run_check",
    description=(
        "Run a single named project check. Allowed checks: 'pytest' (full "
        "suite) and 'py_compile' (compile project sources). No other commands "
        "are available."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "check": {
                "type": "string",
                "enum": ["pytest", "py_compile"],
                "description": "Which named check to run.",
            }
        },
        "required": ["check"],
        "additionalProperties": False,
    },
    function=run_check,
    category="command",
)

_GIT_STATUS = RegisteredTool(
    name="git_status",
    description=(
        "Show `git status --short` for the workspace. Read-only: it never "
        "stages, commits, or modifies the repository."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    function=git_status,
    category="git",
)

_GIT_DIFF = RegisteredTool(
    name="git_diff",
    description=(
        "Show the unstaged `git diff` for the workspace, with a summary of "
        "files changed and insertion/deletion counts. Read-only."
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    },
    function=git_diff,
    category="git",
)


# The single source of truth for every tool the agent layer may use.
TOOL_REGISTRY: dict[str, RegisteredTool] = {
    _READ_FILE.name: _READ_FILE,
    _LIST_FILES.name: _LIST_FILES,
    _CREATE_FILE.name: _CREATE_FILE,
    _REPLACE_IN_FILE.name: _REPLACE_IN_FILE,
    _WRITE_FILE.name: _WRITE_FILE,
    _RUN_TESTS.name: _RUN_TESTS,
    _RUN_CHECK.name: _RUN_CHECK,
    _GIT_STATUS.name: _GIT_STATUS,
    _GIT_DIFF.name: _GIT_DIFF,
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
# The secure default for any caller that does not pick a profile. Read-only.
READ_ONLY_TOOL_NAMES: frozenset[str] = frozenset({"read_file", "list_files"})

PLANNER_TOOL_NAMES: frozenset[str] = frozenset({"read_file", "list_files"})

# The coder may inspect, mutate, run tests/checks, and inspect git.
CODER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "read_file",
        "list_files",
        "create_file",
        "replace_in_file",
        "write_file",
        "run_tests",
        "run_check",
        "git_diff",
        "git_status",
    }
)

# The reviewer is read-only: inspect files and git, but never mutate.
REVIEWER_TOOL_NAMES: frozenset[str] = frozenset(
    {"read_file", "list_files", "git_diff", "git_status"}
)


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
