from __future__ import annotations

from config import Settings, get_settings
from tools.command_tools import run_process
from tools.results import CommandResult
from tools.schemas import ToolResult

# Git is run by Python with explicit argv (shell=False). The agent never runs
# git itself. Phase 3 only inspects: status and unstaged diff. Nothing here
# stages, commits, or otherwise modifies repository state.


def _run_git(args: list[str], settings: Settings) -> CommandResult:
    return run_process("git", ["git", *args], settings)


def git_toplevel(settings: Settings | None = None) -> str | None:
    """Return the git repo root for the workspace, or None if not a repo."""
    settings = settings or get_settings()
    result = _run_git(["rev-parse", "--show-toplevel"], settings)
    if result.timed_out or result.exit_code != 0:
        return None
    return result.stdout.strip()


def is_workspace_git_root(settings: Settings | None = None) -> bool:
    """
    True only when the workspace IS the git root (not a subdirectory of one).

    This guards against operating on a parent repository that merely contains
    the workspace — exactly the case where running from a parent dir would mix
    in unrelated changes.
    """
    settings = settings or get_settings()
    top = git_toplevel(settings)
    if top is None:
        return False
    from pathlib import Path

    return Path(top).resolve() == settings.tool_workspace_root.resolve()


def _ensure_git_root(settings: Settings) -> ToolResult | None:
    if git_toplevel(settings) is None:
        return ToolResult(
            ok=False,
            content="Workspace is not a git repository.",
            metadata={"git": False},
        )
    if not is_workspace_git_root(settings):
        return ToolResult(
            ok=False,
            content=(
                "Workspace is not the git root; refusing to inspect a parent "
                "repository."
            ),
            metadata={"git": True, "is_root": False},
        )
    return None


def git_status() -> ToolResult:
    """Return `git status --short` for the workspace repository."""
    settings = get_settings()
    guard = _ensure_git_root(settings)
    if guard is not None:
        return guard

    result = _run_git(["status", "--short"], settings)
    if result.exit_code != 0:
        return ToolResult(
            ok=False,
            content=f"git status failed: {result.stderr}",
            metadata={"exit_code": result.exit_code},
        )

    changed = [
        line[3:] for line in result.stdout.splitlines() if line.strip()
    ]
    return ToolResult(
        ok=True,
        content=result.stdout or "(clean)",
        metadata={"changed_paths": changed, "clean": not changed},
    )


def git_diff() -> ToolResult:
    """Return the unstaged `git diff` plus per-file numstat metadata."""
    settings = get_settings()
    guard = _ensure_git_root(settings)
    if guard is not None:
        return guard

    diff = _run_git(["diff", "--no-ext-diff", "--unified=3"], settings)
    if diff.exit_code != 0:
        return ToolResult(
            ok=False,
            content=f"git diff failed: {diff.stderr}",
            metadata={"exit_code": diff.exit_code},
        )

    numstat = _run_git(["diff", "--no-ext-diff", "--numstat"], settings)
    files_changed: list[str] = []
    insertions = 0
    deletions = 0
    if numstat.exit_code == 0:
        for line in numstat.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            added, removed, name = parts
            files_changed.append(name)
            if added.isdigit():
                insertions += int(added)
            if removed.isdigit():
                deletions += int(removed)

    return ToolResult(
        ok=True,
        content=diff.stdout or "(no changes)",
        metadata={
            "files_changed": files_changed,
            "insertions": insertions,
            "deletions": deletions,
            "truncated": diff.output_truncated,
        },
    )


def get_diff_text(settings: Settings | None = None) -> str:
    """Orchestrator-facing: raw unstaged diff text (empty string if none)."""
    settings = settings or get_settings()
    result = _run_git(["diff", "--no-ext-diff", "--unified=3"], settings)
    if result.exit_code != 0:
        return ""
    return result.stdout
