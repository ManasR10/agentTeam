from __future__ import annotations

from pathlib import Path

from config import settings
from tools.safety import PathSafetyError, resolve_inside_workspace
from tools.schemas import ToolResult

# Directories that are noisy, large, or sensitive. They are skipped when
# listing files and never traversed for the model.
IGNORED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
}

# Files that must never be read by a tool, even if they are text.
IGNORED_FILE_NAMES = {
    ".env",
    ".env.local",
    ".DS_Store",
}

# Extensions we are confident are safe, readable text.
TEXT_FILE_EXTENSIONS = {
    ".py",
    ".txt",
    ".md",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".ini",
    ".cfg",
    ".env.example",
    ".gitignore",
}


def is_probably_text_file(path: Path) -> bool:
    """
    Conservative text-file check.

    This avoids reading images, binaries, databases, archives, etc.
    A few dotfiles (no real suffix) are allow-listed by name.
    """
    if path.name in {".gitignore", ".env.example"}:
        return True
    return path.suffix.lower() in TEXT_FILE_EXTENSIONS


def read_file(path: str) -> ToolResult:
    """
    Safely read a text file inside the configured workspace.

    Returns a ToolResult instead of raising, so the tool loop can hand any
    failure back to the model as a normal (non-fatal) observation.
    """
    try:
        resolved = resolve_inside_workspace(
            path,
            workspace_root=settings.tool_workspace_root,
        )

        if not resolved.exists():
            return ToolResult(
                ok=False,
                content=f"File does not exist: {path}",
                metadata={"path": path},
            )
        if not resolved.is_file():
            return ToolResult(
                ok=False,
                content=f"Path is not a file: {path}",
                metadata={"path": path},
            )
        if resolved.name in IGNORED_FILE_NAMES:
            return ToolResult(
                ok=False,
                content=f"Access to this file is blocked: {path}",
                metadata={"path": path},
            )
        if not is_probably_text_file(resolved):
            return ToolResult(
                ok=False,
                content=f"File type is not allowed for reading: {path}",
                metadata={"path": path},
            )

        raw_text = resolved.read_text(encoding="utf-8", errors="replace")

        # Truncate very large files so they cannot flood the model context.
        truncated = False
        text = raw_text
        if len(raw_text) > settings.max_file_read_chars:
            text = raw_text[: settings.max_file_read_chars]
            truncated = True

        return ToolResult(
            ok=True,
            content=text,
            metadata={
                "path": str(
                    resolved.relative_to(settings.tool_workspace_root)
                ),
                "chars": len(raw_text),
                "returned_chars": len(text),
                "truncated": truncated,
            },
        )

    except PathSafetyError as exc:
        return ToolResult(
            ok=False,
            content=str(exc),
            metadata={"path": path},
        )
    except OSError as exc:
        return ToolResult(
            ok=False,
            content=f"Could not read file {path!r}: {exc}",
            metadata={"path": path},
        )


def should_ignore_path(path: Path) -> bool:
    """Return True if a relative path is in an ignored directory or file."""
    parts = set(path.parts)
    if parts.intersection(IGNORED_DIR_NAMES):
        return True
    if path.name in IGNORED_FILE_NAMES:
        return True
    return False


def list_files(path: str = ".") -> ToolResult:
    """
    List files under a directory inside the configured workspace.

    Returns relative paths and skips common noisy/sensitive folders.
    """
    try:
        resolved = resolve_inside_workspace(
            path,
            workspace_root=settings.tool_workspace_root,
        )

        if not resolved.exists():
            return ToolResult(
                ok=False,
                content=f"Directory does not exist: {path}",
                metadata={"path": path},
            )
        if not resolved.is_dir():
            return ToolResult(
                ok=False,
                content=f"Path is not a directory: {path}",
                metadata={"path": path},
            )

        files: list[str] = []
        for child in sorted(resolved.rglob("*")):
            relative = child.relative_to(settings.tool_workspace_root)
            if should_ignore_path(relative):
                continue
            if child.is_file():
                files.append(str(relative))

        if not files:
            return ToolResult(
                ok=True,
                content="No files found.",
                metadata={"count": 0},
            )

        return ToolResult(
            ok=True,
            content="\n".join(files),
            metadata={"count": len(files)},
        )

    except PathSafetyError as exc:
        return ToolResult(
            ok=False,
            content=str(exc),
            metadata={"path": path},
        )
    except OSError as exc:
        return ToolResult(
            ok=False,
            content=f"Could not list files for {path!r}: {exc}",
            metadata={"path": path},
        )
