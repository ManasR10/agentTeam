from __future__ import annotations

import os
import tempfile
from pathlib import Path

from config import Settings, get_settings
from tools.file_tools import calculate_sha256, is_probably_text_file
from tools.mutation_safety import build_mutation_policy, is_protected_mutation_path
from tools.safety import PathSafetyError, resolve_inside_workspace
from tools.schemas import ToolResult


def _relative_to_workspace(resolved: Path, settings: Settings) -> Path:
    """Workspace-relative view of a resolved path, for protected-path checks."""
    return resolved.relative_to(settings.tool_workspace_root.resolve())


def _atomic_write(resolved: Path, content: str) -> None:
    """
    Write `content` to `resolved` atomically.

    A unique temp file is created in the SAME directory (so os.replace is a
    same-filesystem atomic rename) and swapped in. If anything fails, the temp
    file is removed and the original path is left untouched.
    """
    directory = resolved.parent
    fd, tmp_name = tempfile.mkstemp(
        dir=directory,
        prefix=f".{resolved.name}.",
        suffix=".devagent.tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
        os.replace(tmp_name, resolved)
    except BaseException:
        # The rename did not complete; ensure no partial temp file lingers.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _refuse(content: str, path: str, **extra: object) -> ToolResult:
    return ToolResult(ok=False, content=content, metadata={"path": path, **extra})


def _validate_writable_path(
    path: str, settings: Settings
) -> tuple[Path, Path] | ToolResult:
    """
    Resolve and screen a path for any write.

    Returns (resolved, relative) on success, or a refusing ToolResult. This
    handles workspace confinement, symlink/parent escape (via .resolve()), and
    the protected-path block list shared by all three write tools.
    """
    resolved = resolve_inside_workspace(
        path, workspace_root=settings.tool_workspace_root
    )
    relative = _relative_to_workspace(resolved, settings)
    if is_protected_mutation_path(relative):
        return _refuse(f"Modifying this path is not allowed: {path}", path)
    if not is_probably_text_file(resolved):
        return _refuse(f"Only text files may be written: {path}", path)
    return resolved, relative


def _too_large(content: str, settings: Settings) -> bool:
    return len(content) > settings.max_file_write_chars


def create_file(path: str, content: str) -> ToolResult:
    """
    Create a NEW text file inside the workspace. Never overwrites.

    The parent directory must already exist. Protected paths and non-text
    extensions are refused. The write is atomic.
    """
    settings = get_settings()
    policy = build_mutation_policy(settings)
    try:
        validated = _validate_writable_path(path, settings)
        if isinstance(validated, ToolResult):
            return validated
        resolved, relative = validated

        if not policy.allow_create_files:
            return _refuse("File creation is disabled by policy.", path)
        if resolved.exists():
            return _refuse(f"File already exists: {path}", path)
        if not resolved.parent.exists():
            return _refuse(f"Parent directory does not exist: {path}", path)
        if _too_large(content, settings):
            return _refuse(
                f"Content exceeds max_file_write_chars "
                f"({settings.max_file_write_chars}).",
                path,
            )

        _atomic_write(resolved, content)
        return ToolResult(
            ok=True,
            content=f"Created {relative} ({len(content)} chars).",
            metadata={
                "path": str(relative),
                "operation": "create",
                "before_sha256": None,
                "after_sha256": calculate_sha256(content),
                "chars_before": 0,
                "chars_after": len(content),
            },
        )
    except PathSafetyError as exc:
        return _refuse(str(exc), path)
    except OSError as exc:
        return _refuse(f"Could not create file {path!r}: {exc}", path)


def replace_in_file(
    path: str,
    old_text: str,
    new_text: str,
    expected_replacements: int = 1,
) -> ToolResult:
    """
    Replace an exact substring in an existing text file.

    Refuses unless the number of occurrences of `old_text` equals
    `expected_replacements` — optimistic concurrency that prevents accidentally
    editing the wrong (or stale) version of a file. The write is atomic.
    """
    settings = get_settings()
    policy = build_mutation_policy(settings)
    try:
        if not old_text:
            return _refuse("old_text cannot be empty.", path)
        if expected_replacements <= 0:
            return _refuse("expected_replacements must be positive.", path)

        validated = _validate_writable_path(path, settings)
        if isinstance(validated, ToolResult):
            return validated
        resolved, relative = validated

        if not policy.allow_overwrite_files:
            return _refuse("Modifying existing files is disabled by policy.", path)
        if not resolved.exists() or not resolved.is_file():
            return _refuse(f"File does not exist: {path}", path)

        try:
            current = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _refuse(f"File is not valid UTF-8 text: {path}", path)

        count = current.count(old_text)
        if count != expected_replacements:
            return _refuse(
                f"Expected {expected_replacements} occurrence(s) of old_text "
                f"but found {count}. File left unchanged.",
                path,
                found=count,
            )

        new_content = current.replace(old_text, new_text)
        if _too_large(new_content, settings):
            return _refuse(
                f"Result exceeds max_file_write_chars "
                f"({settings.max_file_write_chars}). File left unchanged.",
                path,
            )

        _atomic_write(resolved, new_content)
        return ToolResult(
            ok=True,
            content=f"Replaced {count} occurrence(s) in {relative}.",
            metadata={
                "path": str(relative),
                "operation": "replace",
                "replacements": count,
                "before_sha256": calculate_sha256(current),
                "after_sha256": calculate_sha256(new_content),
                "chars_before": len(current),
                "chars_after": len(new_content),
            },
        )
    except PathSafetyError as exc:
        return _refuse(str(exc), path)
    except OSError as exc:
        return _refuse(f"Could not edit file {path!r}: {exc}", path)


def write_file(path: str, content: str, expected_sha256: str) -> ToolResult:
    """
    Fully rewrite an existing text file, guarded by an expected hash.

    The caller must supply the SHA-256 the file had when it was last read. If
    the file changed since (hashes differ), the stale write is rejected. Large
    files that a read would have truncated are refused outright — a full rewrite
    from a truncated read would silently drop the unseen tail; use
    replace_in_file for those. The write is atomic.
    """
    settings = get_settings()
    policy = build_mutation_policy(settings)
    try:
        if not expected_sha256.strip():
            return _refuse("expected_sha256 is required for write_file.", path)

        validated = _validate_writable_path(path, settings)
        if isinstance(validated, ToolResult):
            return validated
        resolved, relative = validated

        if not policy.allow_overwrite_files:
            return _refuse("Overwriting files is disabled by policy.", path)
        if not resolved.exists() or not resolved.is_file():
            return _refuse(f"File does not exist: {path}", path)

        try:
            current = resolved.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return _refuse(f"File is not valid UTF-8 text: {path}", path)

        if len(current) > settings.max_file_read_chars:
            return _refuse(
                f"File is too large to rewrite safely "
                f"(> max_file_read_chars={settings.max_file_read_chars}); a "
                f"full read would have been truncated. Use replace_in_file.",
                path,
            )

        current_sha = calculate_sha256(current)
        if current_sha != expected_sha256.strip():
            return _refuse(
                "Stale write rejected: file changed since it was read "
                f"(expected {expected_sha256.strip()[:12]}…, "
                f"found {current_sha[:12]}…). Re-read and retry.",
                path,
                expected_sha256=expected_sha256.strip(),
                actual_sha256=current_sha,
            )

        if _too_large(content, settings):
            return _refuse(
                f"Content exceeds max_file_write_chars "
                f"({settings.max_file_write_chars}).",
                path,
            )

        _atomic_write(resolved, content)
        return ToolResult(
            ok=True,
            content=f"Rewrote {relative} ({len(content)} chars).",
            metadata={
                "path": str(relative),
                "operation": "rewrite",
                "before_sha256": current_sha,
                "after_sha256": calculate_sha256(content),
                "chars_before": len(current),
                "chars_after": len(content),
            },
        )
    except PathSafetyError as exc:
        return _refuse(str(exc), path)
    except OSError as exc:
        return _refuse(f"Could not write file {path!r}: {exc}", path)
