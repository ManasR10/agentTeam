from __future__ import annotations

from pathlib import Path


class PathSafetyError(ValueError):
    """Raised when a requested path is outside the allowed workspace."""


def resolve_inside_workspace(
    requested_path: str,
    *,
    workspace_root: Path,
) -> Path:
    """
    Resolve a model-requested path and ensure it stays inside workspace_root.

    This prevents path traversal attacks such as:
        ../../.env
        /etc/passwd
        ~/.ssh/id_rsa

    Args:
        requested_path:
            Path requested by the model/tool caller.
        workspace_root:
            Root directory tools are allowed to access.

    Returns:
        Absolute resolved Path inside workspace_root.

    Raises:
        PathSafetyError:
            If the path is empty or escapes the workspace.
    """
    if not requested_path or not requested_path.strip():
        raise PathSafetyError("Path cannot be empty.")

    workspace = workspace_root.resolve()
    raw_path = Path(requested_path.strip()).expanduser()

    if raw_path.is_absolute():
        candidate = raw_path
    else:
        candidate = workspace / raw_path

    resolved = candidate.resolve()

    # relative_to raises ValueError when `resolved` is not under `workspace`,
    # which is exactly how we detect an escape attempt.
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise PathSafetyError(
            f"Path escapes workspace: {requested_path!r}"
        ) from exc

    return resolved
