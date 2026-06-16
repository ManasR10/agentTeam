from __future__ import annotations

from pathlib import Path

import pytest

from tools.safety import PathSafetyError, resolve_inside_workspace


def test_allows_relative_path_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path
    file_path = workspace / "README.md"
    file_path.write_text("hello", encoding="utf-8")

    resolved = resolve_inside_workspace(
        "README.md",
        workspace_root=workspace,
    )
    assert resolved == file_path.resolve()


def test_blocks_parent_directory_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(PathSafetyError):
        resolve_inside_workspace(
            "../secret.txt",
            workspace_root=workspace,
        )


def test_blocks_absolute_path_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside.txt"
    workspace.mkdir()
    outside.write_text("secret", encoding="utf-8")

    with pytest.raises(PathSafetyError):
        resolve_inside_workspace(
            str(outside),
            workspace_root=workspace,
        )


def test_blocks_empty_path(tmp_path: Path) -> None:
    with pytest.raises(PathSafetyError):
        resolve_inside_workspace(
            "",
            workspace_root=tmp_path,
        )
