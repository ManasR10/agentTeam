from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from config import get_settings
from tools import git_tools
from tools.git_tools import git_diff, git_status, is_workspace_git_root


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture()
def repo(monkeypatch, tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-m", "init")
    settings = replace(get_settings(), tool_workspace_root=tmp_path)
    monkeypatch.setattr(git_tools, "get_settings", lambda: settings)
    return tmp_path


def test_git_status_clean(repo: Path) -> None:
    result = git_status()
    assert result.ok is True
    assert result.metadata is not None
    assert result.metadata["clean"] is True


def test_git_status_parses_changed_paths(repo: Path) -> None:
    (repo / "a.py").write_text("x = 2\n")
    (repo / "new.py").write_text("y = 3\n")
    result = git_status()
    assert result.ok is True
    assert result.metadata is not None
    changed = set(result.metadata["changed_paths"])
    assert "a.py" in changed
    assert "new.py" in changed


def test_git_diff_returns_text_and_stats(repo: Path) -> None:
    (repo / "a.py").write_text("x = 1\nx = 99\n")
    result = git_diff()
    assert result.ok is True
    assert "x = 99" in result.content
    assert result.metadata is not None
    assert "a.py" in result.metadata["files_changed"]
    assert result.metadata["insertions"] >= 1


def test_git_tools_handle_non_git_dir(monkeypatch, tmp_path: Path) -> None:
    settings = replace(get_settings(), tool_workspace_root=tmp_path)
    monkeypatch.setattr(git_tools, "get_settings", lambda: settings)
    status = git_status()
    diff = git_diff()
    assert status.ok is False
    assert diff.ok is False
    assert "not a git repository" in status.content.lower()


def test_refuses_when_workspace_is_subdir_of_repo(
    monkeypatch, repo: Path
) -> None:
    """If the workspace is a child of the git root, inspection is refused."""
    sub = repo / "subdir"
    sub.mkdir()
    settings = replace(get_settings(), tool_workspace_root=sub)
    monkeypatch.setattr(git_tools, "get_settings", lambda: settings)
    assert is_workspace_git_root(settings) is False
    result = git_status()
    assert result.ok is False
    assert "not the git root" in result.content.lower()
