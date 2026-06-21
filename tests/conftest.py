from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from config import Settings, get_settings


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def init_git_repo(repo: Path) -> str:
    """Create a clean git repo with one committed file; return the workspace path.

    Used by workflow/service and integration tests that need real git baseline
    behaviour (HEAD, clean worktree, diff) without touching the network.
    """
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "todo.py").write_text(
        "class TodoList:\n"
        "    def __init__(self):\n"
        "        self._items = []\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "baseline")
    return str(repo)


@pytest.fixture()
def git_repo(tmp_path) -> Path:
    repo = tmp_path / "sample_repo"
    init_git_repo(repo)
    return repo


@pytest.fixture()
def repo_settings(git_repo) -> Settings:
    """Settings pointed at a temp git repo, with storage under it."""
    return replace(
        get_settings(),
        tool_workspace_root=git_repo,
        devagent_data_dir=git_repo / ".devagent",
        devagent_database_path=git_repo / ".devagent" / "devagent.db",
    )
