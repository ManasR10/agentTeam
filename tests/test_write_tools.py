from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from config import Settings, get_settings
from tools import write_tools
from tools.file_tools import calculate_sha256
from tools.write_tools import create_file, replace_in_file, write_file


@pytest.fixture()
def workspace(monkeypatch, tmp_path: Path) -> Path:
    """Point all write tools at an isolated tmp workspace."""
    settings = replace(get_settings(), tool_workspace_root=tmp_path)
    monkeypatch.setattr(write_tools, "get_settings", lambda: settings)
    return tmp_path


def _settings_for(tmp_path: Path, **overrides) -> Settings:
    return replace(get_settings(), tool_workspace_root=tmp_path, **overrides)


# --- create_file -----------------------------------------------------------


def test_create_file_creates_text_file(workspace: Path) -> None:
    result = create_file("notes.md", "hello")
    assert result.ok is True
    assert (workspace / "notes.md").read_text() == "hello"
    assert result.metadata is not None
    assert result.metadata["operation"] == "create"
    assert result.metadata["after_sha256"] == calculate_sha256("hello")
    assert result.metadata["chars_before"] == 0


def test_create_file_refuses_existing(workspace: Path) -> None:
    (workspace / "a.py").write_text("x")
    result = create_file("a.py", "y")
    assert result.ok is False
    assert "already exists" in result.content.lower()
    assert (workspace / "a.py").read_text() == "x"


def test_create_file_blocks_outside_workspace(workspace: Path) -> None:
    result = create_file("../escape.py", "x")
    assert result.ok is False
    assert "escapes workspace" in result.content.lower()


def test_create_file_blocks_env(workspace: Path) -> None:
    for name in [".env", ".env.local", ".env.example"]:
        result = create_file(name, "SECRET=1")
        assert result.ok is False, name
        assert "not allowed" in result.content.lower()


def test_create_file_blocks_binary_extension(workspace: Path) -> None:
    result = create_file("image.png", "x")
    assert result.ok is False
    assert "text files" in result.content.lower()


def test_create_file_enforces_size(monkeypatch, tmp_path: Path) -> None:
    settings = _settings_for(tmp_path, max_file_write_chars=5)
    monkeypatch.setattr(write_tools, "get_settings", lambda: settings)
    result = create_file("big.txt", "way too long")
    assert result.ok is False
    assert "exceeds" in result.content.lower()
    assert not (tmp_path / "big.txt").exists()


def test_create_file_missing_parent(workspace: Path) -> None:
    result = create_file("nope/child.py", "x")
    assert result.ok is False
    assert "parent directory" in result.content.lower()


# --- replace_in_file -------------------------------------------------------


def test_replace_in_file_replaces_once(workspace: Path) -> None:
    (workspace / "m.py").write_text("a = 1\nb = 2\n")
    result = replace_in_file("m.py", "b = 2", "b = 3")
    assert result.ok is True
    assert (workspace / "m.py").read_text() == "a = 1\nb = 3\n"
    assert result.metadata is not None
    assert result.metadata["replacements"] == 1
    assert result.metadata["before_sha256"] == calculate_sha256("a = 1\nb = 2\n")
    assert result.metadata["after_sha256"] == calculate_sha256("a = 1\nb = 3\n")


def test_replace_in_file_refuses_zero_matches(workspace: Path) -> None:
    (workspace / "m.py").write_text("a = 1\n")
    result = replace_in_file("m.py", "missing", "x")
    assert result.ok is False
    assert "found 0" in result.content.lower()
    assert (workspace / "m.py").read_text() == "a = 1\n"


def test_replace_in_file_refuses_unexpected_count(workspace: Path) -> None:
    (workspace / "m.py").write_text("x\nx\nx\n")
    result = replace_in_file("m.py", "x", "y", expected_replacements=1)
    assert result.ok is False
    assert "found 3" in result.content.lower()
    assert (workspace / "m.py").read_text() == "x\nx\nx\n"


def test_replace_in_file_multiple_expected(workspace: Path) -> None:
    (workspace / "m.py").write_text("x\nx\nx\n")
    result = replace_in_file("m.py", "x", "y", expected_replacements=3)
    assert result.ok is True
    assert (workspace / "m.py").read_text() == "y\ny\ny\n"


def test_replace_in_file_blocks_protected(workspace: Path) -> None:
    result = replace_in_file(".env", "a", "b")
    assert result.ok is False
    assert "not allowed" in result.content.lower()


def test_replace_in_file_missing_file(workspace: Path) -> None:
    result = replace_in_file("ghost.py", "a", "b")
    assert result.ok is False
    assert "does not exist" in result.content.lower()


# --- write_file ------------------------------------------------------------


def test_write_file_succeeds_with_matching_hash(workspace: Path) -> None:
    (workspace / "f.py").write_text("old")
    result = write_file("f.py", "new", expected_sha256=calculate_sha256("old"))
    assert result.ok is True
    assert (workspace / "f.py").read_text() == "new"
    assert result.metadata is not None
    assert result.metadata["operation"] == "rewrite"
    assert result.metadata["before_sha256"] == calculate_sha256("old")


def test_write_file_rejects_stale_hash(workspace: Path) -> None:
    (workspace / "f.py").write_text("current")
    result = write_file("f.py", "new", expected_sha256=calculate_sha256("stale"))
    assert result.ok is False
    assert "stale write" in result.content.lower()
    assert (workspace / "f.py").read_text() == "current"


def test_write_file_refuses_large_file(monkeypatch, tmp_path: Path) -> None:
    """A file bigger than a read would return must go through replace_in_file."""
    settings = _settings_for(tmp_path, max_file_read_chars=5)
    monkeypatch.setattr(write_tools, "get_settings", lambda: settings)
    big = "abcdefghij"  # 10 chars > 5
    (tmp_path / "f.py").write_text(big)
    result = write_file("f.py", "x", expected_sha256=calculate_sha256(big))
    assert result.ok is False
    assert "replace_in_file" in result.content.lower()
    assert (tmp_path / "f.py").read_text() == big


def test_write_file_requires_hash(workspace: Path) -> None:
    (workspace / "f.py").write_text("old")
    result = write_file("f.py", "new", expected_sha256="")
    assert result.ok is False
    assert "required" in result.content.lower()


# --- atomicity -------------------------------------------------------------


def test_failed_write_leaves_no_temp_and_keeps_original(
    monkeypatch, workspace: Path
) -> None:
    (workspace / "f.py").write_text("original")

    def boom(_src, _dst):
        raise OSError("disk full")

    monkeypatch.setattr(write_tools.os, "replace", boom)
    result = write_file("f.py", "new", expected_sha256=calculate_sha256("original"))
    assert result.ok is False
    # Original survives, and no .devagent.tmp residue remains.
    assert (workspace / "f.py").read_text() == "original"
    leftovers = list(workspace.glob("*.devagent.tmp"))
    assert leftovers == []


def test_successful_write_leaves_no_temp(workspace: Path) -> None:
    create_file("f.py", "data")
    assert list(workspace.glob("*.devagent.tmp")) == []


def test_reviewer_profile_excludes_write_tools() -> None:
    from tools.registry import REVIEWER_TOOL_NAMES

    assert "write_file" not in REVIEWER_TOOL_NAMES
    assert "create_file" not in REVIEWER_TOOL_NAMES
    assert "replace_in_file" not in REVIEWER_TOOL_NAMES


def test_coder_profile_includes_write_tools() -> None:
    from tools.registry import CODER_TOOL_NAMES

    assert {"create_file", "replace_in_file", "write_file"} <= CODER_TOOL_NAMES
