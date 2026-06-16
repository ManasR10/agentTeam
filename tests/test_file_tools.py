from __future__ import annotations

from tools.file_tools import read_file


def test_read_file_blocks_env_file() -> None:
    result = read_file(".env")
    assert result.ok is False
    assert "blocked" in result.content.lower()


def test_read_file_missing_file_returns_error() -> None:
    result = read_file("this-file-should-not-exist-12345.txt")
    assert result.ok is False
    assert "does not exist" in result.content.lower()


def test_read_file_can_read_readme() -> None:
    result = read_file("README.md")
    assert result.ok is True
    assert len(result.content) > 0
