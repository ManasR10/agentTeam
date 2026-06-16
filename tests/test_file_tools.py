from __future__ import annotations

from pathlib import Path

from config import get_settings
from tools.file_tools import (
    is_blocked_file,
    is_probably_text_file,
    list_files,
    read_file,
)


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


def test_is_probably_text_file_classification() -> None:
    """Pure check (no I/O): text extensions pass, binary ones do not."""
    assert is_probably_text_file(Path("config.py")) is True
    assert is_probably_text_file(Path("README.md")) is True
    assert is_probably_text_file(Path(".gitignore")) is True
    assert is_probably_text_file(Path("image.png")) is False
    assert is_probably_text_file(Path("archive.zip")) is False
    assert is_probably_text_file(Path("app.exe")) is False


def test_is_blocked_file_env_variants() -> None:
    """Any .env* secret is blocked, except the safe committed .env.example."""
    assert is_blocked_file(".env") is True
    assert is_blocked_file(".env.local") is True
    assert is_blocked_file(".env.production") is True
    assert is_blocked_file(".DS_Store") is True
    assert is_blocked_file(".env.example") is False
    assert is_blocked_file("config.py") is False


def test_file_tools_do_not_require_api_key(monkeypatch) -> None:
    """
    Offline file tools must work with no ANTHROPIC_API_KEY set.

    read_file/list_files only need the workspace settings; they never call the
    API. This guards against re-coupling file tools to API-key validation.
    """
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    get_settings.cache_clear()
    try:
        read_result = read_file("README.md")
        assert read_result.ok is True
        assert len(read_result.content) > 0

        list_result = list_files(".")
        assert list_result.ok is True
        assert len(list_result.content) > 0
    finally:
        # Drop the key-less cached settings so later tests reload normally.
        get_settings.cache_clear()


def test_read_file_blocks_binary_file() -> None:
    """read_file must refuse a non-text file living inside the workspace."""
    artifact = get_settings().tool_workspace_root / "_test_binary_artifact.png"
    artifact.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x00binary-bytes")
    try:
        result = read_file("_test_binary_artifact.png")
        assert result.ok is False
        assert "not allowed" in result.content.lower()
    finally:
        artifact.unlink(missing_ok=True)
