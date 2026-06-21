from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agents.state import AgentState
from config import (
    ConfigurationError,
    get_settings,
    read_boolean,
    read_positive_integer,
)
from tools.file_tools import calculate_sha256, read_file
from tools.mutation_safety import (
    build_mutation_policy,
    is_protected_mutation_path,
)


def test_boolean_config_accepts_known_values(monkeypatch) -> None:
    for raw, expected in [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
        (" Yes ", True),
    ]:
        monkeypatch.setenv("DEVAGENT_TEST_BOOL", raw)
        assert read_boolean("DEVAGENT_TEST_BOOL", default=False) is expected


def test_boolean_config_uses_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("DEVAGENT_TEST_BOOL", raising=False)
    assert read_boolean("DEVAGENT_TEST_BOOL", default=True) is True
    assert read_boolean("DEVAGENT_TEST_BOOL", default=False) is False


def test_boolean_config_rejects_unknown_value(monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_TEST_BOOL", "maybe")
    with pytest.raises(ConfigurationError):
        read_boolean("DEVAGENT_TEST_BOOL", default=False)


def test_positive_limits_reject_zero(monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_TEST_INT", "0")
    with pytest.raises(ConfigurationError):
        read_positive_integer("DEVAGENT_TEST_INT", default=5)


def test_positive_limits_reject_negative(monkeypatch) -> None:
    monkeypatch.setenv("DEVAGENT_TEST_INT", "-3")
    with pytest.raises(ConfigurationError):
        read_positive_integer("DEVAGENT_TEST_INT", default=5)


def test_agent_state_has_independent_mutable_defaults() -> None:
    a = AgentState(task="a", workspace_root=Path("."))
    b = AgentState(task="b", workspace_root=Path("."))
    a.changed_files.append("x.py")
    a.command_results.append(object())  # type: ignore[arg-type]
    a.review_history.append(object())  # type: ignore[arg-type]
    assert b.changed_files == []
    assert b.command_results == []
    assert b.review_history == []


def test_mutation_policy_built_from_settings() -> None:
    policy = build_mutation_policy(get_settings())
    assert policy.max_files_changed > 0
    assert policy.max_file_write_chars > 0
    assert policy.max_total_write_chars >= policy.max_file_write_chars
    assert isinstance(policy.allow_create_files, bool)
    assert isinstance(policy.allow_overwrite_files, bool)


def test_protected_mutation_paths() -> None:
    blocked = [
        ".env",
        ".env.local",
        ".env.example",
        ".git/config",
        ".venv/lib/x.py",
        "node_modules/pkg/index.js",
        "requirements.lock.txt",
        "package-lock.json",
        "secrets/server.key",
        "data/app.sqlite",
        "build/out.py",
    ]
    for path in blocked:
        assert is_protected_mutation_path(Path(path)) is True, path

    allowed = ["llm.py", "agents/coder.py", "tests/test_coder.py", "README.md"]
    for path in allowed:
        assert is_protected_mutation_path(Path(path)) is False, path


def test_read_file_reports_full_content_sha256(monkeypatch, tmp_path) -> None:
    """sha256 must hash the full file, not the truncated returned text."""
    settings = replace(
        get_settings(),
        tool_workspace_root=tmp_path,
        max_file_read_chars=10,
    )
    monkeypatch.setattr("tools.file_tools.get_settings", lambda: settings)

    full = "abcdefghijABCDEFGHIJ"  # 20 chars, longer than the 10-char limit
    (tmp_path / "big.txt").write_text(full, encoding="utf-8")

    result = read_file("big.txt")
    assert result.ok is True
    assert result.metadata is not None
    assert result.metadata["truncated"] is True
    assert result.metadata["returned_chars"] == 10
    # Digest is over the whole file, so it matches what a write tool recomputes.
    assert result.metadata["sha256"] == calculate_sha256(full)
