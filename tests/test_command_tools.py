from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from config import get_settings
from tools import command_tools
from tools.command_tools import (
    UnsupportedCommandError,
    build_check_argv,
    run_check,
    run_process,
    run_tests,
)


@pytest.fixture()
def ws(monkeypatch, tmp_path: Path) -> Path:
    settings = replace(get_settings(), tool_workspace_root=tmp_path)
    monkeypatch.setattr(command_tools, "get_settings", lambda: settings)
    return tmp_path


class _Completed:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_run_tests_builds_argv_without_shell(monkeypatch, ws: Path) -> None:
    (ws / "tests").mkdir()
    (ws / "tests" / "test_x.py").write_text("def test_a():\n    assert True\n")
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _Completed(0, stdout="1 passed")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_tests(["tests/test_x.py"], ["-q"])

    assert result.ok is True
    argv = captured["argv"]
    assert argv[0] == sys.executable  # never a bare "pytest"
    assert argv[1:3] == ["-m", "pytest"]
    assert "tests/test_x.py" in argv
    assert "-q" in argv
    kwargs = captured["kwargs"]
    assert kwargs["shell"] is False
    assert kwargs["cwd"] == ws


def test_run_tests_rejects_path_outside_tests(ws: Path) -> None:
    result = run_tests(["llm.py"])
    assert result.ok is False
    assert "under tests/" in result.content


def test_run_tests_rejects_escaping_path(ws: Path) -> None:
    result = run_tests(["../secrets.py"])
    assert result.ok is False


def test_run_tests_rejects_disallowed_flag(ws: Path) -> None:
    (ws / "tests").mkdir()
    result = run_tests(["tests"], ["--maxfail=1; rm -rf /"])
    assert result.ok is False
    assert "disallowed" in result.content.lower()


def test_run_tests_captures_failing_exit_code(monkeypatch, ws: Path) -> None:
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Completed(1, stdout="1 failed")
    )
    result = run_tests([])
    assert result.ok is False
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 1
    assert result.metadata["status"] == "FAIL"


def test_run_process_handles_timeout(monkeypatch, ws: Path) -> None:
    def fake_run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd=argv, timeout=1, output="partial")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_process("pytest", [sys.executable, "-c", "pass"], get_settings_ws(ws))
    assert result.timed_out is True
    assert result.exit_code is None


def get_settings_ws(tmp_path: Path):
    return replace(get_settings(), tool_workspace_root=tmp_path)


def test_output_is_truncated(monkeypatch, tmp_path: Path) -> None:
    settings = replace(
        get_settings(), tool_workspace_root=tmp_path, max_command_output_chars=20
    )
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _Completed(0, "X" * 100))
    result = run_process("pytest", [sys.executable, "-c", "pass"], settings)
    assert result.output_truncated is True
    assert "output truncated" in result.stdout
    assert len(result.stdout) < 100


def test_unsupported_check_is_rejected(ws: Path) -> None:
    result = run_check("rm")
    assert result.ok is False
    assert "unsupported" in result.content.lower()


def test_build_check_argv_unknown_raises(ws: Path) -> None:
    with pytest.raises(UnsupportedCommandError):
        build_check_argv("curl", get_settings_ws(ws))


def test_py_compile_fails_clearly_with_no_targets(tmp_path: Path) -> None:
    """An empty workspace must not silently compile the current directory."""
    settings = replace(get_settings(), tool_workspace_root=tmp_path)
    with pytest.raises(UnsupportedCommandError):
        build_check_argv("py_compile", settings)


def test_compile_check_targets_exist_only(monkeypatch, tmp_path: Path) -> None:
    settings = replace(get_settings(), tool_workspace_root=tmp_path)
    (tmp_path / "config.py").write_text("x = 1\n")
    argv = build_check_argv("py_compile", settings)
    assert argv[:4] == [sys.executable, "-m", "compileall", "-q"]
    # Only existing targets are included; nonexistent ones are skipped.
    assert "config.py" in argv
    assert "llm.py" not in argv
