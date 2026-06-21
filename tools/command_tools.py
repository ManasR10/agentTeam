from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from config import Settings, get_settings
from tools.results import CommandResult
from tools.safety import PathSafetyError, resolve_inside_workspace
from tools.schemas import ToolResult

# Phase 3 deliberately exposes NO generic shell command tool. Every command is
# constructed in Python from a fixed template; the model only supplies
# constrained arguments (test paths, a check name). subprocess always runs with
# shell=False and a validated workspace cwd.


class UnsupportedCommandError(ValueError):
    """Raised when a requested check/command is not on the allow-list."""


# Small, explicit allow-list of extra pytest flags the model may pass.
ALLOWED_PYTEST_ARGS = frozenset({"-q", "-x", "-v", "--no-header"})

# Named checks the orchestrator and the coder may run. Kept tiny on purpose.
ALLOWED_CHECKS = frozenset({"pytest", "py_compile"})

# Explicit compile targets. We never compile the whole workspace because it
# contains .venv, caches and .git; we list project source instead.
_COMPILE_TARGET_CANDIDATES = (
    "agents",
    "tools",
    "tests",
    "config.py",
    "llm.py",
    "phase1_demo.py",
    "phase2_demo.py",
    "phase3_demo.py",
    "smoke_test.py",
)


def _truncate(text: str | None, limit: int) -> tuple[str, bool]:
    """Keep both ends of long output; the tail usually holds the traceback."""
    if not text:
        return "", False
    if len(text) <= limit:
        return text, False
    half = limit // 2
    return (
        text[:half] + "\n... output truncated ...\n" + text[-half:],
        True,
    )


def run_process(
    command_name: str,
    argv: list[str],
    settings: Settings,
) -> CommandResult:
    """
    Run `argv` with shell=False inside the workspace, capturing bounded output.

    A timeout terminates the process and is reported as a normal failed
    CommandResult (exit_code None, timed_out True) rather than hanging.
    """
    limit = settings.max_command_output_chars
    # Don't write .pyc into the target repo: it litters the user's workspace and,
    # worse, a stale cache from one test run can mask the next run's source.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    start = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - argv is built from fixed templates
            argv,
            cwd=settings.tool_workspace_root,
            shell=False,
            capture_output=True,
            text=True,
            timeout=settings.command_timeout_seconds,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        duration = time.monotonic() - start
        stdout, t1 = _truncate(exc.stdout, limit)
        stderr, t2 = _truncate(exc.stderr, limit)
        return CommandResult(
            command_name=command_name,
            argv=tuple(argv),
            exit_code=None,
            stdout=stdout,
            stderr=stderr,
            timed_out=True,
            duration_seconds=duration,
            output_truncated=t1 or t2,
        )

    duration = time.monotonic() - start
    stdout, t1 = _truncate(proc.stdout, limit)
    stderr, t2 = _truncate(proc.stderr, limit)
    return CommandResult(
        command_name=command_name,
        argv=tuple(argv),
        exit_code=proc.returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=False,
        duration_seconds=duration,
        output_truncated=t1 or t2,
    )


def command_result_to_tool_result(result: CommandResult) -> ToolResult:
    """Render a CommandResult as a ToolResult for the model-facing tool loop."""
    if result.timed_out:
        status = "TIMEOUT"
        ok = False
    elif result.exit_code == 0:
        status = "PASS"
        ok = True
    else:
        status = "FAIL"
        ok = False
    body = (
        f"[{status}] {result.command_name} "
        f"(exit={result.exit_code}, {result.duration_seconds:.2f}s)\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )
    return ToolResult(
        ok=ok,
        content=body,
        metadata={
            "command_name": result.command_name,
            "argv": list(result.argv),
            "exit_code": result.exit_code,
            "timed_out": result.timed_out,
            "duration_seconds": result.duration_seconds,
            "output_truncated": result.output_truncated,
            "status": status,
        },
    )


def compile_targets(settings: Settings) -> list[str]:
    """Existing project paths to compile (never the whole workspace)."""
    root = settings.tool_workspace_root
    return [name for name in _COMPILE_TARGET_CANDIDATES if (root / name).exists()]


def build_check_argv(check: str, settings: Settings) -> list[str]:
    """Construct the argv for a named check. The model never supplies argv."""
    if check == "pytest":
        return [sys.executable, "-m", "pytest", "-q"]
    if check == "py_compile":
        targets = compile_targets(settings)
        if not targets:
            # Never fall back to bare `compileall -q`, which would compile the
            # current directory (including .venv/.git/caches).
            raise UnsupportedCommandError(
                "py_compile has no known project targets in this workspace."
            )
        return [sys.executable, "-m", "compileall", "-q", *targets]
    raise UnsupportedCommandError(f"Unsupported check: {check!r}")


def run_named_check(check: str, settings: Settings | None = None) -> CommandResult:
    """Run a named check and return its CommandResult (orchestrator-facing)."""
    settings = settings or get_settings()
    argv = build_check_argv(check, settings)
    return run_process(check, argv, settings)


def _validate_test_paths(paths: list[str], settings: Settings) -> list[str]:
    """Resolve test paths and require them to live under tests/."""
    validated: list[str] = []
    for path in paths:
        resolved = resolve_inside_workspace(
            path, workspace_root=settings.tool_workspace_root
        )
        relative = resolved.relative_to(settings.tool_workspace_root.resolve())
        if relative.parts[:1] != ("tests",):
            raise PathSafetyError(f"Test path must be under tests/: {path}")
        validated.append(str(relative))
    return validated


# --- Registered tools ------------------------------------------------------


def run_tests(
    paths: list[str] | None = None,
    extra_args: list[str] | None = None,
) -> ToolResult:
    """Run pytest on validated test paths with an allow-listed set of flags."""
    settings = get_settings()
    try:
        resolved_paths = (
            _validate_test_paths(paths, settings) if paths else []
        )
    except PathSafetyError as exc:
        return ToolResult(ok=False, content=str(exc), metadata={"paths": paths})

    safe_args: list[str] = []
    for arg in extra_args or []:
        if arg not in ALLOWED_PYTEST_ARGS:
            return ToolResult(
                ok=False,
                content=f"Disallowed pytest argument: {arg!r}",
                metadata={"extra_args": extra_args},
            )
        safe_args.append(arg)

    argv = [sys.executable, "-m", "pytest", *resolved_paths, *safe_args]
    return command_result_to_tool_result(run_process("pytest", argv, settings))


def run_check(check: str) -> ToolResult:
    """Run a single named check (pytest or py_compile)."""
    settings = get_settings()
    try:
        argv = build_check_argv(check, settings)
    except UnsupportedCommandError as exc:
        return ToolResult(
            ok=False,
            content=str(exc),
            metadata={"check": check, "allowed": sorted(ALLOWED_CHECKS)},
        )
    return command_result_to_tool_result(run_process(check, argv, settings))
