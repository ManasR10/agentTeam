from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agents import reviewer
from agents.reviewer import (
    ReviewParseError,
    parse_review_result,
    review_implementation,
)
from agents.state import PlanningResult
from tools.results import CommandResult

APPROVED_JSON = """
{
  "verdict": "approved",
  "summary": "Looks correct and scoped.",
  "issues": [],
  "tests_assessment": "Tests cover the new helper."
}
""".strip()

CHANGES_JSON = """
{
  "verdict": "changes_requested",
  "summary": "Stale-write check is incomplete.",
  "issues": [
    {
      "severity": "major",
      "path": "tools/write_tools.py",
      "description": "write_file accepts a hash but does not compare it.",
      "required_change": "Compare current sha256 and reject mismatches."
    }
  ],
  "tests_assessment": "Missing a stale-write test."
}
""".strip()


def _plan() -> PlanningResult:
    return PlanningResult(
        task="Add hash check",
        repo_summary="s",
        relevant_files=[],
        implementation_plan=["Add compare"],
        files_likely_to_change=["tools/write_tools.py"],
        tests_to_add=[],
        risks=[],
        unknowns=[],
        raw_response="{}",
    )


@dataclass
class _FakeRun:
    text: str
    stop_reason: str | None = "end_turn"
    tool_calls: list = None  # type: ignore[assignment]
    iterations: int = 1
    input_tokens: int = 500
    output_tokens: int = 120

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []


def _cmd(name: str, exit_code: int) -> CommandResult:
    return CommandResult(
        command_name=name,
        argv=("python", "-m", name),
        exit_code=exit_code,
        stdout="ok" if exit_code == 0 else "boom",
        stderr="",
        timed_out=False,
        duration_seconds=0.1,
        output_truncated=False,
    )


# --- pure parser -----------------------------------------------------------


def test_approved_review_parses() -> None:
    result = parse_review_result(APPROVED_JSON)
    assert result.verdict == "approved"
    assert result.issues == ()


def test_changes_requested_review_parses() -> None:
    result = parse_review_result(CHANGES_JSON)
    assert result.verdict == "changes_requested"
    assert len(result.issues) == 1
    assert result.issues[0].severity == "major"
    assert result.issues[0].path == "tools/write_tools.py"


def test_approved_with_critical_issue_rejected() -> None:
    bad = """
    {"verdict": "approved", "summary": "x",
     "issues": [{"severity": "critical", "path": null,
                 "description": "d", "required_change": "r"}],
     "tests_assessment": "t"}
    """
    with pytest.raises(ReviewParseError):
        parse_review_result(bad)


def test_changes_requested_with_no_issues_rejected() -> None:
    bad = ('{"verdict": "changes_requested", "summary": "x", '
           '"issues": [], "tests_assessment": "t"}')
    with pytest.raises(ReviewParseError):
        parse_review_result(bad)


def test_invalid_severity_rejected() -> None:
    bad = ('{"verdict": "changes_requested", "summary": "x", '
           '"issues": [{"severity": "blocker", "path": null, '
           '"description": "d", "required_change": "r"}], '
           '"tests_assessment": "t"}')
    with pytest.raises(ReviewParseError):
        parse_review_result(bad)


def test_invalid_verdict_rejected() -> None:
    bad = ('{"verdict": "maybe", "summary": "x", "issues": [], '
           '"tests_assessment": "t"}')
    with pytest.raises(ReviewParseError):
        parse_review_result(bad)


def test_null_path_allowed() -> None:
    js = ('{"verdict": "changes_requested", "summary": "x", '
          '"issues": [{"severity": "minor", "path": null, '
          '"description": "d", "required_change": "r"}], '
          '"tests_assessment": "t"}')
    result = parse_review_result(js)
    assert result.issues[0].path is None


# --- review_implementation -------------------------------------------------


def test_review_prompt_includes_all_evidence(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_call(*, prompt, system, tools, tool_executor, max_tokens,
                  max_iterations):
        captured["prompt"] = prompt
        captured["tools"] = tools
        captured["max_tokens"] = max_tokens
        return _FakeRun(text=APPROVED_JSON)

    monkeypatch.setattr(reviewer, "call_agent_with_tools", fake_call)

    review_implementation(
        "Add hash check",
        _plan(),
        diff="diff --git a/tools/write_tools.py ... +compare",
        command_results=[_cmd("pytest", 0)],
        changed_files=["tools/write_tools.py"],
        coder_summary="Added the compare.",
    )

    prompt = captured["prompt"]
    assert "Add hash check" in prompt          # task
    assert "Add compare" in prompt              # plan
    assert "diff --git" in prompt               # diff
    assert "[PASS] pytest" in prompt            # test output
    assert "Added the compare." in prompt       # coder summary
    assert captured["max_tokens"] == 3000


def test_reviewer_has_no_mutation_tools(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_call(*, prompt, system, tools, tool_executor, max_tokens,
                  max_iterations):
        captured["tools"] = tools
        return _FakeRun(text=APPROVED_JSON)

    monkeypatch.setattr(reviewer, "call_agent_with_tools", fake_call)
    review_implementation("Add hash check", _plan(), diff="", command_results=[])

    names = {t["name"] for t in captured["tools"]}
    assert "write_file" not in names
    assert "create_file" not in names
    assert "replace_in_file" not in names
    assert "run_tests" not in names
    # It can still read and inspect git.
    assert {"read_file", "git_diff"} <= names


def test_truncated_review_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        reviewer,
        "call_agent_with_tools",
        lambda **kw: _FakeRun(text='{"verdict": "approved"', stop_reason="max_tokens"),
    )
    with pytest.raises(ReviewParseError, match="truncated"):
        review_implementation("Add hash check", _plan(), diff="", command_results=[])


def test_command_results_render_not_run() -> None:
    from agents.reviewer import format_command_results

    assert "NOT RUN" in format_command_results([])
    assert "[FAIL]" in format_command_results([_cmd("pytest", 1)])
