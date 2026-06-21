from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agents import coder
from agents.coder import (
    CodingParseError,
    implement_repo_task,
    parse_coding_result,
)
from agents.state import PlanningResult

VALID_CODER_JSON = """
{
  "summary": "Added a token formatter and used it in the output.",
  "changed_files": [
    {"path": "agents/formatting.py", "reason": "Use the new helper."},
    {"path": "tests/test_formatting.py", "reason": "Cover the helper."}
  ],
  "tests_requested": ["tests/test_formatting.py"],
  "known_issues": []
}
""".strip()


def _plan() -> PlanningResult:
    return PlanningResult(
        task="Add a token formatter",
        repo_summary="summary",
        relevant_files=[],
        implementation_plan=["Add helper", "Use it", "Test it"],
        files_likely_to_change=["agents/formatting.py"],
        tests_to_add=["tests/test_formatting.py"],
        risks=[],
        unknowns=[],
        raw_response="{}",
    )


@dataclass
class _FakeToolCall:
    name: str
    input: dict[str, Any]
    ok: bool = True


@dataclass
class _FakeRun:
    text: str
    stop_reason: str | None = "end_turn"
    tool_calls: list[_FakeToolCall] = None  # type: ignore[assignment]
    iterations: int = 3
    input_tokens: int = 1200
    output_tokens: int = 400

    def __post_init__(self) -> None:
        if self.tool_calls is None:
            self.tool_calls = []


# --- pure parser -----------------------------------------------------------


def test_valid_coder_json_parses() -> None:
    result = parse_coding_result(VALID_CODER_JSON)
    assert result.summary.startswith("Added")
    assert len(result.reported_changed_files) == 2
    assert result.reported_changed_files[0].path == "agents/formatting.py"
    assert result.tests_requested == ("tests/test_formatting.py",)
    # Attached fields are empty until implement_repo_task fills them.
    assert result.actual_changed_files == ()
    assert result.iterations == 0


def test_missing_summary_rejected() -> None:
    with pytest.raises(CodingParseError):
        parse_coding_result('{"changed_files": [], "tests_requested": [], '
                             '"known_issues": []}')


def test_wrong_changed_files_type_rejected() -> None:
    with pytest.raises(CodingParseError):
        parse_coding_result('{"summary": "x", "changed_files": "nope", '
                             '"tests_requested": [], "known_issues": []}')


def test_changed_file_missing_reason_rejected() -> None:
    with pytest.raises(CodingParseError):
        parse_coding_result('{"summary": "x", "changed_files": '
                             '[{"path": "a.py"}], "tests_requested": [], '
                             '"known_issues": []}')


def test_invalid_json_rejected() -> None:
    with pytest.raises(CodingParseError):
        parse_coding_result("not json at all")


# --- implement_repo_task ---------------------------------------------------


def test_actual_changed_files_come_from_git_not_model(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_call(*, prompt, system, tools, tool_executor, max_tokens,
                  max_iterations):
        captured["prompt"] = prompt
        captured["system"] = system
        captured["tools"] = tools
        captured["max_tokens"] = max_tokens
        captured["max_iterations"] = max_iterations
        return _FakeRun(
            text=VALID_CODER_JSON,
            tool_calls=[
                _FakeToolCall("read_file", {"path": "agents/formatting.py"}),
                _FakeToolCall("replace_in_file", {"path": "agents/formatting.py"}),
            ],
        )

    monkeypatch.setattr(coder, "call_agent_with_tools", fake_call)
    # Git is the source of truth; it reports a DIFFERENT set than the model.
    monkeypatch.setattr(
        coder, "get_changed_paths", lambda: ("agents/formatting.py",)
    )

    result = implement_repo_task("Add a token formatter", _plan())

    # The model claimed two files; git says one. We trust git.
    assert result.actual_changed_files == ("agents/formatting.py",)
    assert len(result.reported_changed_files) == 2
    # Evidence + usage attached from the run.
    assert result.tool_calls == (
        "read_file(agents/formatting.py): ok",
        "replace_in_file(agents/formatting.py): ok",
    )
    assert result.iterations == 3
    assert result.input_tokens == 1200
    assert result.output_tokens == 400


def test_coder_gets_plan_and_coder_profile(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_call(*, prompt, system, tools, tool_executor, max_tokens,
                  max_iterations):
        captured["prompt"] = prompt
        captured["tools"] = tools
        captured["max_tokens"] = max_tokens
        captured["max_iterations"] = max_iterations
        return _FakeRun(text=VALID_CODER_JSON)

    monkeypatch.setattr(coder, "call_agent_with_tools", fake_call)
    monkeypatch.setattr(coder, "get_changed_paths", lambda: ())

    implement_repo_task("Add a token formatter", _plan())

    # Plan content reaches the coder prompt.
    assert "Add helper" in captured["prompt"]
    assert "agents/formatting.py" in captured["prompt"]
    # Coder profile includes write + command + git tools.
    names = {t["name"] for t in captured["tools"]}
    assert {"create_file", "replace_in_file", "write_file",
            "run_tests", "git_diff"} <= names
    # Limits overridden for coding.
    assert captured["max_tokens"] == 4096
    assert captured["max_iterations"] == 20


def test_truncated_coder_response_raises(monkeypatch) -> None:
    monkeypatch.setattr(
        coder,
        "call_agent_with_tools",
        lambda **kw: _FakeRun(text='{"summary": "x"', stop_reason="max_tokens"),
    )
    monkeypatch.setattr(coder, "get_changed_paths", lambda: ())
    with pytest.raises(CodingParseError, match="truncated"):
        implement_repo_task("Add a token formatter", _plan())


def test_empty_task_rejected() -> None:
    with pytest.raises(ValueError):
        implement_repo_task("   ", _plan())


def test_repair_uses_review_feedback(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_call(*, prompt, **kw):
        captured["prompt"] = prompt
        return _FakeRun(text=VALID_CODER_JSON)

    monkeypatch.setattr(coder, "call_agent_with_tools", fake_call)
    monkeypatch.setattr(coder, "get_changed_paths", lambda: ())

    implement_repo_task(
        "Add a token formatter",
        _plan(),
        review_feedback="major: fix the missing hash comparison",
    )
    assert "reviewer requested changes" in captured["prompt"].lower()
    assert "missing hash comparison" in captured["prompt"]
