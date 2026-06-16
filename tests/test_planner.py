from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agents.planner import (
    PlanningParseError,
    extract_json_object,
    parse_planning_result,
    plan_repo_task,
    strip_json_fences,
)
from agents.prompts import build_planning_prompt
from tests.fixtures import VALID_PLANNER_JSON


def test_strip_json_fences_handles_json_fence() -> None:
    raw = f"```json\n{VALID_PLANNER_JSON}\n```"
    assert strip_json_fences(raw) == VALID_PLANNER_JSON


def test_strip_json_fences_handles_plain_fence() -> None:
    raw = f"```\n{VALID_PLANNER_JSON}\n```"
    assert strip_json_fences(raw) == VALID_PLANNER_JSON


def test_extract_json_object_strips_leading_prose() -> None:
    raw = f"Here is the plan:\n\n{VALID_PLANNER_JSON}"
    assert extract_json_object(raw) == VALID_PLANNER_JSON


def test_parse_planning_result_tolerates_prose_prefix() -> None:
    raw = f"I now understand the repo. Let me produce JSON.\n\n{VALID_PLANNER_JSON}"
    result = parse_planning_result(raw)
    assert result.task == "Add CLI"


def test_parse_planning_result_valid_json() -> None:
    result = parse_planning_result(VALID_PLANNER_JSON)
    assert result.task == "Add CLI"
    assert result.repo_summary.startswith("DevAgent")
    assert len(result.relevant_files) == 2
    assert result.relevant_files[0].path == "llm.py"
    assert result.implementation_plan[0] == "Create cli.py"
    assert "README.md" in result.files_likely_to_change
    assert result.raw_response == VALID_PLANNER_JSON
    # Metadata defaults until plan_repo_task fills it in.
    assert result.tool_calls == []
    assert result.iterations == 0


def test_parse_planning_result_rejects_invalid_json() -> None:
    with pytest.raises(PlanningParseError):
        parse_planning_result("not json")


def test_parse_planning_result_rejects_top_level_list() -> None:
    with pytest.raises(PlanningParseError):
        parse_planning_result("[1, 2, 3]")


def test_parse_planning_result_rejects_missing_required_key() -> None:
    with pytest.raises(PlanningParseError):
        parse_planning_result("{}")


def test_parse_planning_result_rejects_wrong_list_type() -> None:
    bad_json = """
    {
      "task": "Add CLI",
      "repo_summary": "summary",
      "relevant_files": [],
      "implementation_plan": "not a list",
      "files_likely_to_change": [],
      "tests_to_add": [],
      "risks": [],
      "unknowns": []
    }
    """
    with pytest.raises(PlanningParseError):
        parse_planning_result(bad_json)


def test_parse_planning_result_rejects_relevant_file_missing_path() -> None:
    bad_json = """
    {
      "task": "Add CLI",
      "repo_summary": "summary",
      "relevant_files": [{"reason": "no path here"}],
      "implementation_plan": [],
      "files_likely_to_change": [],
      "tests_to_add": [],
      "risks": [],
      "unknowns": []
    }
    """
    with pytest.raises(PlanningParseError):
        parse_planning_result(bad_json)


def test_parse_planning_result_rejects_relevant_file_missing_reason() -> None:
    bad_json = """
    {
      "task": "Add CLI",
      "repo_summary": "summary",
      "relevant_files": [{"path": "llm.py"}],
      "implementation_plan": [],
      "files_likely_to_change": [],
      "tests_to_add": [],
      "risks": [],
      "unknowns": []
    }
    """
    with pytest.raises(PlanningParseError):
        parse_planning_result(bad_json)


def test_build_planning_prompt_rejects_empty_task() -> None:
    with pytest.raises(ValueError):
        build_planning_prompt("   ")


@dataclass(frozen=True, slots=True)
class _FakeToolCall:
    name: str
    input: dict[str, object]
    ok: bool = True


@dataclass(frozen=True, slots=True)
class _FakeRunResult:
    text: str
    stop_reason: str = "end_turn"
    iterations: int = 2
    input_tokens: int = 1500
    output_tokens: int = 300
    tool_calls: list[_FakeToolCall] = field(default_factory=list)


def test_plan_repo_task_uses_tool_loop_and_parses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    def fake_call_agent_with_tools(
        *, prompt: str, system: str, max_tokens: int, max_iterations: int
    ) -> _FakeRunResult:
        calls["prompt"] = prompt
        calls["system"] = system
        calls["max_tokens"] = max_tokens
        calls["max_iterations"] = max_iterations
        return _FakeRunResult(
            text=VALID_PLANNER_JSON,
            tool_calls=[
                _FakeToolCall(name="list_files", input={}),
                _FakeToolCall(name="read_file", input={"path": "llm.py"}),
                _FakeToolCall(name="read_file", input={"path": "ghost.py"}, ok=False),
            ],
        )

    monkeypatch.setattr(
        "agents.planner.call_agent_with_tools",
        fake_call_agent_with_tools,
    )

    result = plan_repo_task("Add CLI")

    assert result.task == "Add CLI"
    assert "Add CLI" in str(calls["prompt"])
    assert "repo-inspection planning agent" in str(calls["system"])
    assert calls["max_tokens"] == 4096
    assert calls["max_iterations"] == 12
    # Grounding metadata is attached from the run result.
    assert result.iterations == 2
    assert result.tool_calls == [
        "list_files: ok",
        "read_file(llm.py): ok",
        "read_file(ghost.py): failed",
    ]
    assert result.input_tokens == 1500
    assert result.output_tokens == 300


def test_plan_repo_task_raises_on_truncated_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call_agent_with_tools(
        *, prompt: str, system: str, max_tokens: int, max_iterations: int
    ) -> _FakeRunResult:
        return _FakeRunResult(text='{"task": "Add CLI"', stop_reason="max_tokens")

    monkeypatch.setattr(
        "agents.planner.call_agent_with_tools",
        fake_call_agent_with_tools,
    )

    with pytest.raises(PlanningParseError, match="truncated"):
        plan_repo_task("Add CLI")
