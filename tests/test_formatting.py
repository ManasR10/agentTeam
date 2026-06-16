from __future__ import annotations

from dataclasses import replace

from agents.formatting import format_planning_result
from agents.planner import parse_planning_result
from tests.fixtures import VALID_PLANNER_JSON


def test_format_planning_result_contains_main_sections() -> None:
    result = parse_planning_result(VALID_PLANNER_JSON)
    rendered = format_planning_result(result)

    assert "# DevAgent Implementation Plan" in rendered
    assert "## Task" in rendered
    assert "## Repo summary" in rendered
    assert "## Relevant files" in rendered
    assert "## Implementation plan" in rendered
    assert "`llm.py`" in rendered
    assert "Create cli.py" in rendered


def test_format_planning_result_strips_model_numbering() -> None:
    result = replace(
        parse_planning_result(VALID_PLANNER_JSON),
        implementation_plan=["1. First step", "2) Second step"],
    )
    rendered = format_planning_result(result)

    # Our own numbering, with the model's leading number removed (no "1. 1.").
    assert "1. First step" in rendered
    assert "2. Second step" in rendered
    assert "1. 1." not in rendered
    assert "2. 2)" not in rendered


def test_format_planning_result_hides_usage_by_default() -> None:
    result = parse_planning_result(VALID_PLANNER_JSON)
    assert "## Token usage" not in format_planning_result(result)


def test_format_planning_result_shows_usage_when_requested() -> None:
    result = replace(
        parse_planning_result(VALID_PLANNER_JSON),
        input_tokens=1500,
        output_tokens=300,
    )
    rendered = format_planning_result(result, include_usage=True)

    assert "## Token usage" in rendered
    assert "Input tokens: 1500" in rendered
    assert "Output tokens: 300" in rendered
    assert "Total tokens: 1800" in rendered
