from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from agents.planner import extract_json_object
from agents.prompts import (
    CODER_SYSTEM_PROMPT,
    build_coding_prompt,
    build_repair_prompt,
)
from agents.results import ChangedFileSummary, CodingResult
from agents.state import PlanningResult
from llm import call_agent_with_tools
from tools.git_tools import get_changed_paths
from tools.registry import (
    CODER_TOOL_NAMES,
    get_tool_definitions,
    make_tool_executor,
)

# Coding needs more tool rounds than planning: read, edit, test, inspect diff.
CODER_MAX_TOKENS = 4096
CODER_MAX_ITERATIONS = 20


class CodingParseError(RuntimeError):
    """Raised when the coder returns invalid or incomplete JSON."""


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise CodingParseError(f"Expected {key!r} to be a string.")
    return value


def _require_string_list(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if not isinstance(value, list):
        raise CodingParseError(f"Expected {key!r} to be a list.")
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise CodingParseError(f"Expected {key}[{index}] to be a string.")
    return tuple(value)


def _parse_changed_files(data: dict[str, Any]) -> tuple[ChangedFileSummary, ...]:
    value = data.get("changed_files")
    if not isinstance(value, list):
        raise CodingParseError("Expected 'changed_files' to be a list.")
    result: list[ChangedFileSummary] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise CodingParseError(
                f"Expected changed_files[{index}] to be an object."
            )
        path = item.get("path")
        reason = item.get("reason")
        if not isinstance(path, str):
            raise CodingParseError(
                f"Expected changed_files[{index}].path to be a string."
            )
        if not isinstance(reason, str):
            raise CodingParseError(
                f"Expected changed_files[{index}].reason to be a string."
            )
        result.append(ChangedFileSummary(path=path, reason=reason))
    return tuple(result)


def parse_coding_result(raw_response: str) -> CodingResult:
    """
    Parse and validate the coder's JSON into a CodingResult.

    Pure function: it only reads the model JSON. The authoritative changed-file
    list, tool-call evidence, and token usage are attached afterwards by
    implement_repo_task — never trusted from the model's own output.
    """
    clean_response = extract_json_object(raw_response)
    try:
        data = json.loads(clean_response)
    except json.JSONDecodeError as exc:
        raise CodingParseError(f"Coder returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise CodingParseError("Coder JSON must be an object.")

    return CodingResult(
        summary=_require_string(data, "summary"),
        reported_changed_files=_parse_changed_files(data),
        tests_requested=_require_string_list(data, "tests_requested"),
        known_issues=_require_string_list(data, "known_issues"),
    )


def _summarize_tool_calls(tool_calls: list[Any]) -> tuple[str, ...]:
    """Render tool-call records as readable 'name(path): ok/failed' strings."""
    summaries: list[str] = []
    for record in tool_calls:
        path = record.input.get("path") if isinstance(record.input, dict) else None
        base = f"{record.name}({path})" if path else record.name
        status = "ok" if record.ok else "failed"
        summaries.append(f"{base}: {status}")
    return tuple(summaries)


def _plan_to_text(plan: PlanningResult) -> str:
    """Compact, plain-text rendering of the plan to feed the coder prompt."""
    lines: list[str] = []
    if plan.implementation_plan:
        lines.append("Steps:")
        for index, step in enumerate(plan.implementation_plan, 1):
            lines.append(f"{index}. {step}")
    if plan.files_likely_to_change:
        lines.append(
            "Files likely to change: "
            + ", ".join(plan.files_likely_to_change)
        )
    if plan.tests_to_add:
        lines.append("Tests to add: " + ", ".join(plan.tests_to_add))
    return "\n".join(lines) if lines else "(no plan details)"


def implement_repo_task(
    task: str,
    plan: PlanningResult,
    *,
    review_feedback: str | None = None,
    tool_executor: Any = None,
) -> CodingResult:
    """
    Execute the approved plan using the coder's capability-scoped tools.

    When `review_feedback` is supplied, this runs a repair attempt addressing
    the reviewer's issues instead of a fresh implementation. `tool_executor`
    lets a caller (e.g. the Phase 4 workflow) wrap the coder's tools to capture
    rollback snapshots and audit events; it defaults to the standard
    capability-scoped executor.
    """
    plan_text = _plan_to_text(plan)
    if review_feedback is not None:
        prompt = build_repair_prompt(task, plan_text, review_feedback)
    else:
        prompt = build_coding_prompt(task, plan_text)

    run_result = call_agent_with_tools(
        prompt=prompt,
        system=CODER_SYSTEM_PROMPT,
        tools=get_tool_definitions(CODER_TOOL_NAMES),
        tool_executor=tool_executor or make_tool_executor(CODER_TOOL_NAMES),
        max_tokens=CODER_MAX_TOKENS,
        max_iterations=CODER_MAX_ITERATIONS,
    )

    if run_result.stop_reason == "max_tokens":
        raise CodingParseError(
            "Coder response was truncated (hit max_tokens). "
            "Increase max_tokens and retry."
        )

    result = parse_coding_result(run_result.text)
    return replace(
        result,
        # Authoritative: what git says changed, not what the model claimed.
        actual_changed_files=get_changed_paths(),
        tool_calls=_summarize_tool_calls(run_result.tool_calls),
        iterations=run_result.iterations,
        input_tokens=run_result.input_tokens,
        output_tokens=run_result.output_tokens,
    )
