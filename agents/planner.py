from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

from agents.prompts import PLANNER_SYSTEM_PROMPT, build_planning_prompt
from agents.state import PlanningResult, RelevantFile
from llm import call_agent_with_tools

# A planning run reads several files and then emits the full JSON plan. The
# Phase 0/1 default of 1024 output tokens is too small for that final turn and
# would truncate the JSON mid-object, so the planner asks for more headroom.
# max_tokens is only a ceiling: unused tokens are never billed.
PLANNER_MAX_TOKENS = 4096

# A planning run lists the repo and reads several files before answering. The
# Phase 1 default of 5 iterations is too tight — the planner can read files one
# per turn and hit the cap before emitting JSON. Give it more rounds.
PLANNER_MAX_ITERATIONS = 12


class PlanningParseError(RuntimeError):
    """Raised when the planner returns invalid or incomplete JSON."""


def strip_json_fences(text: str) -> str:
    """
    Remove common markdown JSON fences if the model accidentally returns them.

    The prompt says JSON only, but this makes the parser slightly more robust
    against a stray ```json ... ``` wrapper.
    """
    clean = text.strip()
    if clean.startswith("```json"):
        clean = clean.removeprefix("```json").strip()
    elif clean.startswith("```"):
        clean = clean.removeprefix("```").strip()
    if clean.endswith("```"):
        clean = clean.removesuffix("```").strip()
    return clean


def extract_json_object(text: str) -> str:
    """
    Pull the JSON object out of a model response.

    The prompt asks for JSON only, but models sometimes wrap it in a fence or
    prefix a sentence like "Here is the plan:". After stripping fences, if the
    text isn't already a bare object, slice from the first '{' to the last '}'.
    """
    clean = strip_json_fences(text)
    if clean.startswith("{") and clean.endswith("}"):
        return clean
    start = clean.find("{")
    end = clean.rfind("}")
    if start != -1 and end > start:
        return clean[start : end + 1]
    return clean


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise PlanningParseError(f"Expected {key!r} to be a string.")
    return value


def _require_string_list(data: dict[str, Any], key: str) -> list[str]:
    value = data.get(key)
    if not isinstance(value, list):
        raise PlanningParseError(f"Expected {key!r} to be a list.")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise PlanningParseError(f"Expected {key}[{index}] to be a string.")
        result.append(item)
    return result


def _parse_relevant_files(data: dict[str, Any]) -> list[RelevantFile]:
    value = data.get("relevant_files")
    if not isinstance(value, list):
        raise PlanningParseError("Expected 'relevant_files' to be a list.")
    result: list[RelevantFile] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise PlanningParseError(
                f"Expected relevant_files[{index}] to be an object."
            )
        path = item.get("path")
        reason = item.get("reason")
        if not isinstance(path, str):
            raise PlanningParseError(
                f"Expected relevant_files[{index}].path to be a string."
            )
        if not isinstance(reason, str):
            raise PlanningParseError(
                f"Expected relevant_files[{index}].reason to be a string."
            )
        result.append(RelevantFile(path=path, reason=reason))
    return result


def parse_planning_result(raw_response: str) -> PlanningResult:
    """
    Parse and validate the planner's JSON response into a PlanningResult.

    Pure function: it only reads the JSON text. Tool-call/iteration metadata is
    attached later by plan_repo_task. Never trusts the model's output shape.
    """
    clean_response = extract_json_object(raw_response)
    try:
        data = json.loads(clean_response)
    except json.JSONDecodeError as exc:
        raise PlanningParseError(f"Planner returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise PlanningParseError("Planner JSON must be an object.")

    return PlanningResult(
        task=_require_string(data, "task"),
        repo_summary=_require_string(data, "repo_summary"),
        relevant_files=_parse_relevant_files(data),
        implementation_plan=_require_string_list(data, "implementation_plan"),
        files_likely_to_change=_require_string_list(
            data, "files_likely_to_change"
        ),
        tests_to_add=_require_string_list(data, "tests_to_add"),
        risks=_require_string_list(data, "risks"),
        unknowns=_require_string_list(data, "unknowns"),
        raw_response=raw_response,
    )


def _summarize_tool_calls(tool_calls: list[Any]) -> list[str]:
    """
    Render tool-call records as readable strings with their outcome.

    Keeping the ok/failed status (e.g. "read_file(llm.py): ok") makes the
    grounding evidence honest — a failed read is not proof the file was read.
    """
    summaries: list[str] = []
    for record in tool_calls:
        path = record.input.get("path") if isinstance(record.input, dict) else None
        base = f"{record.name}({path})" if path else record.name
        status = "ok" if record.ok else "failed"
        summaries.append(f"{base}: {status}")
    return summaries


def plan_repo_task(task: str) -> PlanningResult:
    """
    Inspect the repo using Phase 1 read-only tools and return a structured plan.

    Wires the planning prompt through the Phase 1 tool loop, validates the JSON
    response, and records which tools were actually called so the result can
    prove it inspected real files.
    """
    prompt = build_planning_prompt(task)
    run_result = call_agent_with_tools(
        prompt=prompt,
        system=PLANNER_SYSTEM_PROMPT,
        max_tokens=PLANNER_MAX_TOKENS,
        max_iterations=PLANNER_MAX_ITERATIONS,
    )

    # A truncated plan and genuine non-JSON both fail json.loads; distinguish
    # the common truncation case so the error is actionable.
    if run_result.stop_reason == "max_tokens":
        raise PlanningParseError(
            "Planner response was truncated (hit max_tokens). "
            "Increase max_tokens and retry."
        )

    result = parse_planning_result(run_result.text)
    return replace(
        result,
        tool_calls=_summarize_tool_calls(run_result.tool_calls),
        iterations=run_result.iterations,
        input_tokens=run_result.input_tokens,
        output_tokens=run_result.output_tokens,
    )
