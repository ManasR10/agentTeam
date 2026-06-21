from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from agents.coder import _plan_to_text
from agents.planner import extract_json_object
from agents.prompts import REVIEWER_SYSTEM_PROMPT, build_review_prompt
from agents.results import ReviewIssue, ReviewResult
from agents.state import PlanningResult
from llm import AgentRunResult, call_agent_with_tools
from tools.results import CommandResult
from tools.registry import (
    REVIEWER_TOOL_NAMES,
    get_tool_definitions,
    make_tool_executor,
)

# The reviewer usually needs one turn if all evidence is in the prompt.
REVIEWER_MAX_TOKENS = 3000
REVIEWER_MAX_ITERATIONS = 6

_VALID_VERDICTS = {"approved", "changes_requested"}
_VALID_SEVERITIES = {"critical", "major", "minor"}


class ReviewParseError(RuntimeError):
    """Raised when the reviewer returns invalid or inconsistent JSON."""


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str):
        raise ReviewParseError(f"Expected {key!r} to be a string.")
    return value


def _parse_issues(data: dict[str, Any]) -> tuple[ReviewIssue, ...]:
    value = data.get("issues")
    if not isinstance(value, list):
        raise ReviewParseError("Expected 'issues' to be a list.")
    issues: list[ReviewIssue] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ReviewParseError(f"Expected issues[{index}] to be an object.")
        severity = item.get("severity")
        if severity not in _VALID_SEVERITIES:
            raise ReviewParseError(
                f"issues[{index}].severity must be one of {_VALID_SEVERITIES}."
            )
        path = item.get("path")
        if path is not None and not isinstance(path, str):
            raise ReviewParseError(
                f"issues[{index}].path must be a string or null."
            )
        description = item.get("description")
        required_change = item.get("required_change")
        if not isinstance(description, str):
            raise ReviewParseError(
                f"issues[{index}].description must be a string."
            )
        if not isinstance(required_change, str):
            raise ReviewParseError(
                f"issues[{index}].required_change must be a string."
            )
        issues.append(
            ReviewIssue(
                severity=severity,
                path=path,
                description=description,
                required_change=required_change,
            )
        )
    return tuple(issues)


def parse_review_result(raw_response: str) -> ReviewResult:
    """
    Parse and validate the reviewer's JSON into a ReviewResult.

    Enforces cross-field consistency: an 'approved' verdict cannot carry a
    'critical' issue, and 'changes_requested' must list at least one issue.
    """
    clean_response = extract_json_object(raw_response)
    try:
        data = json.loads(clean_response)
    except json.JSONDecodeError as exc:
        raise ReviewParseError(f"Reviewer returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ReviewParseError("Reviewer JSON must be an object.")

    verdict = _require_string(data, "verdict")
    if verdict not in _VALID_VERDICTS:
        raise ReviewParseError(f"verdict must be one of {_VALID_VERDICTS}.")
    issues = _parse_issues(data)

    if verdict == "approved" and any(i.severity == "critical" for i in issues):
        raise ReviewParseError(
            "Inconsistent review: 'approved' cannot include a critical issue."
        )
    if verdict == "changes_requested" and not issues:
        raise ReviewParseError(
            "Inconsistent review: 'changes_requested' must list at least one "
            "issue."
        )

    return ReviewResult(
        verdict=verdict,  # type: ignore[arg-type]
        summary=_require_string(data, "summary"),
        issues=issues,
        tests_assessment=_require_string(data, "tests_assessment"),
    )


def format_command_results(results: Sequence[CommandResult]) -> str:
    """Render command results as honest PASS/FAIL/TIMEOUT/NOT RUN lines."""
    if not results:
        return "NOT RUN — no verification commands were executed."
    lines: list[str] = []
    for result in results:
        if result.timed_out:
            status = "TIMEOUT"
        elif result.exit_code == 0:
            status = "PASS"
        else:
            status = "FAIL"
        lines.append(
            f"[{status}] {result.command_name} (exit={result.exit_code}, "
            f"{result.duration_seconds:.2f}s)"
        )
        tail = (result.stderr or result.stdout or "").strip()
        if status != "PASS" and tail:
            lines.append(tail[-2000:])
    return "\n".join(lines)


def run_reviewer(
    task: str,
    plan: PlanningResult,
    diff: str,
    command_results: Sequence[CommandResult],
    *,
    changed_files: Sequence[str] = (),
    coder_summary: str = "",
) -> tuple[ReviewResult, AgentRunResult]:
    """
    Run the reviewer and return both the verdict and the raw run (for tokens).

    Orchestrator-facing: the run result lets P3.7 aggregate token usage.
    """
    prompt = build_review_prompt(
        task=task,
        plan_text=_plan_to_text(plan),
        diff_text=diff or "(no diff)",
        tests_text=format_command_results(command_results),
        changed_files="\n".join(changed_files) or "(none)",
        coder_summary=coder_summary or "(none provided)",
    )
    run_result = call_agent_with_tools(
        prompt=prompt,
        system=REVIEWER_SYSTEM_PROMPT,
        tools=get_tool_definitions(REVIEWER_TOOL_NAMES),
        tool_executor=make_tool_executor(REVIEWER_TOOL_NAMES),
        max_tokens=REVIEWER_MAX_TOKENS,
        max_iterations=REVIEWER_MAX_ITERATIONS,
    )
    if run_result.stop_reason == "max_tokens":
        raise ReviewParseError(
            "Reviewer response was truncated (hit max_tokens). "
            "Increase max_tokens and retry."
        )
    return parse_review_result(run_result.text), run_result


def review_implementation(
    task: str,
    plan: PlanningResult,
    diff: str,
    command_results: Sequence[CommandResult],
    *,
    changed_files: Sequence[str] = (),
    coder_summary: str = "",
) -> ReviewResult:
    """Public API: review an implementation and return the verdict."""
    result, _run = run_reviewer(
        task,
        plan,
        diff,
        command_results,
        changed_files=changed_files,
        coder_summary=coder_summary,
    )
    return result
