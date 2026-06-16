from __future__ import annotations

import re

from agents.state import PlanningResult

# Strips a leading "1. " / "2) " the model sometimes bakes into a plan step, so
# we don't render "1. 1. Do the thing" once our own numbering is added.
_LEADING_NUMBER = re.compile(r"^\s*\d+[.)]\s*")


def format_planning_result(
    result: PlanningResult,
    *,
    include_usage: bool = False,
) -> str:
    """
    Render a PlanningResult as readable markdown for terminal output.

    With include_usage=True, a token-usage section is appended at the bottom as
    run metadata (input/output/total tokens). This data lives on the result
    object, not in the planner's JSON, so it never pollutes the plan itself.
    """
    lines: list[str] = []

    lines.append("# DevAgent Implementation Plan")
    lines.append("")

    lines.append("## Task")
    lines.append(result.task)
    lines.append("")

    lines.append("## Repo summary")
    lines.append(result.repo_summary)
    lines.append("")

    lines.append("## Relevant files")
    if result.relevant_files:
        for item in result.relevant_files:
            lines.append(f"- `{item.path}` — {item.reason}")
    else:
        lines.append("- None identified")
    lines.append("")

    lines.append("## Implementation plan")
    if result.implementation_plan:
        for index, step in enumerate(result.implementation_plan, start=1):
            clean_step = _LEADING_NUMBER.sub("", step)
            lines.append(f"{index}. {clean_step}")
    else:
        lines.append("No implementation steps returned.")
    lines.append("")

    lines.append("## Files likely to change")
    if result.files_likely_to_change:
        for path in result.files_likely_to_change:
            lines.append(f"- `{path}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Tests to add")
    if result.tests_to_add:
        for test in result.tests_to_add:
            lines.append(f"- {test}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Risks")
    if result.risks:
        for risk in result.risks:
            lines.append(f"- {risk}")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Unknowns")
    if result.unknowns:
        for unknown in result.unknowns:
            lines.append(f"- {unknown}")
    else:
        lines.append("- None")

    if include_usage:
        lines.append("")
        lines.append("## Token usage")
        lines.append(f"- Input tokens: {result.input_tokens}")
        lines.append(f"- Output tokens: {result.output_tokens}")
        lines.append(
            f"- Total tokens: {result.input_tokens + result.output_tokens}"
        )

    return "\n".join(lines)
