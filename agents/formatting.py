from __future__ import annotations

import re

from agents.results import ImplementationRunResult
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


def _command_status(exit_code: int | None, timed_out: bool) -> str:
    if timed_out:
        return "TIMEOUT"
    if exit_code == 0:
        return "PASS"
    return "FAIL"


def _diff_stats(diff: str) -> tuple[int, int]:
    """Count insertions/deletions from diff text (ignoring file headers)."""
    insertions = 0
    deletions = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            insertions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return insertions, deletions


def format_implementation_result(
    result: ImplementationRunResult,
    *,
    show_diff: bool = False,
) -> str:
    """Render an ImplementationRunResult as readable markdown for the CLI."""
    lines: list[str] = []
    lines.append("# DevAgent implementation")
    lines.append("")
    lines.append("## Status")
    lines.append(result.status)
    lines.append("")
    lines.append("## Task")
    lines.append(result.task)
    lines.append("")
    lines.append("## Summary")
    lines.append(result.summary)
    lines.append("")

    lines.append("## Changed files")
    if result.changed_files:
        for path in result.changed_files:
            lines.append(f"- `{path}`")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Verification")
    if result.command_results:
        for cmd in result.command_results:
            status = _command_status(cmd.exit_code, cmd.timed_out)
            lines.append(f"- {cmd.command_name}: {status}")
    else:
        lines.append("- NOT RUN")
    lines.append("")

    lines.append("## Review")
    if result.reviews:
        last = result.reviews[-1]
        lines.append(f"- Verdict: {last.verdict}")
        for issue in last.issues:
            location = issue.path or "(general)"
            lines.append(
                f"  - [{issue.severity}] {location}: {issue.required_change}"
            )
    else:
        lines.append("- No review performed")
    lines.append("")

    insertions, deletions = _diff_stats(result.diff)
    lines.append("## Diff summary")
    lines.append(f"- {len(result.changed_files)} file(s) changed")
    lines.append(f"- {insertions} insertion(s)")
    lines.append(f"- {deletions} deletion(s)")
    if show_diff and result.diff:
        lines.append("")
        lines.append("## Diff")
        lines.append("```diff")
        lines.append(result.diff)
        lines.append("```")
    lines.append("")

    lines.append("## Token usage")
    lines.append(f"- Input tokens: {result.input_tokens}")
    lines.append(f"- Output tokens: {result.output_tokens}")
    lines.append(
        f"- Total tokens: {result.input_tokens + result.output_tokens}"
    )

    return "\n".join(lines)
