from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class RelevantFile:
    """A repository file the planner believes is relevant to the task."""

    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class PlanningResult:
    """
    Structured planning output from the Phase 2 repo-inspection agent.

    This is the grounded, validated plan a future coder phase can consume.
    `tool_calls` and `iterations` are populated by plan_repo_task (not by the
    JSON parser) so the result can prove the planner actually inspected files.
    """

    task: str
    repo_summary: str
    relevant_files: list[RelevantFile]
    implementation_plan: list[str]
    files_likely_to_change: list[str]
    tests_to_add: list[str]
    risks: list[str]
    unknowns: list[str]
    raw_response: str
    tool_calls: list[str] = field(default_factory=list)
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
