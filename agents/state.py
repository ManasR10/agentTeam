from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type checking to avoid a runtime import cycle:
    # agents.results imports PlanningResult from this module. `from __future__
    # import annotations` keeps these annotations as strings at runtime.
    from agents.results import ReviewResult
    from tools.results import CommandResult


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


@dataclass(slots=True)
class AgentState:
    """
    Mutable orchestration state threaded through a Phase 3 run.

    This holds only orchestration data — plans, file lists, command and review
    history, counters. It must never store raw Anthropic SDK response objects.
    It gives Phase 4 (LangGraph) a direct migration target: today's manual
    Python orchestration maps onto a graph state object unchanged.
    """

    task: str
    workspace_root: Path
    plan: PlanningResult | None = None
    changed_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    command_results: list[CommandResult] = field(default_factory=list)
    review_history: list[ReviewResult] = field(default_factory=list)
    implementation_iterations: int = 0
    review_iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
