from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from agents.state import PlanningResult
from tools.results import CommandResult

# Agent-layer result models. Unlike tools/results.py, these may depend on both
# the planning result and low-level command results.


@dataclass(frozen=True, slots=True)
class ChangedFileSummary:
    """A file the coder *claims* it changed, with its stated reason."""

    path: str
    reason: str


@dataclass(frozen=True, slots=True)
class CodingResult:
    """
    Result of one coding (or repair) attempt.

    `reported_changed_files` is the model's own claim and is not trusted as the
    source of truth. `actual_changed_files` is filled in by the orchestrator
    from git, and is authoritative.
    """

    summary: str
    reported_changed_files: tuple[ChangedFileSummary, ...]
    actual_changed_files: tuple[str, ...]
    tests_requested: tuple[str, ...]
    known_issues: tuple[str, ...]
    tool_calls: tuple[str, ...]
    iterations: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ReviewIssue:
    """A single problem the reviewer found, with the change it requires."""

    severity: Literal["critical", "major", "minor"]
    path: str | None
    description: str
    required_change: str


@dataclass(frozen=True, slots=True)
class ReviewResult:
    """The reviewer's verdict on one implementation attempt."""

    verdict: Literal["approved", "changes_requested"]
    summary: str
    issues: tuple[ReviewIssue, ...]
    tests_assessment: str


@dataclass(frozen=True, slots=True)
class ImplementationRunResult:
    """The final auditable report for a full execute_repo_task run."""

    task: str
    status: Literal["planned", "completed", "changes_requested", "failed"]
    plan: PlanningResult
    changed_files: tuple[str, ...]
    command_results: tuple[CommandResult, ...]
    reviews: tuple[ReviewResult, ...]
    diff: str
    summary: str
    input_tokens: int
    output_tokens: int
