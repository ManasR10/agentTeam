from __future__ import annotations

from collections.abc import Sequence

from agents.coder import CodingParseError, implement_repo_task
from agents.planner import plan_repo_task
from agents.results import (
    CodingResult,
    ImplementationRunResult,
    ReviewResult,
)
from agents.reviewer import ReviewParseError, run_reviewer
from agents.state import AgentState, PlanningResult
from config import Settings, get_settings
from tools.command_tools import UnsupportedCommandError, run_named_check
from tools.git_tools import (
    get_changed_paths,
    get_diff_text,
    is_workspace_git_root,
)
from tools.mutation_safety import MutationPolicy, build_mutation_policy
from tools.results import CommandResult

# initial implementation + at most this many repair attempts
MAX_REVIEW_CYCLES = 2

# Final verification the orchestrator ENFORCES itself (never trusting the
# coder's self-reported results). Hard-coded Python profile for this repo.
VERIFICATION_CHECKS = ("pytest", "py_compile")


class DirtyWorktreeError(RuntimeError):
    """Raised when --apply is requested but the worktree is not clean."""


class NotAGitRepoError(RuntimeError):
    """Raised when the workspace is not the root of a git repository."""


def ensure_ready_worktree(settings: Settings) -> None:
    """
    Require the workspace to be a clean git root before mutating anything.

    A clean baseline means every later change is attributable to the agent, and
    makes rollback/inspection well defined.
    """
    if not is_workspace_git_root(settings):
        raise NotAGitRepoError(
            "Workspace is not the root of a git repository; refusing to apply."
        )
    changed = get_changed_paths(settings)
    if changed:
        raise DirtyWorktreeError(
            "Worktree has uncommitted changes; commit or stash first. "
            f"Changed: {', '.join(changed)}"
        )


def run_verification(settings: Settings) -> tuple[CommandResult, ...]:
    """Run the enforced checks (pytest, py_compile) and return their results."""
    results: list[CommandResult] = []
    for check in VERIFICATION_CHECKS:
        try:
            results.append(run_named_check(check, settings))
        except UnsupportedCommandError:
            # e.g. py_compile has no targets — skip rather than fabricate a pass.
            continue
    return tuple(results)


def enforce_run_limits(
    policy: MutationPolicy,
    changed_paths: Sequence[str],
    settings: Settings,
) -> str | None:
    """
    Enforce the run-level mutation limits that single tools cannot.

    Returns a human-readable reason string if a limit is exceeded, else None.
    `max_file_write_chars` is already enforced per-write inside the tools; here
    we enforce the cross-call totals using git's authoritative changed list.
    """
    if len(changed_paths) > policy.max_files_changed:
        return (
            f"Run limit exceeded: {len(changed_paths)} files changed "
            f"> max_files_changed={policy.max_files_changed}."
        )
    total = 0
    root = settings.tool_workspace_root
    for relative in changed_paths:
        path = root / relative
        try:
            total += len(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            continue
    if total > policy.max_total_write_chars:
        return (
            f"Run limit exceeded: {total} changed chars "
            f"> max_total_write_chars={policy.max_total_write_chars}."
        )
    return None


def _apply_coding_to_state(state: AgentState, coding: CodingResult) -> None:
    state.changed_files = list(coding.actual_changed_files)
    state.implementation_iterations += 1
    state.input_tokens += coding.input_tokens
    state.output_tokens += coding.output_tokens


def _format_review_feedback(review: ReviewResult) -> str:
    """Render reviewer issues as actionable text for a repair attempt."""
    lines = [f"Summary: {review.summary}"]
    for issue in review.issues:
        location = issue.path or "(general)"
        lines.append(
            f"- [{issue.severity}] {location}: {issue.description} "
            f"-> {issue.required_change}"
        )
    return "\n".join(lines)


def _failed(
    task: str,
    plan: PlanningResult,
    summary: str,
    input_tokens: int,
    output_tokens: int,
    *,
    changed_files: Sequence[str] = (),
    command_results: Sequence[CommandResult] = (),
    reviews: Sequence[ReviewResult] = (),
    diff: str = "",
) -> ImplementationRunResult:
    return ImplementationRunResult(
        task=task,
        status="failed",
        plan=plan,
        changed_files=tuple(changed_files),
        command_results=tuple(command_results),
        reviews=tuple(reviews),
        diff=diff,
        summary=summary,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def execute_repo_task(
    task: str,
    *,
    apply: bool = False,
    max_review_cycles: int | None = None,
) -> ImplementationRunResult:
    """
    Plan -> (optionally) code -> verify -> review -> bounded repair loop.

    Dry-run (apply=False, the default) stops after planning and changes nothing.
    Apply mode requires a clean git-root worktree.
    """
    settings = get_settings()
    policy = build_mutation_policy(settings)
    max_cycles = (
        max_review_cycles if max_review_cycles is not None else MAX_REVIEW_CYCLES
    )
    if max_cycles <= 0:
        raise ValueError("max_review_cycles must be greater than zero")

    # Planning is read-only and safe in both modes.
    plan = plan_repo_task(task)
    input_tokens = plan.input_tokens
    output_tokens = plan.output_tokens

    if not apply:
        return ImplementationRunResult(
            task=task,
            status="planned",
            plan=plan,
            changed_files=(),
            command_results=(),
            reviews=(),
            diff="",
            summary="Dry run: plan produced, no changes applied.",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # Apply mode from here on.
    ensure_ready_worktree(settings)
    state = AgentState(
        task=task,
        workspace_root=settings.tool_workspace_root,
        plan=plan,
    )

    try:
        coding = implement_repo_task(task, plan)
    except CodingParseError as exc:
        return _failed(task, plan, f"Coder failed: {exc}", input_tokens,
                       output_tokens)
    input_tokens += coding.input_tokens
    output_tokens += coding.output_tokens
    _apply_coding_to_state(state, coding)

    reason = enforce_run_limits(policy, coding.actual_changed_files, settings)
    if reason is not None:
        return _failed(task, plan, reason, input_tokens, output_tokens,
                       changed_files=coding.actual_changed_files)

    reviews: list[ReviewResult] = []
    command_results: tuple[CommandResult, ...] = ()
    diff = ""

    for cycle in range(1, max_cycles + 1):
        command_results = run_verification(settings)
        diff = get_diff_text(settings)
        try:
            review, run = run_reviewer(
                task,
                plan,
                diff,
                command_results,
                changed_files=coding.actual_changed_files,
                coder_summary=coding.summary,
            )
        except ReviewParseError as exc:
            return _failed(task, plan, f"Reviewer failed: {exc}", input_tokens,
                           output_tokens,
                           changed_files=coding.actual_changed_files,
                           command_results=command_results, reviews=reviews,
                           diff=diff)
        input_tokens += run.input_tokens
        output_tokens += run.output_tokens
        reviews.append(review)
        state.review_history.append(review)
        state.review_iterations += 1

        if review.verdict == "approved":
            return ImplementationRunResult(
                task=task,
                status="completed",
                plan=plan,
                changed_files=coding.actual_changed_files,
                command_results=command_results,
                reviews=tuple(reviews),
                diff=diff,
                summary=review.summary,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        if cycle == max_cycles:
            return ImplementationRunResult(
                task=task,
                status="changes_requested",
                plan=plan,
                changed_files=coding.actual_changed_files,
                command_results=command_results,
                reviews=tuple(reviews),
                diff=diff,
                summary="Max review cycles reached; changes still requested.",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        # Repair attempt using the reviewer's exact issues.
        feedback = _format_review_feedback(review)
        try:
            coding = implement_repo_task(task, plan, review_feedback=feedback)
        except CodingParseError as exc:
            return _failed(task, plan, f"Repair failed: {exc}", input_tokens,
                           output_tokens,
                           changed_files=coding.actual_changed_files,
                           command_results=command_results, reviews=reviews,
                           diff=diff)
        input_tokens += coding.input_tokens
        output_tokens += coding.output_tokens
        _apply_coding_to_state(state, coding)

        reason = enforce_run_limits(policy, coding.actual_changed_files, settings)
        if reason is not None:
            return _failed(task, plan, reason, input_tokens, output_tokens,
                           changed_files=coding.actual_changed_files,
                           command_results=command_results, reviews=reviews,
                           diff=diff)

    # Defensive: the loop always returns above.
    raise RuntimeError("execute_repo_task loop exited unexpectedly")
