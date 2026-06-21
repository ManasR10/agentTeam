from __future__ import annotations

import argparse
import sys

from agents.orchestrator import DirtyWorktreeError, NotAGitRepoError
from config import get_settings
from storage.base import RunNotFoundError
from storage.sqlite_store import SqliteRunStore
from tools.git_tools import get_diff_text
from workflow.approvals import ApprovalError
from workflow.errors import RunNotResumableError
from workflow.models import RunStatus, WorkflowRun
from workflow.rollback import RollbackError, rollback_run
from workflow.service import WorkflowService

# Exit codes (documented contract for scripts/CI):
EXIT_OK = 0            # command succeeded (incl. a run paused for approval)
EXIT_FAILURE = 1       # run failed, or changes still requested
EXIT_USAGE = 2         # CLI usage error
EXIT_REFUSED = 3       # operation refused (dirty repo, bad approval state, ...)
EXIT_NOT_FOUND = 4     # run id not found
EXIT_APPROVAL = 5      # resume blocked: plan needs approval first
EXIT_NOT_RESUMABLE = 6 # run is finished and cannot be resumed
EXIT_ROLLBACK = 7      # rollback refused or failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devagent.py",
        description="DevAgent Phase 4: durable, approval-gated workflow runs.",
    )
    sub = parser.add_subparsers(dest="command")

    p_start = sub.add_parser("start", help="Plan a new task (stops for approval).")
    p_start.add_argument("task", nargs="+", help="The engineering task.")

    p_show = sub.add_parser("show", help="Show a run in detail.")
    p_show.add_argument("run_id")

    p_approve = sub.add_parser("approve", help="Approve a run's plan.")
    p_approve.add_argument("run_id")
    p_approve.add_argument("--comment", default=None)

    p_reject = sub.add_parser("reject", help="Reject a run's plan.")
    p_reject.add_argument("run_id")
    p_reject.add_argument("--comment", default=None)

    p_resume = sub.add_parser("resume", help="Advance a run to the next gate.")
    p_resume.add_argument("run_id")

    p_list = sub.add_parser("list", help="List recent runs.")
    p_list.add_argument("--status", default=None)
    p_list.add_argument("--limit", type=int, default=20)

    p_events = sub.add_parser("events", help="Show a run's audit events.")
    p_events.add_argument("run_id")

    p_diff = sub.add_parser("diff", help="Show the current git diff for a run.")
    p_diff.add_argument("run_id")

    p_rollback = sub.add_parser("rollback", help="Undo a run's file changes.")
    p_rollback.add_argument("run_id")
    p_rollback.add_argument(
        "--yes", action="store_true", help="Skip the confirmation prompt."
    )

    return parser


def main(argv: list[str] | None = None, *, service: WorkflowService | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_usage()
        return EXIT_USAGE

    owns_service = service is None
    svc = service or _build_service()
    try:
        return _dispatch(args, svc)
    except RunNotFoundError as exc:
        print(f"Run not found: {exc}")
        return EXIT_NOT_FOUND
    except (DirtyWorktreeError, NotAGitRepoError) as exc:
        print(f"Refused: {exc}")
        return EXIT_REFUSED
    except ApprovalError as exc:
        print(f"Refused: {exc}")
        return EXIT_REFUSED
    except RunNotResumableError as exc:
        print(f"Not resumable: {exc}")
        return EXIT_NOT_RESUMABLE
    except RollbackError as exc:
        print(f"Rollback refused: {exc}")
        return EXIT_ROLLBACK
    finally:
        if owns_service:
            svc.store.close()


def _build_service() -> WorkflowService:
    settings = get_settings()
    store = SqliteRunStore(settings.devagent_database_path)
    return WorkflowService(store, settings=settings)


def _dispatch(args: argparse.Namespace, svc: WorkflowService) -> int:
    if args.command == "start":
        return _cmd_start(args, svc)
    if args.command == "show":
        print(_format_run(svc.store.load_run(args.run_id)))
        return EXIT_OK
    if args.command == "approve":
        run = svc.approve_run(args.run_id, comment=args.comment)
        print(f"Approved. Status: {run.status.value}")
        print(f"Run the coder with: python devagent.py resume {run.run_id}")
        return EXIT_OK
    if args.command == "reject":
        run = svc.reject_run(args.run_id, comment=args.comment)
        print(f"Rejected. Status: {run.status.value}")
        return EXIT_OK
    if args.command == "resume":
        return _cmd_resume(args, svc)
    if args.command == "list":
        return _cmd_list(args, svc)
    if args.command == "events":
        return _cmd_events(args, svc)
    if args.command == "diff":
        return _cmd_diff(args, svc)
    if args.command == "rollback":
        return _cmd_rollback(args, svc)
    return EXIT_USAGE


def _cmd_start(args: argparse.Namespace, svc: WorkflowService) -> int:
    task = " ".join(args.task).strip()
    run = svc.start_run(task)
    print(f"Run ID: {run.run_id}")
    print(f"Status: {run.status.value}")
    if run.plan is not None:
        print("Plan:")
        for i, step in enumerate(run.plan.implementation_plan, 1):
            print(f"  {i}. {step}")
    print("No files have been modified.")
    if run.status is RunStatus.AWAITING_PLAN_APPROVAL:
        print(f"Approve with: python devagent.py approve {run.run_id}")
        return EXIT_OK
    return EXIT_FAILURE  # planning failed


def _cmd_resume(args: argparse.Namespace, svc: WorkflowService) -> int:
    run = svc.resume_run(args.run_id)
    print(f"Status: {run.status.value}")
    if run.status is RunStatus.COMPLETED:
        return EXIT_OK
    if run.status is RunStatus.AWAITING_PLAN_APPROVAL:
        print("This run needs plan approval before it can proceed.")
        return EXIT_APPROVAL
    if run.status in (RunStatus.FAILED, RunStatus.CHANGES_REQUESTED):
        return EXIT_FAILURE
    return EXIT_OK


def _cmd_list(args: argparse.Namespace, svc: WorkflowService) -> int:
    status = RunStatus(args.status) if args.status else None
    for summary in svc.store.list_runs(limit=args.limit, status=status):
        print(f"{summary.run_id}  {summary.status.value:<22}  {summary.task}")
    return EXIT_OK


def _cmd_events(args: argparse.Namespace, svc: WorkflowService) -> int:
    for event in svc.store.list_events(args.run_id):
        stage = event.stage or ""
        print(f"{event.sequence_number:>3}  {event.event_type.value:<20}  {stage}")
    return EXIT_OK


def _cmd_diff(args: argparse.Namespace, svc: WorkflowService) -> int:
    run = svc.store.load_run(args.run_id)
    if run.workspace_root != str(svc.settings.tool_workspace_root):
        print("Workspace differs from this run; cannot show a meaningful diff.")
        return EXIT_REFUSED
    print(get_diff_text(svc.settings) or "(no changes)")
    return EXIT_OK


def _cmd_rollback(args: argparse.Namespace, svc: WorkflowService) -> int:
    if not args.yes:
        answer = input(f"Roll back all files changed by run {args.run_id}? [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted.")
            return EXIT_OK
    rollback_run(args.run_id, svc.store, svc.settings)
    print("Rolled back.")
    return EXIT_OK


def _format_run(run: WorkflowRun) -> str:
    lines = [
        f"Run ID: {run.run_id}",
        f"Task: {run.task}",
        f"Status: {run.status.value}",
        f"Created: {run.created_at.isoformat()}",
        f"Updated: {run.updated_at.isoformat()}",
    ]
    if run.plan is not None:
        lines.append(f"Plan: {len(run.plan.implementation_plan)} step(s)")
    if run.approval is not None:
        lines.append(f"Approval: {run.approval.decision.value}")
    lines.append(f"Coding attempts: {len(run.coding_attempts)}")
    lines.append(f"Verification runs: {len(run.verification_runs)}")
    lines.append(f"Review attempts: {len(run.review_attempts)}")
    if run.changed_files:
        lines.append("Changed files: " + ", ".join(run.changed_files))
    if run.review_attempts:
        lines.append(f"Latest verdict: {run.review_attempts[-1].result.verdict}")
    lines.append(
        f"Tokens: {run.total_input_tokens} in / {run.total_output_tokens} out"
    )
    if run.last_error is not None:
        lines.append(
            f"Last error ({run.last_error.stage}): "
            f"{run.last_error.error_type}: {run.last_error.message}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
