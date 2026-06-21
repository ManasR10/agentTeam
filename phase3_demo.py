from __future__ import annotations

import argparse
import sys

from agents.formatting import format_implementation_result
from agents.orchestrator import (
    DirtyWorktreeError,
    NotAGitRepoError,
    execute_repo_task,
)

# Exit codes (documented contract for scripts/CI):
EXIT_OK = 0           # completed, or dry-run plan produced
EXIT_FAILURE = 1      # implementation / review / test failure (not completed)
EXIT_USAGE = 2        # CLI usage error
EXIT_DIRTY = 3        # refused: worktree not clean / not a git root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase3_demo.py",
        description="DevAgent Phase 3: plan, implement, verify, and review a task.",
    )
    parser.add_argument("task", nargs="*", help="The engineering task to perform.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Actually make changes (requires a clean git worktree).",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only, make no changes (this is the default).",
    )
    parser.add_argument(
        "--show-diff",
        action="store_true",
        help="Print the full git diff, not just a summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    task = " ".join(args.task).strip()
    if not task:
        parser.print_usage()
        print('Provide a task, e.g. python phase3_demo.py "Add a helper" --apply')
        return EXIT_USAGE

    apply = bool(args.apply)  # default (and --dry-run) => False
    mode = "apply" if apply else "dry-run"
    print(f"Running DevAgent Phase 3 ({mode})...")
    print()

    try:
        result = execute_repo_task(task, apply=apply)
    except (DirtyWorktreeError, NotAGitRepoError) as exc:
        print("Refused: the workspace is not ready for --apply.")
        print(f"Reason: {exc}")
        return EXIT_DIRTY
    except Exception as exc:  # noqa: BLE001 - CLI boundary: report, don't crash
        print("Phase 3 run failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        return EXIT_FAILURE

    print(format_implementation_result(result, show_diff=args.show_diff))
    print()

    if result.status in ("completed", "planned"):
        return EXIT_OK
    # failed or changes_requested
    return EXIT_FAILURE


if __name__ == "__main__":
    raise SystemExit(main())
