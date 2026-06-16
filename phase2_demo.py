from __future__ import annotations

import sys

from agents.formatting import format_planning_result
from agents.planner import plan_repo_task

USAGE = (
    'Usage: python phase2_demo.py "<your task>"\n'
    'Example: python phase2_demo.py "Add user login"'
)


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    task = " ".join(args).strip() if args else ""

    # No task -> show usage and exit WITHOUT making a (paid) API call. A bare
    # run should never silently spend tokens on a placeholder task.
    if not task:
        print(USAGE)
        return 2

    print("Running DevAgent Phase 2 planning demo...")
    print()
    print("Task:")
    print(task)
    print()

    try:
        result = plan_repo_task(task)
    except Exception as exc:
        print("Phase 2 demo failed.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        return 1

    print(format_planning_result(result, include_usage=True))
    print()

    # Grounding evidence: show that the planner actually inspected files
    # (read_file / list_files calls) rather than guessing contents.
    print(f"## Tools used ({result.iterations} iterations)")
    if result.tool_calls:
        for call in result.tool_calls:
            print(f"- {call}")
    else:
        print("- None (planner did not inspect any files)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
