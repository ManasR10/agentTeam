# DevAgent

DevAgent is an approval-based AI software-engineering system that plans
development tasks, modifies code through controlled tools, runs tests,
reviews its own changes, and produces human-reviewable output.

## Current status

Phase 3: a controlled implementation agent. It plans a task, changes code
through safe tools, runs the tests itself, and reviews the result before
reporting back. Dry-run is the default; mutation requires `--apply` and a clean
git worktree.

## Requirements

- Python 3.12
- Anthropic API key (only for live scripts — see below)

An API key is required for the live scripts (`smoke_test.py`, `phase1_demo.py`,
`phase2_demo.py`, `phase3_demo.py`). It is **not** required for the offline unit
tests or the read-only file tools, which run without contacting the API.

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock.txt
cp .env.example .env
```

Add your Anthropic API key to `.env`.

## Run the smoke test

```bash
python smoke_test.py
```

Expected result:

```text
DevAgent Phase 0 setup OK
```

## Phase 1: Manual tool-use loop

Phase 1 adds a controlled Anthropic tool-use loop.

Available tools:

- `list_files`: lists files under the configured workspace
- `read_file`: reads allowed text files inside the configured workspace

Safety boundaries:

- tools cannot access paths outside `TOOL_WORKSPACE_ROOT`
- `.env` files are blocked
- common noisy directories like `.git`, `.venv`, and `node_modules` are ignored
- large file reads are truncated with metadata
- tool loop iterations are capped by `TOOL_MAX_ITERATIONS`

Run the demo:

```bash
python phase1_demo.py
```

Run tests:

```bash
pytest
```

## Phase 2: Repo-inspection planning agent

Phase 2 adds a planning layer on top of the Phase 1 tool loop. Given an
engineering task, it inspects the repository with the read-only tools and
returns a structured, validated implementation plan.

Flow:

```
plan_repo_task(task)
  -> build planning prompt
  -> call_agent_with_tools(..., system=PLANNER_SYSTEM_PROMPT)
  -> Claude uses list_files / read_file
  -> Claude returns JSON
  -> parse_planning_result() validates it into a PlanningResult
  -> format_planning_result() renders it for the terminal
```

The plan is read-only and planning-only: no `write_file`, `run_command`, or
`run_tests`. The result records which tools were actually called, so it can
prove it inspected real files rather than guessing.

Run the demo (a task argument is required — a bare run just prints usage and
makes no API call):

```bash
python phase2_demo.py "Add a command line interface for the planning agent"
```

To plan against a different project, point the workspace at it:

```bash
TOOL_WORKSPACE_ROOT=/path/to/repo python phase2_demo.py "Add rate limiting"
```

## Phase 3: Implementation agent

Phase 3 lets DevAgent actually change a repository, under tight control. It
plans the task, edits code with safe tools, runs the tests, and has a separate
reviewer judge the result, retrying with the reviewer's feedback if needed.

Flow:

```
execute_repo_task(task, apply=...)
  -> plan_repo_task(task)                  # read-only planning
  -> (dry-run stops here: plan only, no changes)
  -> ensure clean git-root worktree        # refuse otherwise
  -> implement_repo_task(...)              # coder edits files
  -> enforce run-level mutation limits     # max files / total chars
  -> run verification (pytest, py_compile) # the orchestrator, not the model
  -> review_implementation(...)            # approved / changes_requested
  -> approved? done : repair using the reviewer's issues (bounded)
```

Three agents, each with only the tools its role needs:

- **planner** — read-only (`list_files`, `read_file`).
- **coder** — read, write (`create_file`, `replace_in_file`, `write_file`),
  run tests/checks, and inspect git.
- **reviewer** — read-only plus git inspection; it never edits code.

Safety boundaries on top of Phase 1:

- writes are atomic and refuse protected paths (`.env*`, `.git`, lockfiles,
  secrets); editing needs an exact match or the file's last-read hash
- there is no generic shell; every command runs `shell=False` from a fixed argv
- `--apply` requires the workspace to *be* a clean git root, so every change is
  attributable
- the orchestrator runs verification itself and trusts git — not the model —
  for what actually changed
- per-run limits cap how many files and how many characters a single run may
  change (`MAX_FILES_CHANGED`, `MAX_TOTAL_WRITE_CHARS`)

Plan only (the default — safe, makes changes nowhere):

```bash
python phase3_demo.py "Add a count() method to TodoList"
```

Apply changes (needs a clean git worktree); `--show-diff` prints the full diff:

```bash
python phase3_demo.py "Add a count() method to TodoList" --apply --show-diff
```

To run against a different project, point the workspace at it (use an absolute
path — a relative one resolves against the DevAgent project root):

```bash
TOOL_WORKSPACE_ROOT=/path/to/repo python phase3_demo.py "Add rate limiting" --apply
```

Exit codes: `0` completed or dry-run plan, `1` failed or changes still
requested, `2` usage error, `3` refused (worktree not clean / not a git root).

## Current architecture

- validated environment configuration (`config.py`)
- Anthropic SDK client + text-only LLM wrapper (`llm.py`)
- manual tool-use loop: `call_agent_with_tools()` (`llm.py`)
- read-only file tools with path safety (`tools/`)
- token usage + tool-call tracing metadata
- repo-inspection planning agent (`agents/`): prompts, planner, validated
  `PlanningResult`, markdown formatting
- safe file mutation, restricted command, and read-only git tools (`tools/`)
  with a capability-scoped registry that hands each agent only what it needs
- coder, reviewer, and orchestrator agents (`agents/`) with a bounded
  plan -> code -> verify -> review repair loop and run-level mutation limits
- live API smoke test (`smoke_test.py`) and demos (`phase1_demo.py`,
  `phase2_demo.py`, `phase3_demo.py`)

Multi-agent orchestration via a real graph framework, richer language/check
support, and persistent run history are up next.
