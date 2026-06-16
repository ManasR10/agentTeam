# DevAgent

DevAgent is an approval-based AI software-engineering system that plans
development tasks, modifies code through controlled tools, runs tests,
reviews its own changes, and produces human-reviewable output.

## Current status

Phase 2: repo-inspection planning agent that turns a task into a grounded,
validated implementation plan using the Phase 1 read-only tools.

## Requirements

- Python 3.12
- Anthropic API key (only for live scripts — see below)

An API key is required for the live scripts (`smoke_test.py`, `phase1_demo.py`,
`phase2_demo.py`). It is **not** required for the offline unit tests or the
read-only file tools, which run without contacting the API.

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

## Current architecture

- validated environment configuration (`config.py`)
- Anthropic SDK client + text-only LLM wrapper (`llm.py`)
- manual tool-use loop: `call_agent_with_tools()` (`llm.py`)
- read-only file tools with path safety (`tools/`)
- token usage + tool-call tracing metadata
- repo-inspection planning agent (`agents/`): prompts, planner, validated
  `PlanningResult`, markdown formatting
- live API smoke test (`smoke_test.py`) and demos (`phase1_demo.py`,
  `phase2_demo.py`)

Tool *writing*, command execution, and multi-agent orchestration will be
introduced in later phases.
