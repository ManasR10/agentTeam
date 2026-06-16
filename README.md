# DevAgent

DevAgent is an approval-based AI software-engineering system that plans
development tasks, modifies code through controlled tools, runs tests,
reviews its own changes, and produces human-reviewable output.

## Current status

Phase 1: manual Anthropic tool-use loop with safe, read-only file tools.

## Requirements

- Python 3.12
- Anthropic API key

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

## Current architecture

- validated environment configuration (`config.py`)
- Anthropic SDK client + text-only LLM wrapper (`llm.py`)
- manual tool-use loop: `call_agent_with_tools()` (`llm.py`)
- read-only file tools with path safety (`tools/`)
- token usage + tool-call tracing metadata
- live API smoke test (`smoke_test.py`) and demo (`phase1_demo.py`)

Tool *writing*, command execution, and multi-agent orchestration will be
introduced in later phases.
