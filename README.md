# NPW

NPW is an approval-based AI software-engineering system that plans
development tasks, modifies code through controlled tools, runs tests,
reviews its own changes, and produces human-reviewable output.

## Current status

Phase 0: project foundation and raw Anthropic API integration.

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
NPW Phase 0 setup OK
```

## Current architecture

Phase 0 contains:

- validated environment configuration
- Anthropic SDK client
- text-only LLM wrapper
- token usage metadata
- live API smoke test

Tool calling and agents will be introduced in later phases.
