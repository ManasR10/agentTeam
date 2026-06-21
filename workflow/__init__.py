"""
Durable workflow engine for DevAgent (Phase 4).

Phase 3 runs a task to completion inside one Python process. This package makes
a run survive the process: lifecycle models, validated state transitions,
persisted state, an append-only event log, a human approval gate, resumable
stages, and run-scoped rollback. The agents in `agents/` do the actual work;
this layer decides when they run, records what happened, and lets a run be
stopped and picked up again later.
"""
