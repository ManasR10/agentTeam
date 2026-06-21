-- Phase 4 durable workflow schema. Applied with executescript on every store
-- open; every statement is IF NOT EXISTS so it is safe to run repeatedly.

CREATE TABLE IF NOT EXISTS workflow_runs (
    run_id          TEXT PRIMARY KEY,
    task            TEXT NOT NULL,
    workspace_root  TEXT NOT NULL,
    status          TEXT NOT NULL,
    current_stage   TEXT,
    -- The full validated run, as written by workflow.serialization. The columns
    -- above are denormalised copies kept for cheap listing and filtering.
    state_json      TEXT NOT NULL,
    schema_version  INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_workflow_runs_status
    ON workflow_runs (status);

CREATE TABLE IF NOT EXISTS workflow_events (
    event_id        TEXT PRIMARY KEY,
    run_id          TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    event_type      TEXT NOT NULL,
    stage           TEXT,
    agent_name      TEXT,
    payload_json    TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES workflow_runs (run_id),
    -- Per-run sequence numbers are unique, so the append path can't write two
    -- events at the same position even under a race.
    UNIQUE (run_id, sequence_number)
);

CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    run_id      TEXT NOT NULL,
    decision    TEXT NOT NULL,
    plan_hash   TEXT NOT NULL,
    comment     TEXT,
    created_at  TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES workflow_runs (run_id)
);
