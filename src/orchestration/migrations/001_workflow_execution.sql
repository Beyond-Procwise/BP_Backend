-- src/orchestration/migrations/001_workflow_execution.sql
-- Orchestration rearchitecture: durable workflow state tables
-- Spec: docs/superpowers/specs/2026-03-31-orchestration-rearchitecture-design.md Section 5

BEGIN;

-- Workflow execution state (replaces Redis-based checkpoints)
CREATE TABLE IF NOT EXISTS proc.workflow_execution (
    execution_id    SERIAL PRIMARY KEY,
    workflow_id     TEXT NOT NULL,
    workflow_name   TEXT NOT NULL,
    user_id         TEXT,
    status          TEXT NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending','running','paused','completed','failed','cancelled')),
    shared_data     JSONB NOT NULL DEFAULT '{}',
    current_round   INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at    TIMESTAMPTZ
);

-- Unique active workflow constraint: only one non-terminal execution per workflow_id
CREATE UNIQUE INDEX IF NOT EXISTS uix_workflow_execution_active
    ON proc.workflow_execution (workflow_id)
    WHERE status NOT IN ('completed', 'failed', 'cancelled');

CREATE INDEX IF NOT EXISTS ix_workflow_execution_status
    ON proc.workflow_execution (status);

-- Node execution state (one row per node per round)
CREATE TABLE IF NOT EXISTS proc.node_execution (
    node_execution_id   SERIAL PRIMARY KEY,
    execution_id        INTEGER NOT NULL REFERENCES proc.workflow_execution(execution_id),
    node_name           TEXT NOT NULL,
    agent_type          TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','ready','running','completed','failed','skipped','timed_out')),
    attempt             INTEGER NOT NULL DEFAULT 0,
    input_data          JSONB,
    output_data         JSONB,
    pass_fields         JSONB,
    error               TEXT,
    dispatched_at       TIMESTAMPTZ,
    completed_at        TIMESTAMPTZ,
    duration_ms         INTEGER,
    round               INTEGER NOT NULL DEFAULT 0,
    UNIQUE (execution_id, node_name, round)
);

CREATE INDEX IF NOT EXISTS ix_node_execution_status
    ON proc.node_execution (execution_id, status);

CREATE INDEX IF NOT EXISTS ix_node_execution_stale
    ON proc.node_execution (status, dispatched_at)
    WHERE status IN ('running', 'ready');

-- Workflow events (observability + audit trail)
CREATE TABLE IF NOT EXISTS proc.workflow_events (
    event_id        SERIAL PRIMARY KEY,
    workflow_id     TEXT NOT NULL,
    node_name       TEXT,
    event_type      TEXT NOT NULL,
    agent_type      TEXT,
    payload         JSONB NOT NULL DEFAULT '{}',
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT now(),
    round           INTEGER
);

CREATE INDEX IF NOT EXISTS ix_workflow_events_workflow
    ON proc.workflow_events (workflow_id, timestamp);

CREATE INDEX IF NOT EXISTS ix_workflow_events_type
    ON proc.workflow_events (event_type, timestamp);

COMMIT;
