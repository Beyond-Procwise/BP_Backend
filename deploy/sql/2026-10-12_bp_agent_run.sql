-- One row per reasoning run of a created agent: the system prompt as sent, the
-- model, each tool call with what it was asked and what it returned, and the
-- answer. The server creates this itself at startup
-- (repositories/agent_run_repo.py, same DDL). Additive; safe to run twice.

CREATE SCHEMA IF NOT EXISTS proc;

CREATE TABLE IF NOT EXISTS proc.bp_agent_run (
    run_id         TEXT PRIMARY KEY,
    agent_slug     TEXT NOT NULL,
    workflow_id    TEXT,
    user_id        TEXT,
    model          TEXT,
    system_prompt  TEXT NOT NULL,
    system_sha256  TEXT NOT NULL,
    task           TEXT NOT NULL,
    tools_offered  JSONB NOT NULL,
    calls          JSONB NOT NULL,
    answer         TEXT,
    outputs        JSONB,
    rounds         INTEGER NOT NULL DEFAULT 0,
    error          TEXT,
    started_at     TIMESTAMPTZ NOT NULL,
    finished_at    TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_bp_agent_run_agent
    ON proc.bp_agent_run (agent_slug, started_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_agent_run_workflow
    ON proc.bp_agent_run (workflow_id);
