-- Agent action event log. Replaces the never-created proc.action table.
-- action_id is a client-generated UUID (text) so existing callers need no
-- change beyond the table rename. process_id maps to the integer process PK.
-- process_output and action_desc are stored as JSONB-compatible text so the
-- Python layer can serialise arbitrary payloads without a strict schema.
-- updated_at is set by the UPDATE path; action_date is set on INSERT.
-- Idempotent.

CREATE TABLE IF NOT EXISTS proc.bp_action (
    action_id       TEXT        PRIMARY KEY,
    process_id      BIGINT      NOT NULL,
    run_id          TEXT,
    agent_type      TEXT        NOT NULL,
    process_output  TEXT,
    status          TEXT        NOT NULL DEFAULT 'validated',
    action_desc     TEXT,
    action_date     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS ix_bp_action_process_id
    ON proc.bp_action (process_id);
CREATE INDEX IF NOT EXISTS ix_bp_action_action_date
    ON proc.bp_action (action_date DESC);
