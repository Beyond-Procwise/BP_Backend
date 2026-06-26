-- Fine-grained, append-only agent action event log.
-- One row per action/step across extraction / validation / consolidation phases.
CREATE TABLE IF NOT EXISTS proc.bp_agent_actions (
    action_id          bigserial PRIMARY KEY,
    created_at         timestamptz NOT NULL DEFAULT now(),
    deal_id            text,
    document_id        text,
    doc_pk             text,
    doc_type           text,
    process_monitor_id integer,
    trace_id           text,
    phase              text NOT NULL,
    action_type        text NOT NULL,
    agent              text,
    field_name         text,
    status             text,
    summary            text,
    details            jsonb,
    confidence         numeric,
    pipeline_version   text
);

CREATE INDEX IF NOT EXISTS ix_bp_agent_actions_deal     ON proc.bp_agent_actions (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_agent_actions_document ON proc.bp_agent_actions (document_id);
CREATE INDEX IF NOT EXISTS ix_bp_agent_actions_doc_pk   ON proc.bp_agent_actions (doc_pk);
CREATE INDEX IF NOT EXISTS ix_bp_agent_actions_trace    ON proc.bp_agent_actions (trace_id);
CREATE INDEX IF NOT EXISTS ix_bp_agent_actions_created  ON proc.bp_agent_actions (created_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_agent_actions_phase    ON proc.bp_agent_actions (phase);
