-- email_agent: what each assured draft rested on, and what happened to it.
--
-- Apply to bp_testdb first. New schema, new tables; touches nothing existing.
-- Reversible: 2026-10-07_email_agent_capture_rollback.sql (drops the schema CASCADE,
-- which is safe only while nothing outside this schema references it).
--
-- WHAT IS DELIBERATELY NOT HERE: the text that was sent. The style package already
-- decided a score is enough to know a draft was wrong and that retaining sent mail
-- is the exception that gets forgotten. What is stored instead is the MODEL'S own
-- draft (needed to measure the edit later) and, per send, the figures that changed.

BEGIN;

CREATE SCHEMA IF NOT EXISTS email_agent;

CREATE TABLE IF NOT EXISTS email_agent.bp_draft_capture (
    capture_id          BIGSERIAL PRIMARY KEY,
    unique_id           TEXT        NOT NULL,
    workflow_id         TEXT,
    supplier_id         TEXT,
    path                TEXT,                 -- NEGOTIATION_COUNTER | PROMPT_COMPOSE
    family_id           TEXT        NOT NULL,
    family_version      INTEGER,
    mode                TEXT,                 -- shadow | enforce, as at drafting time
    assurance_status    TEXT        NOT NULL, -- verified | needs_review | unassured
    request_text        TEXT,                 -- the person's own instruction, when there was one
    facts               JSONB,                -- value + table, column, row_id, retrieved_at
    carried_unverified  JSONB,
    conflicts           JSONB,
    reasoned            JSONB,                -- value + basis
    assumptions         JSONB,
    unverified_figures  JSONB,
    violations          JSONB,
    repaired            BOOLEAN,
    draft_text          TEXT        NOT NULL, -- the model's output, not anything sent
    draft_hash          TEXT        NOT NULL,
    user_id             TEXT,
    team_id             TEXT,                 -- no source today (bp_role_assignment is empty)
    captured_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_bp_draft_capture_unique
    ON email_agent.bp_draft_capture (unique_id, captured_at DESC);

CREATE TABLE IF NOT EXISTS email_agent.bp_draft_outcome (
    outcome_id          BIGSERIAL PRIMARY KEY,
    capture_id          BIGINT      NOT NULL REFERENCES email_agent.bp_draft_capture (capture_id),
    outcome             TEXT        NOT NULL CHECK (outcome IN ('sent', 'abandoned')),
    edit_distance       NUMERIC(4,3),         -- word level, 0 identical .. 1 total rewrite
    edit_class          TEXT,                 -- none | fact | reasoned | figure_other | wording
    removed_figures     JSONB,                -- [{value, class}] in the draft, not in what was sent
    added_figures       JSONB,                -- [{value, class}] in what was sent, not in the draft
    drafted_words       INTEGER,
    sent_words          INTEGER,
    regeneration_count  INTEGER     NOT NULL DEFAULT 0,
    time_to_send_s      INTEGER,
    supplier_replied    BOOLEAN,
    reply_latency_s     INTEGER,
    issue_resolved      BOOLEAN,
    recorded_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- A draft is sent once. A second 'sent' row for the same capture is a bug, not data.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_draft_outcome_sent_once
    ON email_agent.bp_draft_outcome (capture_id) WHERE outcome = 'sent';

COMMIT;
