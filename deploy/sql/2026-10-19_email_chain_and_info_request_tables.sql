-- Three tables that exist on bp_testdb but are created by no migration or application code:
-- the email dispatch chain, the message-to-RFQ thread map, and the supplier information-request ledger.
-- Definitions are copied from bp_testdb (columns, defaults, constraints, indexes). Idempotent; one transaction.
-- Reply matching (email_dispatch_chains, email_thread_map) is used by the supplier-response and negotiation email flow.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.email_dispatch_chains (
    id                   SERIAL PRIMARY KEY,
    rfq_id               TEXT NOT NULL,
    message_id           TEXT NOT NULL UNIQUE,
    thread_index         INTEGER NOT NULL DEFAULT 1,
    supplier_id          TEXT,
    workflow_ref         TEXT,
    recipients           JSONB,
    subject              TEXT NOT NULL,
    body                 TEXT NOT NULL,
    dispatch_metadata    JSONB,
    awaiting_response    BOOLEAN NOT NULL DEFAULT TRUE,
    responded_at         TIMESTAMPTZ,
    response_message_id  TEXT,
    response_metadata    JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS proc.email_thread_map (
    message_id   TEXT PRIMARY KEY,
    rfq_id       TEXT NOT NULL,
    supplier_id  TEXT,
    recipients   JSONB,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS proc.bp_supplier_info_request (
    request_id            BIGSERIAL PRIMARY KEY,
    requirement_id        TEXT NOT NULL,
    supplier_id           TEXT NOT NULL,
    criterion             TEXT NOT NULL,
    state                 TEXT NOT NULL DEFAULT 'open'
                          CONSTRAINT bp_supplier_info_request_state_check
                          CHECK (state IN ('open','drafted','asked','answered','escalated','closed')),
    attempts              INTEGER NOT NULL DEFAULT 0,
    draft_id              BIGINT,
    draft_unique_id       TEXT,
    last_asked_at         TIMESTAMPTZ,
    answered_value        NUMERIC,
    answered_unit         TEXT,
    answered_response_id  BIGINT,
    needs_review          BOOLEAN NOT NULL DEFAULT FALSE,
    escalated_reason      TEXT,
    created_date          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_modified_date    TIMESTAMPTZ NOT NULL DEFAULT now(),
    prior_unique_ids      TEXT[] NOT NULL DEFAULT '{}'
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_supplier_info_request_gap
    ON proc.bp_supplier_info_request (requirement_id, supplier_id, criterion);
CREATE INDEX IF NOT EXISTS ix_bp_supplier_info_request_state
    ON proc.bp_supplier_info_request (state);
CREATE INDEX IF NOT EXISTS ix_bp_supplier_info_request_draft
    ON proc.bp_supplier_info_request (draft_unique_id);
CREATE INDEX IF NOT EXISTS ix_bp_supplier_info_request_prior_ids
    ON proc.bp_supplier_info_request USING gin (prior_unique_ids);

COMMIT;
