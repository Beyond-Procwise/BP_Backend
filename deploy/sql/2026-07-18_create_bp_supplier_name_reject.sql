-- Rejected supplier-name candidates. Idempotent.
--
-- The resolver drops strings that are not supplier names (table-header
-- fragments, form labels, merged table cells, addresses, ...) before they can
-- be auto-created in proc.bp_supplier. Nothing is silently discarded: every
-- rejection is recorded here with the LITERAL extracted value and the rule
-- that fired, so a false positive is visible and recoverable.
--
-- This table never rewrites a document value; it is an audit log only.
CREATE TABLE IF NOT EXISTS proc.bp_supplier_name_reject (
    reject_id      BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    first_seen     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen      TIMESTAMPTZ NOT NULL DEFAULT now(),
    seen_count     INTEGER     NOT NULL DEFAULT 1,
    extracted_name TEXT        NOT NULL,   -- literal value from the document
    reason         TEXT        NOT NULL,   -- rule code that rejected it
    doc_type       TEXT,
    doc_pk         TEXT,
    trace_id       TEXT,
    status         TEXT        NOT NULL DEFAULT 'rejected'  -- rejected|overturned
);

-- One row per distinct literal value; repeats bump seen_count.
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_supplier_name_reject_name
    ON proc.bp_supplier_name_reject (LOWER(extracted_name));
CREATE INDEX IF NOT EXISTS ix_bp_supplier_name_reject_reason
    ON proc.bp_supplier_name_reject (reason);
