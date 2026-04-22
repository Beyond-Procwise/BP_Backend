-- 2026-04-21-engineered-extraction.sql
-- Adds: anchor_patterns column on bp_extraction_patterns,
--       new extraction_type_priors, extraction_review_queue, extraction_provenance tables,
--       resolution trigger for review queue.

ALTER TABLE proc.bp_extraction_patterns
    ADD COLUMN IF NOT EXISTS anchor_patterns JSONB;

CREATE TABLE IF NOT EXISTS proc.bp_extraction_type_priors (
    doc_type     TEXT PRIMARY KEY,
    priors       JSONB NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS proc.extraction_review_queue (
    id                   BIGSERIAL PRIMARY KEY,
    process_monitor_id   INT REFERENCES proc.process_monitor(id) ON DELETE CASCADE,
    file_path            TEXT NOT NULL,
    doc_type             TEXT NOT NULL,
    partial_header       JSONB,
    partial_line_items   JSONB,
    failed_fields        TEXT[],
    parsed_text          TEXT,
    attempt_count        INT NOT NULL DEFAULT 0,
    last_attempt_at      TIMESTAMPTZ,
    signals_json         JSONB,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at          TIMESTAMPTZ,
    resolved_by          TEXT
);
CREATE INDEX IF NOT EXISTS idx_eq_unresolved ON proc.extraction_review_queue (resolved_at) WHERE resolved_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_eq_doc_type ON proc.extraction_review_queue (doc_type);

CREATE TABLE IF NOT EXISTS proc.bp_extraction_provenance (
    id                  BIGSERIAL PRIMARY KEY,
    parent_table        TEXT NOT NULL,
    parent_pk           TEXT NOT NULL,
    field_name          TEXT NOT NULL,
    source              TEXT NOT NULL,
    anchor_ref          JSONB,
    derivation_trace    JSONB,
    confidence          NUMERIC(3,2),
    attempt             INT,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_prov_parent ON proc.bp_extraction_provenance (parent_table, parent_pk);

CREATE OR REPLACE FUNCTION proc.fn_resolve_extraction_review() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.resolved_at IS NOT NULL AND OLD.resolved_at IS NULL THEN
        UPDATE proc.process_monitor
           SET status='Completed', start_ts=NULL, end_ts=NULL
         WHERE id = NEW.process_monitor_id;
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_resolve_extraction_review ON proc.extraction_review_queue;
CREATE TRIGGER trg_resolve_extraction_review
AFTER UPDATE ON proc.extraction_review_queue
FOR EACH ROW EXECUTE FUNCTION proc.fn_resolve_extraction_review();
