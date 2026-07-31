-- deploy/sql/2026-08-01_bp_extraction_verdict.sql
-- What a human decided about a value we extracted.
--
-- bp_extraction_discrepancy already records resolved_value / resolution_action / resolved_by,
-- but those describe the FINDING. This records the judgement on the VALUE, joined to the
-- reader that produced it (proc.bp_extraction_provenance), which is what makes an accuracy
-- rate computable per reader rather than per document.
--
-- Additive + idempotent.
BEGIN;

CREATE TABLE IF NOT EXISTS proc.bp_extraction_verdict (
    verdict_id       BIGSERIAL PRIMARY KEY,
    doc_type         TEXT        NOT NULL,
    doc_pk           TEXT        NOT NULL,
    field_name       TEXT        NOT NULL,
    -- Copied from provenance at verdict time so the rate survives a provenance purge and
    -- stays correct if the same field is later re-extracted by a different reader.
    source           TEXT,
    pattern_name     TEXT,
    prior_confidence NUMERIC,
    verdict          TEXT        NOT NULL CHECK (verdict IN ('confirmed', 'corrected', 'rejected')),
    extracted_value  TEXT,
    corrected_value  TEXT,
    decided_by       TEXT,
    decided_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE proc.bp_extraction_verdict IS
    'One row per human judgement on an extracted value. confirmed = a human saw it and let '
    'it stand; corrected = a human replaced it (the strongest negative signal there is); '
    'rejected = the finding was dismissed as not a real problem, which is a vote FOR the '
    'extracted value, not against it.';

CREATE INDEX IF NOT EXISTS ix_bp_extraction_verdict_doc_type_field
    ON proc.bp_extraction_verdict (doc_type, field_name);
CREATE INDEX IF NOT EXISTS ix_bp_extraction_verdict_source_pattern
    ON proc.bp_extraction_verdict (source, pattern_name);

COMMIT;
