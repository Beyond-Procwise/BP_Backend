-- scripts/migrations/2026-05-08-extraction-provenance-v3.sql
-- Pipeline V3 provenance: one row per committed field, with the bbox + evidence_text
-- substring + model + judge actions that produced it. Coexists with the legacy
-- proc.bp_extraction_provenance (different schema, AgentNick-era).

CREATE TABLE IF NOT EXISTS proc.bp_extraction_provenance_v3 (
    provenance_id    BIGSERIAL PRIMARY KEY,
    doc_type         TEXT NOT NULL,
    doc_pk           TEXT NOT NULL,
    field_path       TEXT NOT NULL,
    value            TEXT NOT NULL,
    page             INT NOT NULL,
    bbox_x0          REAL NOT NULL,
    bbox_y0          REAL NOT NULL,
    bbox_x1          REAL NOT NULL,
    bbox_y1          REAL NOT NULL,
    evidence_text    TEXT NOT NULL,
    model            TEXT NOT NULL,
    model_confidence REAL NOT NULL,
    judge_actions    JSONB,
    final_confidence REAL NOT NULL,
    extracted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pipeline_version TEXT NOT NULL,
    UNIQUE (doc_type, doc_pk, field_path, extracted_at)
);

CREATE INDEX IF NOT EXISTS idx_provenance_v3_doc
    ON proc.bp_extraction_provenance_v3 (doc_type, doc_pk);
