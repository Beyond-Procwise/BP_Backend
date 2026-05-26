-- 2026-05-26 schema changes (applied live to bp_sqldb)

-- 1. Quote USD conversion columns (mirror invoice/PO) so quotes get USD computed.
ALTER TABLE proc.bp_quote_raw ADD COLUMN IF NOT EXISTS exchange_rate_to_usd numeric;
ALTER TABLE proc.bp_quote_raw ADD COLUMN IF NOT EXISTS converted_amount_usd numeric;
ALTER TABLE proc.bp_quote_stg ADD COLUMN IF NOT EXISTS exchange_rate_to_usd numeric;
ALTER TABLE proc.bp_quote_stg ADD COLUMN IF NOT EXISTS converted_amount_usd numeric;

-- 2. Extraction telemetry: per-document quality, pattern and gap capture.
CREATE TABLE IF NOT EXISTS proc.bp_extraction_telemetry (
    telemetry_id        bigserial PRIMARY KEY,
    captured_at         timestamptz NOT NULL DEFAULT now(),
    process_monitor_id  integer,
    doc_type            text,
    file_path           text,
    vendor_hint         text,
    doc_pk              text,
    status              text,
    completeness_status text,
    confidence          numeric,
    header_fields       integer,
    line_items          integer,
    n_discrepancies     integer,
    discrepancy_types   jsonb,
    missing_required    text,
    currency            text,
    converted_amount_usd numeric,
    parser_backend      text,
    page_count          integer,
    pipeline_version    text,
    trace_id            text,
    error_detail        text,
    notes               text
);
CREATE INDEX IF NOT EXISTS ix_bp_extr_telemetry_captured ON proc.bp_extraction_telemetry (captured_at DESC);
CREATE INDEX IF NOT EXISTS ix_bp_extr_telemetry_doctype  ON proc.bp_extraction_telemetry (doc_type, completeness_status);
