-- scripts/migrations/2026-05-12-bp_sqldb-two-stage-init.sql
-- Two-stage extraction init for bp_sqldb.
-- Phase 1: _raw tables (JSONB), discrepancy table, and all supporting tables
-- that exist in uicanvas but not yet in bp_sqldb.
--
-- Run against: bp_sqldb (NOT uicanvas — uicanvas is preserved as archive).

-- ============================================================
-- 0. Ensure proc schema exists
-- ============================================================
CREATE SCHEMA IF NOT EXISTS proc;

-- ============================================================
-- 1. _raw tables — JSONB-based first-stage landing zone
--    Captures whatever the extraction pipeline yielded,
--    regardless of schema.  Promotion to _stg happens after
--    validation.  Clean rows are deleted from _raw after
--    promotion; discrepancy rows stay with status='discrepancy'.
-- ============================================================

CREATE TABLE IF NOT EXISTS proc.bp_invoice_raw (
    raw_id              BIGSERIAL PRIMARY KEY,
    doc_pk_candidate    TEXT,            -- best-effort invoice_id; nullable
    source_file         TEXT NOT NULL,
    raw_payload         JSONB NOT NULL,  -- header + line_items + every extracted field
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pipeline_version    TEXT NOT NULL,
    promotion_status    TEXT NOT NULL DEFAULT 'pending'
        CHECK (promotion_status IN ('pending', 'promoted', 'discrepancy'))
);
CREATE INDEX IF NOT EXISTS idx_invoice_raw_status  ON proc.bp_invoice_raw (promotion_status);
CREATE INDEX IF NOT EXISTS idx_invoice_raw_doc_pk  ON proc.bp_invoice_raw (doc_pk_candidate);

CREATE TABLE IF NOT EXISTS proc.bp_purchase_order_raw (
    raw_id              BIGSERIAL PRIMARY KEY,
    doc_pk_candidate    TEXT,
    source_file         TEXT NOT NULL,
    raw_payload         JSONB NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pipeline_version    TEXT NOT NULL,
    promotion_status    TEXT NOT NULL DEFAULT 'pending'
        CHECK (promotion_status IN ('pending', 'promoted', 'discrepancy'))
);
CREATE INDEX IF NOT EXISTS idx_po_raw_status       ON proc.bp_purchase_order_raw (promotion_status);
CREATE INDEX IF NOT EXISTS idx_po_raw_doc_pk       ON proc.bp_purchase_order_raw (doc_pk_candidate);

CREATE TABLE IF NOT EXISTS proc.bp_quote_raw (
    raw_id              BIGSERIAL PRIMARY KEY,
    doc_pk_candidate    TEXT,
    source_file         TEXT NOT NULL,
    raw_payload         JSONB NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pipeline_version    TEXT NOT NULL,
    promotion_status    TEXT NOT NULL DEFAULT 'pending'
        CHECK (promotion_status IN ('pending', 'promoted', 'discrepancy'))
);
CREATE INDEX IF NOT EXISTS idx_quote_raw_status    ON proc.bp_quote_raw (promotion_status);
CREATE INDEX IF NOT EXISTS idx_quote_raw_doc_pk    ON proc.bp_quote_raw (doc_pk_candidate);

CREATE TABLE IF NOT EXISTS proc.bp_contract_raw (
    raw_id              BIGSERIAL PRIMARY KEY,
    doc_pk_candidate    TEXT,
    source_file         TEXT NOT NULL,
    raw_payload         JSONB NOT NULL,
    extracted_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    pipeline_version    TEXT NOT NULL,
    promotion_status    TEXT NOT NULL DEFAULT 'pending'
        CHECK (promotion_status IN ('pending', 'promoted', 'discrepancy'))
);
CREATE INDEX IF NOT EXISTS idx_contract_raw_status ON proc.bp_contract_raw (promotion_status);
CREATE INDEX IF NOT EXISTS idx_contract_raw_doc_pk ON proc.bp_contract_raw (doc_pk_candidate);

-- ============================================================
-- 2. Discrepancy table
--    Rows written here when a _raw record fails validation.
--    The _raw row stays with promotion_status='discrepancy';
--    this table holds one row per flagged field/issue.
-- ============================================================

CREATE TABLE IF NOT EXISTS proc.bp_extraction_discrepancy (
    discrepancy_id      BIGSERIAL PRIMARY KEY,
    doc_type            TEXT NOT NULL,
    raw_id              BIGINT,          -- FK to proc.bp_<type>_raw (soft ref, no CASCADE)
    source_file         TEXT NOT NULL,
    doc_pk_candidate    TEXT,
    field_name          TEXT NOT NULL,
    raw_value           TEXT,            -- value as extracted
    expected_value      TEXT,            -- if a check predicted something different
    computed_value      TEXT,            -- if validation tried to derive a value
    issue_type          TEXT NOT NULL,   -- 'missing_required' | 'type_bind_error' |
                                         -- 'invariant_failed' | 'no_evidence' |
                                         -- 'value_out_of_range' | 'classifier_rejected' |
                                         -- 'date_invalid' | 'amount_mismatch'
    severity            TEXT NOT NULL    -- 'critical' | 'warning' | 'info'
        CHECK (severity IN ('critical', 'warning', 'info')),
    status              TEXT NOT NULL DEFAULT 'open'
        CHECK (status IN ('open', 'resolved', 'ignored')),
    notes               TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at         TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_discrep_doctype ON proc.bp_extraction_discrepancy (doc_type, status);
CREATE INDEX IF NOT EXISTS idx_discrep_raw_id  ON proc.bp_extraction_discrepancy (raw_id);

-- ============================================================
-- 3. Supporting tables — copied DDL from uicanvas
-- ============================================================

-- 3a. bp_supplier
CREATE TABLE IF NOT EXISTS proc.bp_supplier (
    supplier_id               TEXT,
    supplier_name             TEXT,
    trading_name              TEXT,
    supplier_type             TEXT,
    legal_structure           CHARACTER VARYING(10),
    tax_id                    CHARACTER VARYING(20),
    vat_number                CHARACTER VARYING(20),
    duns_number               CHARACTER VARYING(20),
    parent_company_id         TEXT,
    registered_country        CHARACTER VARYING(50),
    registration_number       CHARACTER VARYING(50),
    is_preferred_supplier     BOOLEAN,
    risk_score                CHARACTER VARYING(10),
    credit_limit_amount       NUMERIC(18,2),
    esg_cert_iso14001         BOOLEAN,
    esg_cert_sa8000           BOOLEAN,
    esg_cert_ecovadis         BOOLEAN,
    diversity_women_owned     BOOLEAN,
    diversity_minority_owned  BOOLEAN,
    diversity_veteran_owned   BOOLEAN,
    insurance_coverage_type   CHARACTER VARYING(30),
    insurance_coverage_amount NUMERIC(18,2),
    insurance_expiry_date     DATE,
    bank_name                 TEXT,
    bank_account_number       CHARACTER VARYING(30),
    bank_swift                CHARACTER VARYING(30),
    bank_iban                 CHARACTER VARYING(30),
    default_currency          CHARACTER VARYING(5),
    incoterms                 CHARACTER VARYING(5),
    delivery_lead_time_days   CHARACTER VARYING(3),
    address_line1             TEXT,
    address_line2             TEXT,
    city                      TEXT,
    postal_code               TEXT,
    country                   TEXT,
    website_url               TEXT,
    edi_enabled               BOOLEAN,
    api_enabled               BOOLEAN,
    ariba_integrated          BOOLEAN,
    contact_name_1            TEXT,
    contact_role_1            TEXT,
    contact_email_1           TEXT,
    contact_phone_1           TEXT,
    contact_name_2            TEXT,
    contact_role_2            TEXT,
    contact_email_2           TEXT,
    contact_phone_2           TEXT,
    created_date              TIMESTAMP WITHOUT TIME ZONE,
    created_by                TEXT,
    last_modified_by          TEXT,
    last_modified_date        TIMESTAMP WITHOUT TIME ZONE
);

-- 3b. process_monitor (with LISTEN/NOTIFY trigger)
CREATE SEQUENCE IF NOT EXISTS proc.process_monitor_id_seq;

CREATE TABLE IF NOT EXISTS proc.process_monitor (
    id                INTEGER NOT NULL DEFAULT nextval('proc.process_monitor_id_seq') PRIMARY KEY,
    process_name      CHARACTER VARYING(100),
    type              CHARACTER VARYING(50),
    status            CHARACTER VARYING(50),
    file_path         TEXT,
    start_ts          TIMESTAMP WITHOUT TIME ZONE,
    created_date      TIMESTAMP WITHOUT TIME ZONE,
    created_by        CHARACTER VARYING(50),
    lastmodified_date TIMESTAMP WITHOUT TIME ZONE,
    end_ts            TIMESTAMP WITHOUT TIME ZONE,
    category          TEXT,
    document_type     TEXT,
    user_id           INTEGER,
    total_count       INTEGER
);

CREATE OR REPLACE FUNCTION proc.notify_process_monitor_ready() RETURNS TRIGGER AS $$
BEGIN
    IF NEW.status IN ('Completed', 'Running') AND NEW.id IS NOT NULL THEN
        PERFORM pg_notify('process_monitor_ready', NEW.id::text);
    END IF;
    RETURN NEW;
END
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_process_monitor_ready ON proc.process_monitor;
CREATE TRIGGER trg_process_monitor_ready
AFTER INSERT OR UPDATE ON proc.process_monitor
FOR EACH ROW EXECUTE FUNCTION proc.notify_process_monitor_ready();

-- 3c. extraction_review_queue (depends on process_monitor)
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
CREATE INDEX IF NOT EXISTS idx_eq_doc_type   ON proc.extraction_review_queue (doc_type);

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

-- 3d. bp_extraction_provenance_v3 (from 2026-05-08 migration)
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

-- 3e. bp_extraction_template (vendor template store)
CREATE TABLE IF NOT EXISTS proc.bp_extraction_template (
    fingerprint      TEXT NOT NULL PRIMARY KEY,
    vendor_name      TEXT,
    doc_type         TEXT NOT NULL,
    field_hints      JSONB NOT NULL DEFAULT '{}',
    line_item_hints  JSONB,
    success_count    INTEGER NOT NULL DEFAULT 0,
    correction_count INTEGER NOT NULL DEFAULT 0,
    created_at       TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_used_at     TIMESTAMP WITH TIME ZONE
);
CREATE INDEX IF NOT EXISTS idx_bp_extraction_template_doc_type ON proc.bp_extraction_template (doc_type);
CREATE INDEX IF NOT EXISTS idx_bp_extraction_template_vendor   ON proc.bp_extraction_template (vendor_name);

-- 3f. bp_extraction_type_priors (from 2026-04-21 migration)
CREATE TABLE IF NOT EXISTS proc.bp_extraction_type_priors (
    doc_type   TEXT PRIMARY KEY,
    priors     JSONB NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3g. bp_extraction_patterns (used by template store + pattern learning)
CREATE TABLE IF NOT EXISTS proc.bp_extraction_patterns (
    id              BIGSERIAL PRIMARY KEY,
    doc_type        TEXT NOT NULL,
    field_name      TEXT NOT NULL,
    pattern_type    TEXT NOT NULL,
    pattern_value   TEXT NOT NULL,
    anchor_patterns JSONB,
    confidence      NUMERIC(3,2) NOT NULL DEFAULT 1.0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3h. Legacy bp_extraction_provenance (for backward-compatible reads by KG builder)
CREATE TABLE IF NOT EXISTS proc.bp_extraction_provenance (
    id               BIGSERIAL PRIMARY KEY,
    parent_table     TEXT NOT NULL,
    parent_pk        TEXT NOT NULL,
    field_name       TEXT NOT NULL,
    source           TEXT NOT NULL,
    anchor_ref       JSONB,
    derivation_trace JSONB,
    confidence       NUMERIC(3,2),
    attempt          INT,
    extracted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_prov_parent ON proc.bp_extraction_provenance (parent_table, parent_pk);
