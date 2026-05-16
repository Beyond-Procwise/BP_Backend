-- scripts/migrations/2026-05-16-extraction-raw-flat-columns.sql
-- Additive: ALTER existing proc.bp_<doctype>_raw tables to ADD flat columns
-- that mirror the corresponding proc.bp_<doctype>_stg schema, plus control
-- columns (parser_snapshot, trace_id, process_monitor_id, promoted_at).
--
-- raw_payload JSONB is RETAINED. A follow-up migration drops it after the
-- new pipeline is verified writing to the flat columns end-to-end.
--
-- Existing non-promoted rows are backfilled from raw_payload by
-- 2026-05-16-backfill-raw-jsonb-to-columns.py (run separately after this).
--
-- Run against: bp_sqldb.

BEGIN;

-- ============================================================
-- INVOICE _raw — additive ALTER + create line_items_raw
-- ============================================================
ALTER TABLE proc.bp_invoice_raw
    ADD COLUMN IF NOT EXISTS process_monitor_id  INT REFERENCES proc.process_monitor(id),
    ADD COLUMN IF NOT EXISTS parser_snapshot     JSONB,
    ADD COLUMN IF NOT EXISTS trace_id            UUID,
    ADD COLUMN IF NOT EXISTS promoted_at         TIMESTAMPTZ,
    -- field columns mirror bp_invoice_stg (extraction-time only; audit cols
    -- like created_date/last_modified_* are populated at promotion)
    ADD COLUMN IF NOT EXISTS invoice_id              TEXT,
    ADD COLUMN IF NOT EXISTS po_id                   TEXT,
    ADD COLUMN IF NOT EXISTS supplier_id             TEXT,
    ADD COLUMN IF NOT EXISTS supplier_name           TEXT,
    ADD COLUMN IF NOT EXISTS buyer_id                TEXT,
    ADD COLUMN IF NOT EXISTS requisition_id          TEXT,
    ADD COLUMN IF NOT EXISTS requested_by            TEXT,
    ADD COLUMN IF NOT EXISTS requested_date          DATE,
    ADD COLUMN IF NOT EXISTS invoice_date            DATE,
    ADD COLUMN IF NOT EXISTS due_date                DATE,
    ADD COLUMN IF NOT EXISTS invoice_paid_date       DATE,
    ADD COLUMN IF NOT EXISTS payment_terms           TEXT,
    ADD COLUMN IF NOT EXISTS currency                VARCHAR,
    ADD COLUMN IF NOT EXISTS invoice_amount          NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS tax_percent             NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS tax_amount              NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS invoice_total_incl_tax  NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS exchange_rate_to_usd    NUMERIC(10,4),
    ADD COLUMN IF NOT EXISTS converted_amount_usd    NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS country                 TEXT,
    ADD COLUMN IF NOT EXISTS region                  TEXT;

CREATE INDEX IF NOT EXISTS idx_invoice_raw_trace ON proc.bp_invoice_raw (trace_id);
CREATE INDEX IF NOT EXISTS idx_invoice_raw_pm    ON proc.bp_invoice_raw (process_monitor_id);

CREATE TABLE IF NOT EXISTS proc.bp_invoice_line_items_raw (
    line_raw_id            BIGSERIAL PRIMARY KEY,
    raw_id                 BIGINT NOT NULL REFERENCES proc.bp_invoice_raw(raw_id) ON DELETE CASCADE,
    line_no                INTEGER NOT NULL,
    item_id                TEXT,
    item_description       TEXT,
    quantity               INTEGER,
    unit_of_measure        TEXT,
    unit_price             NUMERIC(10,2),
    line_amount            NUMERIC(18,2),
    tax_percent            NUMERIC(5,2),
    tax_amount             NUMERIC(18,2),
    total_amount_incl_tax  NUMERIC(18,2),
    po_id                  TEXT,
    delivery_date          DATE,
    country                TEXT,
    region                 TEXT,
    UNIQUE (raw_id, line_no)
);
CREATE INDEX IF NOT EXISTS idx_invoice_line_raw_raw ON proc.bp_invoice_line_items_raw (raw_id);

-- ============================================================
-- PURCHASE_ORDER _raw — additive ALTER + create line_items_raw
-- ============================================================
ALTER TABLE proc.bp_purchase_order_raw
    ADD COLUMN IF NOT EXISTS process_monitor_id  INT REFERENCES proc.process_monitor(id),
    ADD COLUMN IF NOT EXISTS parser_snapshot     JSONB,
    ADD COLUMN IF NOT EXISTS trace_id            UUID,
    ADD COLUMN IF NOT EXISTS promoted_at         TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS po_id                   TEXT,
    ADD COLUMN IF NOT EXISTS supplier_id             TEXT,
    ADD COLUMN IF NOT EXISTS supplier_name           TEXT,
    ADD COLUMN IF NOT EXISTS buyer_id                TEXT,
    ADD COLUMN IF NOT EXISTS requisition_id          TEXT,
    ADD COLUMN IF NOT EXISTS requested_by            TEXT,
    ADD COLUMN IF NOT EXISTS requested_date          DATE,
    ADD COLUMN IF NOT EXISTS order_date              DATE,
    ADD COLUMN IF NOT EXISTS expected_delivery_date  DATE,
    ADD COLUMN IF NOT EXISTS payment_terms           VARCHAR,
    ADD COLUMN IF NOT EXISTS currency                VARCHAR,
    ADD COLUMN IF NOT EXISTS total_amount            NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS ship_to_country         TEXT,
    ADD COLUMN IF NOT EXISTS delivery_region         TEXT,
    ADD COLUMN IF NOT EXISTS incoterm                TEXT,
    ADD COLUMN IF NOT EXISTS tax_percent             NUMERIC,
    ADD COLUMN IF NOT EXISTS tax_amount              NUMERIC,
    ADD COLUMN IF NOT EXISTS total_amount_incl_tax   NUMERIC,
    ADD COLUMN IF NOT EXISTS delivery_address_line1  TEXT,
    ADD COLUMN IF NOT EXISTS delivery_address_line2  TEXT,
    ADD COLUMN IF NOT EXISTS delivery_city           TEXT,
    ADD COLUMN IF NOT EXISTS postal_code             TEXT,
    ADD COLUMN IF NOT EXISTS exchange_rate_to_usd    NUMERIC(18,4),
    ADD COLUMN IF NOT EXISTS converted_amount_usd    NUMERIC(18,4),
    ADD COLUMN IF NOT EXISTS contract_id             TEXT;

CREATE INDEX IF NOT EXISTS idx_po_raw_trace ON proc.bp_purchase_order_raw (trace_id);
CREATE INDEX IF NOT EXISTS idx_po_raw_pm    ON proc.bp_purchase_order_raw (process_monitor_id);

CREATE TABLE IF NOT EXISTS proc.bp_po_line_items_raw (
    line_raw_id      BIGSERIAL PRIMARY KEY,
    raw_id           BIGINT NOT NULL REFERENCES proc.bp_purchase_order_raw(raw_id) ON DELETE CASCADE,
    line_number      INTEGER NOT NULL,
    item_id          TEXT,
    item_description TEXT,
    quote_number     TEXT,
    quantity         NUMERIC(18,2),
    unit_price       NUMERIC(18,2),
    unit_of_measure  TEXT,
    currency         VARCHAR,
    line_total       NUMERIC(18,2),
    tax_percent      SMALLINT,
    tax_amount       NUMERIC(18,2),
    total_amount     NUMERIC(18,2),
    UNIQUE (raw_id, line_number)
);
CREATE INDEX IF NOT EXISTS idx_po_line_raw_raw ON proc.bp_po_line_items_raw (raw_id);

-- ============================================================
-- QUOTE _raw — additive ALTER + create line_items_raw
-- ============================================================
ALTER TABLE proc.bp_quote_raw
    ADD COLUMN IF NOT EXISTS process_monitor_id  INT REFERENCES proc.process_monitor(id),
    ADD COLUMN IF NOT EXISTS parser_snapshot     JSONB,
    ADD COLUMN IF NOT EXISTS trace_id            UUID,
    ADD COLUMN IF NOT EXISTS promoted_at         TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS quote_id                TEXT,
    ADD COLUMN IF NOT EXISTS deal_id                 TEXT,
    ADD COLUMN IF NOT EXISTS supplier_id             TEXT,
    ADD COLUMN IF NOT EXISTS supplier_name           TEXT,
    ADD COLUMN IF NOT EXISTS buyer_id                TEXT,
    ADD COLUMN IF NOT EXISTS supplier_address        TEXT,
    ADD COLUMN IF NOT EXISTS buyer_address           TEXT,
    ADD COLUMN IF NOT EXISTS quote_date              DATE,
    ADD COLUMN IF NOT EXISTS validity_date           DATE,
    ADD COLUMN IF NOT EXISTS currency                VARCHAR,
    ADD COLUMN IF NOT EXISTS total_amount            NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS tax_percent             NUMERIC(5,2),
    ADD COLUMN IF NOT EXISTS tax_amount              NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS total_amount_incl_tax   NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS po_id                   TEXT,
    ADD COLUMN IF NOT EXISTS country                 TEXT,
    ADD COLUMN IF NOT EXISTS region                  TEXT;

CREATE INDEX IF NOT EXISTS idx_quote_raw_trace ON proc.bp_quote_raw (trace_id);
CREATE INDEX IF NOT EXISTS idx_quote_raw_pm    ON proc.bp_quote_raw (process_monitor_id);

CREATE TABLE IF NOT EXISTS proc.bp_quote_line_items_raw (
    line_raw_id      BIGSERIAL PRIMARY KEY,
    raw_id           BIGINT NOT NULL REFERENCES proc.bp_quote_raw(raw_id) ON DELETE CASCADE,
    line_number      INTEGER NOT NULL,
    item_id          TEXT,
    item_description TEXT,
    quantity         INTEGER,
    unit_of_measure  TEXT,
    unit_price       NUMERIC(18,2),
    line_total       NUMERIC(18,2),
    tax_percent      NUMERIC(5,2),
    tax_amount       NUMERIC(18,2),
    total_amount     NUMERIC(18,2),
    currency         VARCHAR,
    UNIQUE (raw_id, line_number)
);
CREATE INDEX IF NOT EXISTS idx_quote_line_raw_raw ON proc.bp_quote_line_items_raw (raw_id);

-- ============================================================
-- CONTRACT _raw — additive ALTER (bp_contracts has no _stg; mirrors target)
-- ============================================================
ALTER TABLE proc.bp_contract_raw
    ADD COLUMN IF NOT EXISTS process_monitor_id  INT REFERENCES proc.process_monitor(id),
    ADD COLUMN IF NOT EXISTS parser_snapshot     JSONB,
    ADD COLUMN IF NOT EXISTS trace_id            UUID,
    ADD COLUMN IF NOT EXISTS promoted_at         TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS contract_id                TEXT,
    ADD COLUMN IF NOT EXISTS contract_title             TEXT,
    ADD COLUMN IF NOT EXISTS contract_type              TEXT,
    ADD COLUMN IF NOT EXISTS supplier_id                TEXT,
    ADD COLUMN IF NOT EXISTS buyer_org_id               TEXT,
    ADD COLUMN IF NOT EXISTS contract_start_date        DATE,
    ADD COLUMN IF NOT EXISTS contract_end_date          DATE,
    ADD COLUMN IF NOT EXISTS currency                   TEXT,
    ADD COLUMN IF NOT EXISTS total_contract_value       NUMERIC(18,2),
    ADD COLUMN IF NOT EXISTS spend_category             TEXT,
    ADD COLUMN IF NOT EXISTS business_unit_id           TEXT,
    ADD COLUMN IF NOT EXISTS cost_centre_id             TEXT,
    ADD COLUMN IF NOT EXISTS is_amendment               TEXT,
    ADD COLUMN IF NOT EXISTS parent_contract_id         TEXT,
    ADD COLUMN IF NOT EXISTS auto_renew_flag            TEXT,
    ADD COLUMN IF NOT EXISTS renewal_term               TEXT,
    ADD COLUMN IF NOT EXISTS contract_lifecycle_status  TEXT,
    ADD COLUMN IF NOT EXISTS jurisdiction               TEXT,
    ADD COLUMN IF NOT EXISTS governing_law              TEXT,
    ADD COLUMN IF NOT EXISTS contract_signatory_name    TEXT,
    ADD COLUMN IF NOT EXISTS contract_signatory_role    TEXT,
    ADD COLUMN IF NOT EXISTS payment_terms              TEXT,
    ADD COLUMN IF NOT EXISTS risk_assessment_completed  TEXT;

CREATE INDEX IF NOT EXISTS idx_contract_raw_trace ON proc.bp_contract_raw (trace_id);
CREATE INDEX IF NOT EXISTS idx_contract_raw_pm    ON proc.bp_contract_raw (process_monitor_id);

COMMIT;
