-- 2026-06-11 Deal linking: additive schema. Idempotent. No destructive edits.
BEGIN;

-- deal_date = order expected delivery date, stamped on every doc in the deal.
ALTER TABLE proc.bp_invoice_trgt          ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_quote_trgt            ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_purchase_order_trgt   ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_invoice_stg           ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_quote_stg             ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_purchase_order_stg    ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_invoice_raw           ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_quote_raw             ADD COLUMN IF NOT EXISTS deal_date DATE;
ALTER TABLE proc.bp_purchase_order_raw    ADD COLUMN IF NOT EXISTS deal_date DATE;

-- Stable per-document identity within a deal. Supersedes proc.deal_document_id_map.
CREATE TABLE IF NOT EXISTS proc.bp_deal_document_map (
    document_id   VARCHAR PRIMARY KEY,
    deal_id       VARCHAR NOT NULL,
    deal_name     VARCHAR,
    doc_type      VARCHAR NOT NULL,    -- 'quote' | 'po' | 'invoice'
    doc_pk        VARCHAR NOT NULL,    -- quote_id / po_id / invoice_id
    source_file   TEXT,
    assigned_at   TIMESTAMPTZ DEFAULT NOW(),
    assigned_by   VARCHAR DEFAULT 'deal_assignment_service'
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_deal_document_map_natural
    ON proc.bp_deal_document_map (deal_id, doc_type, doc_pk);

CREATE INDEX IF NOT EXISTS ix_bp_invoice_trgt_deal_id        ON proc.bp_invoice_trgt (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_quote_trgt_deal_id          ON proc.bp_quote_trgt (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_purchase_order_trgt_deal_id ON proc.bp_purchase_order_trgt (deal_id);

COMMIT;
