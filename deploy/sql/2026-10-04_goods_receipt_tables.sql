-- The fifth physical pipeline: proc.bp_goods_receipt_{raw,stg,trgt} and its
-- line items. Shaped like the purchase-order family so persistence.write_raw,
-- promotion.promote and deal assignment need no special case:
--   _raw  = control columns (raw_id, doc_pk_candidate, source_file,
--           process_monitor_id, pipeline_version, parser_snapshot, trace_id,
--           promotion_status, promoted_at, extracted_at, raw_payload)
--           + flat business columns
--   _stg  = the business columns, keyed by grn_id, plus the audit quartet and
--           the two scores promote() computes
--   _trgt = _stg plus document_id, keyed by deal_id for the readers
--
-- There is deliberately NO price, amount, total, value, cost, currency or tax
-- column on any of these six tables. A goods receipt proves DELIVERY; it
-- carries quantities and nothing else. A price column here would be filled by
-- the extractor from the PO the note references, and a fabricated price on the
-- document that proves delivery is the worst place in the product for one.
-- tests/sql/test_goods_receipt_tables.py asserts the absence.

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_raw (
    raw_id             bigserial PRIMARY KEY,
    doc_pk_candidate   text,
    source_file        text,
    raw_payload        jsonb NOT NULL DEFAULT '{}'::jsonb,
    extracted_at       timestamptz NOT NULL DEFAULT now(),
    pipeline_version   text,
    promotion_status   text,
    process_monitor_id integer,
    parser_snapshot    jsonb,
    trace_id           uuid,
    promoted_at        timestamptz,
    grn_id             text,
    po_id              text,
    supplier_id        text,
    supplier_name      text,
    receipt_date       date,
    delivery_note_ref  text,
    carrier_ref        text,
    received_by        text,
    ship_to_country    text,
    delivery_region    text,
    deal_id            varchar(100),
    deal_name          varchar(255),
    deal_date          date
);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_raw_po_id
    ON proc.bp_goods_receipt_raw (po_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_raw_grn_id
    ON proc.bp_goods_receipt_raw (grn_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_raw_source_file
    ON proc.bp_goods_receipt_raw (source_file);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_raw_promotion_status
    ON proc.bp_goods_receipt_raw (promotion_status);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_raw_process_monitor_id
    ON proc.bp_goods_receipt_raw (process_monitor_id);

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_line_items_raw (
    line_raw_id        bigserial PRIMARY KEY,
    raw_id             bigint REFERENCES proc.bp_goods_receipt_raw(raw_id),
    line_no            integer,
    item_id            text,
    item_description   text,
    quantity_received  numeric,
    quantity_rejected  numeric,
    unit_of_measure    text,
    po_id              text,
    po_line_ref        text,
    deal_id            varchar(100),
    deal_name          varchar(255)
);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_line_items_raw_raw_id
    ON proc.bp_goods_receipt_line_items_raw (raw_id);

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_stg (
    grn_id             text PRIMARY KEY,
    po_id              text,
    supplier_id        text,
    supplier_name      text,
    receipt_date       date,
    delivery_note_ref  text,
    carrier_ref        text,
    received_by        text,
    ship_to_country    text,
    delivery_region    text,
    created_date       timestamp,
    created_by         text,
    last_modified_by   text,
    last_modified_date timestamp,
    confidence_score   numeric,
    accuracy_score     numeric,
    deal_id            varchar(100),
    deal_name          varchar(255),
    deal_date          date
);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_stg_po_id
    ON proc.bp_goods_receipt_stg (po_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_stg_deal_id
    ON proc.bp_goods_receipt_stg (deal_id);

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_line_items_stg (
    goods_receipt_line_id text PRIMARY KEY,
    grn_id                text,
    line_no               integer,
    item_id               text,
    item_description      text,
    quantity_received     numeric,
    quantity_rejected     numeric,
    unit_of_measure       text,
    po_id                 text,
    po_line_ref           text,
    created_date          timestamp,
    created_by            text,
    last_modified_by      text,
    last_modified_date    timestamp,
    deal_id               varchar(100),
    deal_name             varchar(255)
);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_line_items_stg_grn_id
    ON proc.bp_goods_receipt_line_items_stg (grn_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_line_items_stg_po_id
    ON proc.bp_goods_receipt_line_items_stg (po_id);

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_trgt (
    LIKE proc.bp_goods_receipt_stg INCLUDING DEFAULTS
);
ALTER TABLE proc.bp_goods_receipt_trgt ADD COLUMN IF NOT EXISTS document_id varchar(100);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_goods_receipt_trgt_grn_id
    ON proc.bp_goods_receipt_trgt (grn_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_trgt_deal_id
    ON proc.bp_goods_receipt_trgt (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_trgt_po_id
    ON proc.bp_goods_receipt_trgt (po_id);

CREATE TABLE IF NOT EXISTS proc.bp_goods_receipt_line_items_trgt (
    LIKE proc.bp_goods_receipt_line_items_stg INCLUDING DEFAULTS
);
ALTER TABLE proc.bp_goods_receipt_line_items_trgt ADD COLUMN IF NOT EXISTS document_id varchar(100);
CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_goods_receipt_line_items_trgt_line_id
    ON proc.bp_goods_receipt_line_items_trgt (goods_receipt_line_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_line_items_trgt_deal_id
    ON proc.bp_goods_receipt_line_items_trgt (deal_id);
CREATE INDEX IF NOT EXISTS ix_bp_goods_receipt_line_items_trgt_grn_id
    ON proc.bp_goods_receipt_line_items_trgt (grn_id);
