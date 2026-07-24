BEGIN;
-- deploy/sql/2026-07-24_po_quote_reference.sql
--
-- The quote -> PO award link. Every PO in the corpus names the bid it was raised
-- against ONCE, in its header:
--
--     Reference  Against BAFO quote SDP-Q-44120 · RFQ PROC-2024-RFQ-FRT-005
--
-- There was nowhere to put that. The only existing carrier,
-- bp_po_line_items_stg.quote_number, is per-LINE -- the wrong grain -- and is
-- 0/316 filled in _raw and 0/171 in _stg across the entire database, i.e. no
-- extraction run has ever populated it. Award detection therefore never fired on
-- real data and every PO came back orphaned.
--
-- This adds the header-grain column the documents actually use. The value is
-- stored VERBATIM as cited (a "(V3)" suffix in the doc is kept); consumers
-- version-collapse it. Idempotent / additive DDL only -- no existing column is
-- altered and quote_number is left in place for data that uses it.

ALTER TABLE proc.bp_purchase_order_raw  ADD COLUMN IF NOT EXISTS quote_reference TEXT;
ALTER TABLE proc.bp_purchase_order_stg  ADD COLUMN IF NOT EXISTS quote_reference TEXT;
ALTER TABLE proc.bp_purchase_order_trgt ADD COLUMN IF NOT EXISTS quote_reference TEXT;

COMMENT ON COLUMN proc.bp_purchase_order_stg.quote_reference IS
  'Quote/tender reference this PO was raised against, exactly as cited in the PO '
  'header. NULL when the PO cites none. Version-collapsed by consumers.';

COMMIT;
