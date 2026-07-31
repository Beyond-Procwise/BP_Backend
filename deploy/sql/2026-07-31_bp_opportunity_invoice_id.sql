-- deploy/sql/2026-07-31_bp_opportunity_invoice_id.sql
-- An opportunity that is about ONE invoice can now say which one.
--
-- bp_opportunity already carries quote_id and po_id — the documents a sourcing
-- opportunity is anchored to — but no invoice. A duplicate-invoice recovery is anchored
-- to an invoice and nothing else, and without this column the value-summary reader has no
-- shared key between the discrepancy that FOUND the duplicate and the opportunity that
-- RECOVERS it, so the same money would be counted twice: once as verified, once as
-- potential.
--
-- Additive + idempotent. Existing rows keep invoice_id NULL and behave exactly as before
-- (the reader's doc_pk falls through to po_id / quote_id as it always did).
BEGIN;

ALTER TABLE proc.bp_opportunity
    ADD COLUMN IF NOT EXISTS invoice_id VARCHAR;

COMMENT ON COLUMN proc.bp_opportunity.invoice_id IS
    'The invoice this opportunity is about, when it is about exactly one '
    '(e.g. a duplicate-invoice recovery). NULL for sourcing opportunities, which '
    'are anchored to a quote or a PO instead.';

CREATE INDEX IF NOT EXISTS ix_bp_opportunity_invoice_id
    ON proc.bp_opportunity (invoice_id)
    WHERE invoice_id IS NOT NULL;

COMMIT;
