-- deploy/sql/2026-07-31_bp_invoice_line_currency.sql
-- An invoice line can now say which currency it is in.
--
-- proc.bp_quote_line_items_trgt and proc.bp_po_line_items_trgt both carry a `currency`
-- column; the invoice line tables did not. So every invoice line silently inherited the
-- header currency, and a document billing freight in USD and duty in GBP was recorded
-- entirely in one of them with nothing able to express — let alone flag — the difference.
--
-- The column exists so a per-line disagreement has somewhere to live. It is NOT populated
-- by guesswork: it stays NULL, meaning "same as the header", which is what the overwhelming
-- majority of real invoices are. It is filled only where the line itself carries a symbol
-- or code that says otherwise, and a document printing more than one currency is stopped
-- for human review before it ever gets here (issue_type 'currency_ambiguous').
--
-- Additive + idempotent. Existing rows keep NULL and behave exactly as before.
BEGIN;

ALTER TABLE proc.bp_invoice_line_items_raw  ADD COLUMN IF NOT EXISTS currency VARCHAR;
ALTER TABLE proc.bp_invoice_line_items_stg  ADD COLUMN IF NOT EXISTS currency VARCHAR;
ALTER TABLE proc.bp_invoice_line_items_trgt ADD COLUMN IF NOT EXISTS currency VARCHAR;

COMMENT ON COLUMN proc.bp_invoice_line_items_trgt.currency IS
    'ISO 4217 code for THIS line, when the line states one that differs from the invoice '
    'header. NULL means the line is in the header currency — the normal case. Never '
    'inferred: a document printing more than one currency is held for review instead.';

COMMIT;
