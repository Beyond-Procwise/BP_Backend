-- 2026-06-15 Fix: bp_purchase_order_{stg,trgt}.payment_terms was varchar(30),
-- too narrow for real free-text payment terms (>30 chars), which blocked
-- promotion (StringDataRightTruncation) and stranded PO rows at 'pending'.
-- Invoice/quote tables already use text; align the PO tables. Idempotent.
ALTER TABLE proc.bp_purchase_order_stg  ALTER COLUMN payment_terms TYPE text;
ALTER TABLE proc.bp_purchase_order_trgt ALTER COLUMN payment_terms TYPE text;
