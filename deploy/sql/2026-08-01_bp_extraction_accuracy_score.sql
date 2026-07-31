-- deploy/sql/2026-08-01_bp_extraction_accuracy_score.sql
-- How much the READERS of this document have historically been trusted.
--
-- confidence_score is completeness: how many fields are filled. It rises when a human types
-- a value in, whether or not the value is right. This is the other half — the mean measured
-- agreement rate of whoever produced this document's fields — so "complete" and "correct"
-- stop being the same word.
--
-- NULL means no reader on this document has enough verdicts to have a rate yet, which is
-- the honest answer and is what every row will say until the loop has run for a while.
--
-- Additive + idempotent.
BEGIN;

ALTER TABLE proc.bp_invoice_stg         ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_invoice_trgt        ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_quote_stg           ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_quote_trgt          ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_purchase_order_stg  ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;
ALTER TABLE proc.bp_purchase_order_trgt ADD COLUMN IF NOT EXISTS accuracy_score NUMERIC;

COMMENT ON COLUMN proc.bp_invoice_trgt.accuracy_score IS
    'Mean measured agreement rate (0-100) of the readers that produced this document''s '
    'fields, from proc.bp_extraction_verdict. NULL = not enough verdicts yet. Distinct from '
    'confidence_score, which measures completeness.';

COMMIT;
