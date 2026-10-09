-- How a proposal's price changes over its term.
--
-- A three-year order form prices each line by year (Year 1 / Year 2 / Year 3 / 3-yr subtotal)
-- and states an uplift ("Annual uplift (clause 6.2): 3.5% per annum"). Extraction kept the Year 1
-- figure as the line cost (the user's ruling) and dropped the rest.
-- src/services/extraction/price_schedule.py reads the schedule into `price_schedule`, JSON text
-- ({periods: [{period, label, amount}], term_total}); line_total stays Year 1. Text, not jsonb:
-- both promotions copy a row's values as read, and psycopg2 reads jsonb as a dict it cannot write
-- back, so a jsonb column would break promotion of every line carrying a schedule. Nullable, additive;
-- the raw writer and the raw->stg->trgt promotions copy the columns the two tables share.
--
-- The check that compares a schedule with its stated uplift allows for rounding in a printed
-- schedule: reconciliation_tolerances.uplift_tolerance_pp (percentage points), governed like every
-- other money tolerance. governed_limits.limit() RAISES on an absent key.
--
-- Apply to BOTH bp_testdb and bp_sqldb BEFORE the code that writes the column or reads the limit.
-- Idempotent. Then scripts/backfill_price_schedule.py fills existing documents.

BEGIN;
ALTER TABLE proc.bp_quote_line_items_raw    ADD COLUMN IF NOT EXISTS price_schedule text;
ALTER TABLE proc.bp_quote_line_items_stg    ADD COLUMN IF NOT EXISTS price_schedule text;
ALTER TABLE proc.bp_quote_line_items_trgt   ADD COLUMN IF NOT EXISTS price_schedule text;
ALTER TABLE proc.bp_po_line_items_raw       ADD COLUMN IF NOT EXISTS price_schedule text;
ALTER TABLE proc.bp_po_line_items_stg       ADD COLUMN IF NOT EXISTS price_schedule text;
ALTER TABLE proc.bp_po_line_items_trgt      ADD COLUMN IF NOT EXISTS price_schedule text;
ALTER TABLE proc.bp_invoice_line_items_raw  ADD COLUMN IF NOT EXISTS price_schedule text;
ALTER TABLE proc.bp_invoice_line_items_stg  ADD COLUMN IF NOT EXISTS price_schedule text;
ALTER TABLE proc.bp_invoice_line_items_trgt ADD COLUMN IF NOT EXISTS price_schedule text;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
         policy_details, '{rules}',
         (policy_details -> 'rules') || jsonb_build_object(
            -- How far a printed schedule's year-on-year rise may sit above the uplift the
            -- document states before it is reported (rounding in a printed schedule).
            'uplift_tolerance_pp', 0.25
         )),
       last_modified_date = NOW(),
       last_modified_by   = 'deploy/sql/2026-10-09_price_schedule.sql',
       version            = COALESCE(version, 1) + 1
 WHERE policy_details ->> 'policy_identifier' = 'reconciliation_tolerances';
COMMIT;
