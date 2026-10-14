-- Rollback for 2026-10-14_sales_quote_settings.sql. Drops the per-quote tax rate (any
-- rate a quote was priced at is lost) and removes the three settings from the policy row.

BEGIN;

ALTER TABLE proc.bp_sales_quote DROP COLUMN IF EXISTS tax_rate_pct;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{rules}',
         (policy_details->'rules') - 'tax_rate_pct' - 'validity_days' - 'payment_terms_default'),
       last_modified_date = now(), last_modified_by = 'sales_quote_builder'
 WHERE policy_details->>'policy_identifier' = 'sales_quote_pricing';

COMMIT;
