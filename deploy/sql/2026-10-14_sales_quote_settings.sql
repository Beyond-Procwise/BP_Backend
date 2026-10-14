-- The Quote builder's settings, and the tax rate each quote was priced at (ruled 2026-10-14).
--
-- Company defaults live in the same governed policy row as the margin floor and the
-- discount corridor (`sales_quote_pricing`). Three rules are added, each stated as NOT
-- SET (null): nobody has given a value, and a number written here would be read as a
-- decision. A builder quote cannot go for review, or be sent, while the margin floor or
-- the tax rate is unset.
--
--   tax_rate_pct           percent, e.g. 20 = 20% tax on the quote's ex-tax total.
--   validity_days          whole days a new quote is valid for, e.g. 30.
--   payment_terms_default  words, e.g. "30 days from invoice".
--
-- Each quote records the tax rate it was actually priced at (bp_sales_quote.tax_rate_pct),
-- taken from the default when the quote is started, so a later change of the default
-- never rewrites a quote somebody already holds.
--
-- Idempotent: keys already present keep their value. Apply to BOTH bp_testdb and
-- bp_sqldb, before the code that reads them. Rollback: _rollback.sql beside this file.

BEGIN;

ALTER TABLE proc.bp_sales_quote ADD COLUMN IF NOT EXISTS tax_rate_pct NUMERIC(7, 4);

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(
         policy_details, '{rules}',
         jsonb_build_object('tax_rate_pct', NULL, 'validity_days', NULL,
                            'payment_terms_default', NULL)
         || (policy_details->'rules')),
       last_modified_date = now(), last_modified_by = 'sales_quote_builder'
 WHERE policy_details->>'policy_identifier' = 'sales_quote_pricing'
   AND policy_status = 1
   AND NOT (policy_details->'rules' ?& ARRAY['tax_rate_pct', 'validity_days',
                                             'payment_terms_default']);

COMMIT;
