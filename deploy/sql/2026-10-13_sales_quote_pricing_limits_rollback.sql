-- Rollback for 2026-10-13_sales_quote_pricing_limits.sql. Safe at any time: with the row
-- gone the Quote builder reports both limits as not set, exactly as it does with nulls.

BEGIN;

DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'sales_quote_pricing'
   AND created_by = 'sales_quote_builder';

COMMIT;
