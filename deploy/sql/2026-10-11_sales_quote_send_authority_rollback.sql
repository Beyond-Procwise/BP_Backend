-- Undo 2026-10-11_sales_quote_send_authority.sql: nobody may send a quote again.
-- The row is retired, not deleted, so what permitted earlier sends stays on record.
BEGIN;

UPDATE proc.bp_policy
   SET policy_status = 0, last_modified_date = now(), last_modified_by = 'sales_quote_send'
 WHERE policy_details->>'policy_identifier' = 'sales_quote_send_authority'
   AND policy_status = 1;

COMMIT;
