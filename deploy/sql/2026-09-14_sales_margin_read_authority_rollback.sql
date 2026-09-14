-- Removing the permit withholds cost and margin from everyone; it does not expose them.
BEGIN;
DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'sales_margin_read_authority'
   AND created_by = 'sales_margin_read';
COMMIT;
