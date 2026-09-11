BEGIN;
DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' IN (
    'reseller_catalog', 'sales_quote_approval_authority', 'sales_quote_issue_authority')
   AND created_by = 'reseller_catalog';
COMMIT;
