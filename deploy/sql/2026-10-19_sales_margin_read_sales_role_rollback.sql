-- Sales goes back to being refused cost and margin.
BEGIN;
UPDATE proc.bp_policy
   SET policy_details = policy_details - 'also_allowed_roles',
       version = version + 1,
       last_modified_by = 'sales_margin_read_sales_role_rollback',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'sales_margin_read_authority'
   AND policy_details ? 'also_allowed_roles';
COMMIT;
