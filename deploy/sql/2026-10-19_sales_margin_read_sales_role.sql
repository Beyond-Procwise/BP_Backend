-- Sales may read internal cost and margin, decided 2026-10-19.
--
-- margin.read is gated by required_role = Buyer (rank 2). Sales is rank 1, and so are Viewer
-- and SecurityReviewer, so lowering the required rank would let those two in as well. This
-- names Sales on the one policy instead (details.also_allowed_roles, read by
-- src/services/guardrail.py). Until that code is deployed the key is ignored and the policy
-- behaves exactly as before, so applying this ahead of the code withholds nothing new.
-- Idempotent. Reversible: 2026-10-19_sales_margin_read_sales_role_rollback.sql.
BEGIN;

UPDATE proc.bp_policy
   SET policy_details = jsonb_set(policy_details, '{also_allowed_roles}', '["Sales"]'::jsonb, true),
       version = version + 1,
       last_modified_by = 'sales_margin_read_sales_role',
       last_modified_date = now()
 WHERE policy_status = 1
   AND policy_details->>'policy_identifier' = 'sales_margin_read_authority'
   AND policy_details->'also_allowed_roles' IS DISTINCT FROM '["Sales"]'::jsonb;

COMMIT;
