-- Rollback for 2026-10-03_atb_style_pack_policy.sql. Removes the row; approving then refuses
-- everyone again, which is the pre-migration state.
BEGIN;
DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'style_pack_authority';
COMMIT;
