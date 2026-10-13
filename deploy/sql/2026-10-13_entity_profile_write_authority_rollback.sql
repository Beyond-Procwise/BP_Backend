-- Rollback for 2026-10-13_entity_profile_write_authority.sql. With the permit off nobody
-- can change the company profile; the stored profile is untouched and still read.
BEGIN;

UPDATE proc.bp_policy
   SET policy_status = 0, last_modified_date = now(), last_modified_by = 'sales_quote_builder'
 WHERE policy_details->>'policy_identifier' = 'entity_profile_write_authority'
   AND policy_status = 1;

COMMIT;
