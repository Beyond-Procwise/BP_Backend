-- Rollback for 2026-10-18_daily_brief_settings.sql. Removes the row only while it is still
-- as seeded: once an administrator has saved a change (last_modified_by is theirs), their
-- choices are not deleted by a rollback script. With the row gone the daily brief endpoint
-- refuses (503); Today's brief on Home is a separate endpoint and is unaffected.

BEGIN;

DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'daily_brief'
   AND created_by = 'daily_brief'
   AND last_modified_by = 'daily_brief';

COMMIT;
