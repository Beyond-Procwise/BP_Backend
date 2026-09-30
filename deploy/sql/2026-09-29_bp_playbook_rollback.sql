-- Rollback for 2026-09-29_bp_playbook.sql.
--
-- Run this only alongside reverting the code: the sweep and the playbooks
-- router read these tables and will fail loudly without them.
--
-- Proposals are dropped with the playbooks. They are advisory records of what
-- the system recommended, not a financial ledger; an executed proposal's run
-- survives independently in proc.bp_workflow_run.

BEGIN;

DROP TABLE IF EXISTS proc.bp_playbook_proposal;
DROP TABLE IF EXISTS proc.bp_playbook;

DELETE FROM proc.bp_policy
 WHERE policy_details->>'policy_identifier' = 'playbook_authority'
   AND created_by = 'bp_playbook_migration_2026_09_29';

COMMIT;
