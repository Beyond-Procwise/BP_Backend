BEGIN;
-- deploy/sql/2026-09-16_bp_agent_actions_immutable_rollback.sql
--
-- Reverses 2026-09-16_bp_agent_actions_immutable.sql.
--
-- Running this makes the audit spine writable again: any connection holding the
-- application role -- which OWNS proc.bp_agent_actions -- can then edit a past action,
-- delete one, or empty the table with a single TRUNCATE. The rows survive the rollback
-- itself, but nothing afterwards guarantees the ones you read are the ones that were
-- written, and a rewritten audit row is worse than a missing one because it still looks
-- trustworthy.
--
-- If the reason for reverting is a legitimate need to remove rows (a retention policy,
-- a GDPR erasure), prefer a narrow, reviewed, audited path over standing the invariant
-- down: apply this, do the work, and re-apply the migration in the same maintenance
-- window rather than leaving the table unguarded.

DROP TRIGGER IF EXISTS tr_bp_agent_actions_no_truncate ON proc.bp_agent_actions;
DROP TRIGGER IF EXISTS tr_bp_agent_actions_append_only ON proc.bp_agent_actions;
DROP FUNCTION IF EXISTS proc.bp_agent_actions_append_only();

COMMIT;
