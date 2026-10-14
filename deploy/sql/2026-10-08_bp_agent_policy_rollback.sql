-- Rollback for 2026-10-08_bp_agent_policy.sql. Destroys agent policies; run only to undo stage 1.
BEGIN;
DROP TRIGGER IF EXISTS tr_bp_agent_policy_version_immutable ON proc.bp_agent_policy_version;
DROP TRIGGER IF EXISTS tr_bp_agent_policy_no_delete ON proc.bp_agent_policy;
DROP TABLE IF EXISTS proc.bp_agent_policy_version;
DROP TABLE IF EXISTS proc.bp_agent_policy;
DROP TABLE IF EXISTS proc.bp_orchestrator_registry;
DROP TABLE IF EXISTS proc.bp_business_area;
DROP FUNCTION IF EXISTS proc.bp_agent_policy_version_immutable();
DROP FUNCTION IF EXISTS proc.bp_agent_policy_no_delete();
DELETE FROM proc.bp_admin_config WHERE config_key = 'agent_policy_settings';
COMMIT;
