BEGIN;
-- deploy/sql/2026-10-20_bp_agent_run_immutable_rollback.sql
--
-- Reverses 2026-10-20_bp_agent_run_immutable.sql. Running this makes the run record writable
-- again: the application role, which OWNS proc.bp_agent_run, can then edit, delete or truncate
-- past runs. If the reason is a legitimate removal (retention, erasure), apply this, do the
-- work, and re-apply the migration in the same window rather than leaving the table open.

DROP TRIGGER IF EXISTS tr_bp_agent_run_no_truncate ON proc.bp_agent_run;
DROP TRIGGER IF EXISTS tr_bp_agent_run_append_only ON proc.bp_agent_run;
DROP FUNCTION IF EXISTS proc.bp_agent_run_append_only();

COMMIT;
