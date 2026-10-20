BEGIN;
-- deploy/sql/2026-10-20_bp_agent_run_immutable.sql
--
-- proc.bp_agent_run holds the record of every tool-loop run: the system prompt as sent, the
-- task, each tool call with its arguments and a hash of what it returned, and the answer.
-- It is only evidence if it cannot be rewritten, so make it append-only in the DATABASE.
-- Same reasoning as 2026-09-16_bp_agent_actions_immutable.sql: the application role owns the
-- table, so REVOKE would be theatre; a BEFORE trigger that RAISEs is the only thing that
-- binds an owner. TRUNCATE needs its own statement-level trigger, because Postgres does not
-- route it through a row-level one.
--
-- INSERT stays open (the repository is a bare INSERT). Nothing in the product updates or
-- deletes a run. The one exception is a live-database test that tidied up its own rows; it
-- now leaves them, because an audit row is never tidied away.
--
-- Idempotent: the function is CREATE OR REPLACE and both triggers are dropped first.
-- Reversible: 2026-10-20_bp_agent_run_immutable_rollback.sql.

CREATE OR REPLACE FUNCTION proc.bp_agent_run_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'proc.bp_agent_run is append-only: % is refused. '
        'A run record says what an agent was asked, what it called and what it answered; it is '
        'never corrected and never tidied away. Write a new row instead.',
        TG_OP
        USING ERRCODE = 'integrity_constraint_violation';
    RETURN NULL;  -- unreachable; plpgsql wants a return on the path
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tr_bp_agent_run_append_only ON proc.bp_agent_run;
CREATE TRIGGER tr_bp_agent_run_append_only
    BEFORE UPDATE OR DELETE ON proc.bp_agent_run
    FOR EACH ROW
    EXECUTE FUNCTION proc.bp_agent_run_append_only();

DROP TRIGGER IF EXISTS tr_bp_agent_run_no_truncate ON proc.bp_agent_run;
CREATE TRIGGER tr_bp_agent_run_no_truncate
    BEFORE TRUNCATE ON proc.bp_agent_run
    FOR EACH STATEMENT
    EXECUTE FUNCTION proc.bp_agent_run_append_only();

-- Verify after applying (expects two rows: append_only + no_truncate):
--   SELECT tgname FROM pg_trigger
--    WHERE tgrelid = 'proc.bp_agent_run'::regclass AND NOT tgisinternal;

COMMIT;
