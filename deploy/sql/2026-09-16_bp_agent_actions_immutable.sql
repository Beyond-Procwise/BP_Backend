BEGIN;
-- deploy/sql/2026-09-16_bp_agent_actions_immutable.sql
--
-- proc.bp_agent_actions is the audit spine. Make it append-only in the DATABASE.
--
-- Enforced here rather than in the repository, because the repository is not the only
-- thing that will ever hold a connection to this table. A migration, a support script
-- or a psql session at 2am is exactly when this rule matters most, and none of them
-- import Python. The application already treats the table as append-only -- the writer
-- in services/agent_actions.py is a bare INSERT with no ON CONFLICT and no UPDATE
-- anywhere -- so nothing in the product has to change for this to hold.
--
-- WHY A TRIGGER AND NOT A REVOKE. The application role `procwisedb123` is the OWNER of
-- this table in both bp_testdb and bp_sqldb. An owner may re-grant to itself at will,
-- so REVOKE UPDATE, DELETE would look like a control and stop nothing -- theatre, in
-- the sense the commercial_fact migration used the word about pretend RLS. A BEFORE
-- trigger that RAISEs is the only mechanism that actually binds the owner.
--
-- WHY TRUNCATE IS COVERED SEPARATELY. Postgres does not route TRUNCATE through a
-- row-level trigger. A table guarded only against UPDATE and DELETE can still be
-- emptied in a single statement -- 54,769 rows in bp_sqldb -- which is precisely the
-- move worth preventing. It needs its own statement-level trigger, and it has one
-- below. deploy/sql/truncate_for_fresh_extraction.sql listed this table and no longer
-- does; adding it back will now fail loudly instead of quietly emptying the log.
--
-- WHAT STAYS OPEN: INSERT. record_action_or_fail refuses to let an irreversible action
-- proceed when its audit row cannot be written, so a guard that blocked INSERT would
-- not merely lose the log -- it would stop the product working.
--
-- The rows are not the only thing being protected. Editing history is worse than
-- deleting it, because it still looks trustworthy.
--
-- Idempotent: the function is CREATE OR REPLACE and both triggers are dropped first.
-- Reversible: 2026-09-16_bp_agent_actions_immutable_rollback.sql.

CREATE OR REPLACE FUNCTION proc.bp_agent_actions_append_only()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'proc.bp_agent_actions is append-only: % is refused. '
        'An audit row records something that actually happened; it is never corrected '
        'and never tidied away. Write a new row instead.',
        TG_OP
        USING ERRCODE = 'integrity_constraint_violation';
    RETURN NULL;  -- unreachable; plpgsql wants a return on the path
END;
$$ LANGUAGE plpgsql;

-- UPDATE and DELETE: one row at a time.
DROP TRIGGER IF EXISTS tr_bp_agent_actions_append_only ON proc.bp_agent_actions;
CREATE TRIGGER tr_bp_agent_actions_append_only
    BEFORE UPDATE OR DELETE ON proc.bp_agent_actions
    FOR EACH ROW
    EXECUTE FUNCTION proc.bp_agent_actions_append_only();

-- TRUNCATE: the whole table in one statement, which the row-level trigger never sees.
DROP TRIGGER IF EXISTS tr_bp_agent_actions_no_truncate ON proc.bp_agent_actions;
CREATE TRIGGER tr_bp_agent_actions_no_truncate
    BEFORE TRUNCATE ON proc.bp_agent_actions
    FOR EACH STATEMENT
    EXECUTE FUNCTION proc.bp_agent_actions_append_only();

-- Verify after applying (expects two rows: append_only + no_truncate):
--   SELECT tgname FROM pg_trigger
--    WHERE tgrelid = 'proc.bp_agent_actions'::regclass AND NOT tgisinternal;
-- And the behaviour itself, which is what actually matters:
--   tests/guardrails/test_bp_agent_actions_append_only.py (PROCWISE_TEST_LIVE_DB=1)

COMMIT;
