-- Removes the two roles and every privilege they were granted. Touches nothing else.
-- DROP OWNED BY revokes their privileges in THIS database (they own no objects, so nothing is dropped).
-- DROP ROLE then fails LOUDLY if the role still holds privileges in another database on the cluster:
-- that is correct, and means run this rollback in each database that received the migration first.
-- Any login roles created as members (email_agent_ro_svc, email_agent_rw_svc) are the operator's to drop.
BEGIN;
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'email_agent_reader') THEN
        EXECUTE 'DROP OWNED BY email_agent_reader';
        EXECUTE 'DROP ROLE email_agent_reader';
    END IF;
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'email_agent_writer') THEN
        EXECUTE 'DROP OWNED BY email_agent_writer';
        EXECUTE 'DROP ROLE email_agent_writer';
    END IF;
END $$;
COMMIT;
