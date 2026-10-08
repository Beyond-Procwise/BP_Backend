-- Two dedicated database roles for the email assurance layer. NEW roles only: no existing role, grant,
-- default privilege, schema or table is altered (a test compares every other grantee's privileges before
-- and after). NOT APPLIED to any shared database - see the note on scope below.
--
--   email_agent_reader   SELECT on exactly what the assurance layer reads from the business tables, and
--                        NOTHING else. No write, no DDL, no access to email_agent, no bank-detail columns.
--   email_agent_writer   INSERT/SELECT/UPDATE on the email_agent tables, and NOTHING in proc. No DELETE,
--                        no TRUNCATE: the capture and learning tables are append-and-update.
--
-- SCOPE: a Postgres ROLE IS CLUSTER-WIDE. Creating it on the RDS cluster that hosts bp_testdb creates it for
-- every database on that cluster (bp_sqldb, uicanvas, ses, ...), even though the GRANTS below are per
-- database. That is why this has not been run on bp_testdb "as the non-prod copy": there is no non-prod
-- cluster. It was rehearsed on a throwaway Postgres built from a schema copy of bp_sqldb
-- (specs/2026-10-09-bp-sqldb-email-assurance-ddl-pack.md).
--
-- THE GRANT IS THE CONTROL. A fact_source in an email family names a table and column; the resolver checks
-- they are plain identifiers, but whether the role may read them is decided here. A fact source naming a
-- table that is not granted fails with "permission denied" and the resolver reports it unresolved - it
-- fails closed. Adding a new source table is a deliberate GRANT, in a migration, reviewed.
--
-- default_transaction_read_only = on is set on the reader as a guardrail. It is NOT a security boundary: a
-- session may switch it off. The absence of write privileges is the boundary, and a test proves the reader
-- still cannot write after switching it off.
--
-- LOGIN ROLES: these are NOLOGIN groups. An operator creates the login a service uses, with the password
-- supplied out of band (never in this repo):
--
--     CREATE ROLE email_agent_ro_svc LOGIN PASSWORD :'pw' IN ROLE email_agent_reader;
--     ALTER  ROLE email_agent_ro_svc SET default_transaction_read_only = on;
--     CREATE ROLE email_agent_rw_svc LOGIN PASSWORD :'pw' IN ROLE email_agent_writer;
--
-- then set EMAIL_AGENT_RO_USER / EMAIL_AGENT_RO_PASSWORD and EMAIL_AGENT_RW_USER / EMAIL_AGENT_RW_PASSWORD.
-- Until those are set the application uses an INTERIM control (a read-only SESSION on its existing login).

BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'email_agent_reader') THEN
        CREATE ROLE email_agent_reader NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'email_agent_writer') THEN
        CREATE ROLE email_agent_writer NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
    END IF;
END $$;

-- ---- reader: what the assurance layer reads ------------------------------------------------------------------
GRANT USAGE ON SCHEMA proc TO email_agent_reader;
GRANT SELECT ON proc.supplier_response, proc.workflow_email_tracking TO email_agent_reader;
-- Column-level: contact and tone inputs only. The bank_* columns, tax and registration ids are NOT readable.
GRANT SELECT (supplier_id, supplier_name, contact_name_1, contact_email_1, contact_name_2, contact_email_2,
              contact_role_1, is_preferred_supplier, country)
    ON proc.bp_supplier TO email_agent_reader;
ALTER ROLE email_agent_reader SET default_transaction_read_only = on;

-- The abandoned-draft sweep must confirm a draft was NOT sent before it records it as abandoned. Two columns of the drafts table
-- say whether it went; the body, the recipients and the payload are NOT readable. Skipped where the table is absent.
DO $$
BEGIN
    IF to_regclass('proc.draft_rfq_emails') IS NOT NULL THEN
        GRANT SELECT (unique_id, sent, sent_on) ON proc.draft_rfq_emails TO email_agent_reader;
    END IF;
END $$;

-- ---- writer: the capture and learning tables, nothing else -----------------------------------------------------
GRANT USAGE ON SCHEMA email_agent TO email_agent_writer;
GRANT SELECT, INSERT, UPDATE ON
    email_agent.bp_draft_capture, email_agent.bp_draft_outcome, email_agent.bp_dq_item, email_agent.bp_eval_candidate,
    email_agent.bp_review_item, email_agent.bp_style_rule, email_agent.bp_classifier_example,
    email_agent.bp_exemplar_candidate
    TO email_agent_writer;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA email_agent TO email_agent_writer;

-- Inbound flags: the writer records and decides them. Never deletes. (The reader has no access to email_agent at all.)
DO $$
BEGIN
    IF to_regclass('email_agent.bp_inbound_flag') IS NOT NULL THEN
        GRANT SELECT, INSERT, UPDATE ON email_agent.bp_inbound_flag TO email_agent_writer;
    END IF;
END $$;

-- Sender-authentication results: the writer records them. Never updates or deletes.
DO $$
BEGIN
    IF to_regclass('email_agent.bp_inbound_auth') IS NOT NULL THEN
        GRANT SELECT, INSERT ON email_agent.bp_inbound_auth TO email_agent_writer;
    END IF;
END $$;

-- RAW TEXT: the one table the writer may also DELETE from (the retention job purges it). It is not in the list
-- above on purpose: access to raw text is its own decision. The reader and PUBLIC get nothing. Skipped where
-- 2026-10-08_email_agent_sent_text.sql has not been applied.
DO $$
BEGIN
    IF to_regclass('email_agent.bp_draft_sent_text') IS NOT NULL THEN
        GRANT SELECT, INSERT, DELETE ON email_agent.bp_draft_sent_text TO email_agent_writer;
    END IF;
END $$;

COMMIT;
