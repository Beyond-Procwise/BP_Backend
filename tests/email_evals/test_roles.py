"""The two roles, tested by CONNECTING AS THEM and trying to do harm.

A grant list proves nothing until something has tried to exceed it. Every denial below is a real
`permission denied` from Postgres, not an assertion about what the migration text says.
"""

import psycopg2
import pytest
from psycopg2 import errors, extensions

from evals.email import db as evaldb

PW = "pw-for-tests-only"


def connect_as(eval_db, user):
    p = extensions.parse_dsn(eval_db.dsn)
    c = psycopg2.connect(host=p["host"], port=p["port"], dbname=p["dbname"], user=user, password=PW)
    c.autocommit = True
    return c


@pytest.fixture(scope="module")
def logins(eval_db):
    """Login roles created the way the migration says an operator should: members of the groups."""
    cur = eval_db.cursor()
    cur.execute("DROP TABLE IF EXISTS proc.bp_canary_business")
    cur.execute("CREATE TABLE proc.bp_canary_business (id int, secret text)")
    cur.execute("INSERT INTO proc.bp_canary_business VALUES (1, 'do-not-touch')")
    cur.execute("INSERT INTO proc.supplier_response (workflow_id, supplier_id, round_number, price, currency) "
                "VALUES ('w','S-1',1,10,'GBP') ON CONFLICT DO NOTHING") if False else None
    cur.execute(f"CREATE ROLE ro_test_login LOGIN PASSWORD '{PW}' IN ROLE email_agent_reader")
    cur.execute("ALTER ROLE ro_test_login SET default_transaction_read_only = on")
    cur.execute(f"CREATE ROLE rw_test_login LOGIN PASSWORD '{PW}' IN ROLE email_agent_writer")
    try:
        ro = connect_as(eval_db, "ro_test_login")
        # The login carries default_transaction_read_only = on. Switch it OFF for the denial tests so that the
        # only thing standing between this session and a write is the privilege itself.
        with ro.cursor() as c:
            c.execute("SET default_transaction_read_only = off")
        yield ro, connect_as(eval_db, "rw_test_login")
    finally:
        for r in ("ro_test_login", "rw_test_login"):
            cur.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE usename = %s", (r,))
            cur.execute(f"DROP ROLE IF EXISTS {r}")
        cur.execute("DROP TABLE IF EXISTS proc.bp_canary_business")


def denied(conn, sql, params=None):
    """The statement must fail with a PRIVILEGE error - not a typo, not a missing table, and NOT a
    read-only-transaction error: that one would fire first and hide a privilege that was granted by mistake."""
    with conn.cursor() as cur:
        with pytest.raises(errors.InsufficientPrivilege) as e:
            cur.execute(sql, params)
    return str(e.value)


def ok(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall() if cur.description else None


# --- the reader ---------------------------------------------------------------------------------------------------

def test_the_reader_can_read_exactly_what_the_assurance_layer_reads(logins):
    ro, _ = logins
    ok(ro, "SELECT price, currency, lead_time, rfq_id FROM proc.supplier_response")
    ok(ro, "SELECT count(*) FROM proc.workflow_email_tracking")
    ok(ro, "SELECT supplier_id, contact_name_1, contact_email_1, contact_role_1, is_preferred_supplier, country FROM proc.bp_supplier")


@pytest.mark.parametrize("sql", [
    "INSERT INTO proc.supplier_response (workflow_id) VALUES ('x')",
    "UPDATE proc.supplier_response SET price = 0",
    "DELETE FROM proc.supplier_response",
    "TRUNCATE proc.supplier_response",
    "UPDATE proc.bp_supplier SET contact_email_1 = 'evil@x.test'",
    "INSERT INTO proc.workflow_email_tracking (workflow_id) VALUES ('x')",
    "UPDATE proc.bp_canary_business SET secret = 'changed'",
    "INSERT INTO proc.bp_canary_business VALUES (2, 'x')",
    "DELETE FROM proc.bp_canary_business",
    "UPDATE proc.bp_policy SET policy_status = 0",
    "DELETE FROM proc.bp_policy",
    "UPDATE proc.bp_approval SET status = 'approved'",
])
def test_the_reader_cannot_write_to_any_business_table(logins, sql):
    denied(logins[0], sql)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM proc.bp_canary_business",
    "SELECT secret FROM proc.bp_canary_business",
    "SELECT * FROM proc.bp_policy",
    "SELECT * FROM proc.bp_prompt",
    "SELECT * FROM proc.bp_approval",
    "SELECT bank_iban FROM proc.bp_supplier",
    "SELECT bank_account_number, bank_swift FROM proc.bp_supplier",
    "SELECT tax_id, vat_number FROM proc.bp_supplier",
    "SELECT * FROM proc.bp_supplier",
    "SELECT supplier_id, bank_iban FROM proc.bp_supplier",
    "SELECT * FROM email_agent.bp_draft_capture",
])
def test_the_reader_cannot_read_what_it_was_not_granted(logins, sql):
    denied(logins[0], sql)


@pytest.mark.parametrize("sql", [
    "CREATE TABLE proc.reader_made_this (x int)",
    "CREATE TABLE public.reader_made_this (x int)",
    "CREATE TABLE email_agent.reader_made_this (x int)",
    "CREATE SCHEMA reader_schema",
    "DROP TABLE proc.supplier_response",
    "ALTER TABLE proc.supplier_response ADD COLUMN x int",
    "CREATE ROLE reader_made_a_role",
    "ALTER ROLE ro_test_login SUPERUSER",
    "GRANT SELECT ON proc.bp_canary_business TO ro_test_login",
    "GRANT email_agent_writer TO ro_test_login",
    "CREATE FUNCTION proc.f() RETURNS int LANGUAGE sql AS 'select 1'",
])
def test_the_reader_cannot_change_structure_or_grant_itself_anything(logins, sql):
    denied(logins[0], sql)


def test_the_read_only_default_is_a_guardrail_and_the_missing_privilege_is_the_boundary(eval_db, logins):
    """A fresh login session starts read-only (the guardrail) and a write fails on that. A session can turn the
    default off - and the write STILL fails, on the missing privilege. That is why the denial tests above run with
    it off: the boundary is the privilege, not the setting."""
    fresh = connect_as(eval_db, "ro_test_login")
    assert ok(fresh, "SHOW default_transaction_read_only") == [("on",)]
    with fresh.cursor() as cur:
        with pytest.raises(errors.ReadOnlySqlTransaction):
            cur.execute("UPDATE proc.supplier_response SET price = 0")
    ok(fresh, "SET default_transaction_read_only = off")
    denied(fresh, "UPDATE proc.supplier_response SET price = 0")
    denied(fresh, "INSERT INTO proc.bp_canary_business VALUES (3, 'x')")
    fresh.close()


def test_the_reader_cannot_become_anything_more_powerful(logins):
    ro, _ = logins
    for role in ("postgres", "email_agent_writer"):
        denied(ro, f"SET ROLE {role}")


# --- the writer -------------------------------------------------------------------------------------------------------

def test_the_writer_can_record_and_update_but_not_delete(logins):
    _, rw = logins
    cid = ok(rw, "INSERT INTO email_agent.bp_draft_capture (unique_id, family_id, assurance_status, draft_text, draft_hash) "
                 "VALUES ('U-R', 'f', 'verified', 't', 'h') RETURNING capture_id")[0][0]
    ok(rw, "INSERT INTO email_agent.bp_draft_outcome (capture_id, outcome) VALUES (%s, 'abandoned') RETURNING outcome_id", (cid,))
    ok(rw, "UPDATE email_agent.bp_draft_capture SET ready = true WHERE capture_id = %s", (cid,))
    ok(rw, "SELECT count(*) FROM email_agent.bp_dq_item")
    denied(rw, "DELETE FROM email_agent.bp_draft_capture")
    denied(rw, "TRUNCATE email_agent.bp_draft_capture")
    denied(rw, "DELETE FROM email_agent.bp_draft_outcome")


def test_raw_text_is_the_one_table_the_writer_may_purge_and_nothing_else_is_deletable(logins):
    _, rw = logins
    cid = ok(rw, "INSERT INTO email_agent.bp_draft_capture (unique_id, family_id, assurance_status, draft_text, draft_hash) "
                 "VALUES ('U-RAW', 'f', 'verified', 't', 'h') RETURNING capture_id")[0][0]
    oid = ok(rw, "INSERT INTO email_agent.bp_draft_outcome (capture_id, outcome) VALUES (%s, 'sent') RETURNING outcome_id", (cid,))[0][0]
    ok(rw, "INSERT INTO email_agent.bp_draft_sent_text (outcome_id, capture_id, sent_text, text_hash) VALUES (%s,%s,'x','h')", (oid, cid))
    assert ok(rw, "SELECT sent_text FROM email_agent.bp_draft_sent_text WHERE outcome_id = %s", (oid,)) == [("x",)]
    ok(rw, "DELETE FROM email_agent.bp_draft_sent_text WHERE outcome_id = %s", (oid,))          # the retention purge
    denied(rw, "TRUNCATE email_agent.bp_draft_sent_text")                                       # still no wholesale wipe
    denied(rw, "ALTER TABLE email_agent.bp_draft_sent_text ADD COLUMN x int")
    for other in ("bp_draft_capture", "bp_draft_outcome", "bp_dq_item", "bp_eval_candidate", "bp_review_item",
                  "bp_style_rule", "bp_classifier_example", "bp_exemplar_candidate"):
        denied(rw, f"DELETE FROM email_agent.{other}")


def test_the_reader_and_public_cannot_see_raw_text_at_all(logins, eval_db):
    ro, _ = logins
    denied(ro, "SELECT sent_text FROM email_agent.bp_draft_sent_text")
    denied(ro, "SELECT count(*) FROM email_agent.bp_draft_sent_text")
    cur = eval_db.cursor()
    cur.execute("SELECT has_table_privilege('public', 'email_agent.bp_draft_sent_text', 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')")
    assert cur.fetchone()[0] is False
    cur.execute("SELECT has_table_privilege('email_agent_reader', 'email_agent.bp_draft_sent_text', 'SELECT')")
    assert cur.fetchone()[0] is False


@pytest.mark.parametrize("sql", [
    "SELECT * FROM proc.supplier_response",
    "SELECT * FROM proc.bp_supplier",
    "SELECT * FROM proc.bp_canary_business",
    "INSERT INTO proc.supplier_response (workflow_id) VALUES ('x')",
    "UPDATE proc.supplier_response SET price = 0",
    "DELETE FROM proc.supplier_response",
    "TRUNCATE proc.supplier_response",
    "INSERT INTO proc.bp_canary_business VALUES (4, 'x')",
    "CREATE TABLE proc.writer_made_this (x int)",
    "CREATE TABLE email_agent.writer_made_this (x int)",
    "DROP TABLE email_agent.bp_draft_capture",
    "ALTER TABLE email_agent.bp_draft_capture ADD COLUMN x int",
    "GRANT SELECT ON proc.bp_canary_business TO rw_test_login",
    "CREATE ROLE writer_made_a_role",
])
def test_the_writer_cannot_touch_the_business_tables_or_change_structure(logins, sql):
    denied(logins[1], sql)


# --- the migration itself ------------------------------------------------------------------------------------------------

def test_the_roles_are_groups_with_no_dangerous_attributes(eval_db):
    cur = eval_db.cursor()
    cur.execute("SELECT rolname, rolcanlogin, rolsuper, rolcreaterole, rolcreatedb, rolreplication, rolbypassrls "
                "FROM pg_roles WHERE rolname LIKE 'email_agent_%' ORDER BY 1")
    assert cur.fetchall() == [("email_agent_reader", False, False, False, False, False, False),
                              ("email_agent_writer", False, False, False, False, False, False)]


def test_the_reader_group_carries_the_read_only_default_and_the_writer_does_not(eval_db):
    cur = eval_db.cursor()
    cur.execute("SELECT rolname, rolconfig FROM pg_roles WHERE rolname LIKE 'email_agent_%' ORDER BY 1")
    assert cur.fetchall() == [("email_agent_reader", ["default_transaction_read_only=on"]), ("email_agent_writer", None)]


def test_applying_the_roles_changes_no_other_roles_privileges(eval_dsn):
    """In a database the roles migration has NEVER touched, record what every pre-existing role (and PUBLIC) can
    EFFECTIVELY do - every privilege on every table, column and schema, as Postgres itself answers it - apply the
    migration, and ask again: nothing may differ. (Effective privileges, not the grant list: the first GRANT on an
    object makes the owner's implicit privileges explicit in the list, which looks like a change and is not.
    A probe database, because the shared eval database already had the migration applied when the session began.)"""
    admin = psycopg2.connect(eval_dsn)
    admin.autocommit = True
    admin.cursor().execute("DROP DATABASE IF EXISTS roles_probe")
    admin.cursor().execute("CREATE DATABASE roles_probe")
    probe = psycopg2.connect(eval_dsn.rsplit("/", 1)[0] + "/roles_probe")
    probe.autocommit = True
    cur = probe.cursor()
    try:
        evaldb.load(probe, skip=("2026-10-09_email_agent_roles.sql",), generate=True)
        cur.execute("DROP ROLE IF EXISTS bystander")
        cur.execute("CREATE ROLE bystander NOLOGIN")
        cur.execute("GRANT USAGE ON SCHEMA proc TO bystander")
        cur.execute("GRANT SELECT ON proc.supplier_response TO bystander")      # a grant that already exists must survive untouched

        def effective():
            cur.execute("SELECT c.oid::regclass::text, c.relkind FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                        "WHERE n.nspname IN ('proc', 'email_agent') AND c.relkind IN ('r', 'v', 'S') ORDER BY 1")
            rels = cur.fetchall()
            out = []
            for who in ("public", "bystander"):
                for schema in ("proc", "email_agent"):
                    for priv in ("USAGE", "CREATE"):
                        cur.execute("SELECT has_schema_privilege(%s, %s, %s)", (who, schema, priv))
                        out.append((who, "schema", schema, priv, cur.fetchone()[0]))
                for rel, kind in rels:
                    for priv in (("USAGE", "SELECT", "UPDATE") if kind == "S" else ("SELECT", "INSERT", "UPDATE", "DELETE", "TRUNCATE", "REFERENCES", "TRIGGER")):
                        fn = "has_sequence_privilege" if kind == "S" else "has_table_privilege"
                        cur.execute(f"SELECT {fn}(%s, %s, %s)", (who, rel, priv))
                        out.append((who, "rel", rel, priv, cur.fetchone()[0]))
                cur.execute("SELECT attname FROM pg_attribute WHERE attrelid = 'proc.bp_supplier'::regclass AND attnum > 0 AND NOT attisdropped")
                for (col,) in cur.fetchall():
                    cur.execute("SELECT has_column_privilege(%s, 'proc.bp_supplier', %s, 'SELECT')", (who, col))
                    out.append((who, "col", col, "SELECT", cur.fetchone()[0]))
            return out

        before = effective()
        assert any(r[:4] == ("bystander", "rel", "proc.supplier_response", "SELECT") and r[4] for r in before)   # the control is real
        cur.execute((evaldb.SQL / "2026-10-09_email_agent_roles.sql").read_text())
        after = effective()
        assert len(before) > 300
        assert after == before, [(b, a) for b, a in zip(before, after) if b != a][:5]
    finally:
        probe.close()
        admin.cursor().execute("DROP DATABASE IF EXISTS roles_probe")
        admin.cursor().execute("DROP ROLE IF EXISTS bystander")
        admin.close()


def test_the_migration_is_idempotent(eval_db):
    cur = eval_db.cursor()
    sql = (evaldb.SQL / "2026-10-09_email_agent_roles.sql").read_text()
    cur.execute(sql)
    cur.execute(sql)
    cur.execute("SELECT count(*) FROM pg_roles WHERE rolname LIKE 'email_agent_%'")
    assert cur.fetchone()[0] == 2


def test_every_table_the_assurance_layer_reads_is_granted_so_a_fact_source_naming_another_fails_closed(eval_db):
    """The reader's reach is data, not code: whatever a family's fact_sources name must be in this set."""
    import json
    cur = eval_db.cursor()
    cur.execute("SELECT policy_details -> 'rules' -> 'fact_sources' FROM proc.bp_policy WHERE policy_type = 'email_family'")
    needed = set()
    for (sources,) in cur.fetchall():
        for src in sources.values():
            needed.add((src["table"], src["column"]))
    for table, column in needed:
        cur.execute("SELECT has_column_privilege('email_agent_reader', %s, %s, 'SELECT')", (f"proc.{table}", column))
        assert cur.fetchone()[0], f"the reader cannot read proc.{table}.{column}, which a family fact source needs"
