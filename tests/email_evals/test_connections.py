"""The reader and writer connection helpers: real connections where the behaviour depends on Postgres."""

from types import SimpleNamespace

import psycopg2
import pytest
from psycopg2 import extensions

from src.services.draft_assurance import connections

PW = "pw-for-tests-only"


def nick_for(conn):
    """An agent_nick whose get_db_connection hands back one existing connection (as the eval harness does)."""
    from contextlib import contextmanager

    @contextmanager
    def gdb():
        yield conn
    p = extensions.parse_dsn(conn.dsn)
    return SimpleNamespace(get_db_connection=gdb, settings=SimpleNamespace(db_host=p["host"], db_name=p["dbname"], db_port=p["port"]))


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in ("EMAIL_AGENT_RO_USER", "EMAIL_AGENT_RO_PASSWORD", "EMAIL_AGENT_RW_USER", "EMAIL_AGENT_RW_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    connections._warned.clear()


def writable(conn):
    with conn.cursor() as cur:
        try:
            cur.execute("CREATE TEMP TABLE _w (x int)")
            cur.execute("DROP TABLE _w")
            return True
        except psycopg2.errors.ReadOnlySqlTransaction:
            return False


def test_the_interim_reader_is_read_only_inside_and_gives_the_connection_back_writable_afterwards(eval_db):
    conn = eval_db
    assert writable(conn)
    state = {}
    with connections.reader(nick_for(conn), state) as r:
        assert r is conn and state["control"] == connections.INTERIM
        assert not writable(r)                       # reads only while the reader holds it
    assert writable(conn), "the read-only session leaked onto a connection someone else reuses"
    assert conn.autocommit is True


def test_the_restore_also_works_when_the_connection_was_read_only_before(eval_dsn):
    conn = psycopg2.connect(eval_dsn)
    conn.set_session(readonly=True, autocommit=True)
    with connections.reader(nick_for(conn), {}):
        pass
    assert not writable(conn)                        # it was read-only before; it still is
    conn.set_session(readonly=False)
    assert writable(conn)
    conn.close()


def test_the_reader_restores_even_when_the_work_inside_it_raises(eval_db):
    with pytest.raises(RuntimeError):
        with connections.reader(nick_for(eval_db), {}):
            raise RuntimeError("boom")
    assert writable(eval_db)


def test_a_connection_that_cannot_be_made_read_only_is_reported_unenforced_not_failed(eval_dsn):
    conn = psycopg2.connect(eval_dsn)
    conn.autocommit = False
    with conn.cursor() as cur:
        cur.execute("SELECT 1")                      # now inside a transaction: set_session will refuse
    state = {}
    with connections.reader(nick_for(conn), state) as r:
        assert r is conn
    assert state["control"] == connections.UNENFORCED
    conn.rollback(); conn.close()


def test_a_bare_connection_from_the_agent_is_closed_after_use(eval_dsn):
    raw = psycopg2.connect(eval_dsn)
    raw.autocommit = True
    nick = SimpleNamespace(get_db_connection=lambda: raw, settings=None)
    with connections.reader(nick, {}) as r:
        assert r is raw and not raw.closed
    assert raw.closed


def test_the_dedicated_reader_connects_as_the_role_and_cannot_write(eval_db, monkeypatch):
    cur = eval_db.cursor()
    cur.execute(f"CREATE ROLE ro_helper_login LOGIN PASSWORD '{PW}' IN ROLE email_agent_reader")
    try:
        monkeypatch.setenv("EMAIL_AGENT_RO_USER", "ro_helper_login")
        monkeypatch.setenv("EMAIL_AGENT_RO_PASSWORD", PW)
        state = {}
        with connections.reader(nick_for(eval_db), state) as r:
            assert state["control"] == connections.DEDICATED and connections.read_control() == connections.DEDICATED
            with r.cursor() as c:
                c.execute("SELECT current_user"); assert c.fetchone()[0] == "ro_helper_login"
                c.execute("SELECT count(*) FROM proc.supplier_response")
                with pytest.raises((psycopg2.errors.InsufficientPrivilege, psycopg2.errors.ReadOnlySqlTransaction)):
                    c.execute("UPDATE proc.supplier_response SET price = 0")
        assert r.closed
    finally:
        cur.execute("DROP ROLE ro_helper_login")


def test_the_dedicated_writer_writes_capture_rows_but_nothing_in_proc(eval_db, monkeypatch):
    cur = eval_db.cursor()
    cur.execute(f"CREATE ROLE rw_helper_login LOGIN PASSWORD '{PW}' IN ROLE email_agent_writer")
    try:
        monkeypatch.setenv("EMAIL_AGENT_RW_USER", "rw_helper_login")
        monkeypatch.setenv("EMAIL_AGENT_RW_PASSWORD", PW)
        with connections.writer(nick_for(eval_db)) as w:
            with w.cursor() as c:
                c.execute("INSERT INTO email_agent.bp_draft_capture (unique_id, family_id, assurance_status, draft_text, draft_hash) "
                          "VALUES ('U-W', 'f', 'verified', 't', 'h')")
        cur.execute("SELECT count(*) FROM email_agent.bp_draft_capture WHERE unique_id = 'U-W'")
        assert cur.fetchone()[0] == 1                # committed on a clean exit
        with pytest.raises(psycopg2.errors.InsufficientPrivilege):
            with connections.writer(nick_for(eval_db)) as w:
                with w.cursor() as c:
                    c.execute("UPDATE proc.supplier_response SET price = 0")
        cur.execute("DELETE FROM email_agent.bp_draft_capture WHERE unique_id = 'U-W'")
    finally:
        cur.execute("DROP ROLE rw_helper_login")


def test_a_writer_error_rolls_back(eval_db, monkeypatch):
    cur = eval_db.cursor()
    cur.execute(f"CREATE ROLE rw_helper_login2 LOGIN PASSWORD '{PW}' IN ROLE email_agent_writer")
    try:
        monkeypatch.setenv("EMAIL_AGENT_RW_USER", "rw_helper_login2")
        monkeypatch.setenv("EMAIL_AGENT_RW_PASSWORD", PW)
        with pytest.raises(RuntimeError):
            with connections.writer(nick_for(eval_db)) as w:
                with w.cursor() as c:
                    c.execute("INSERT INTO email_agent.bp_draft_capture (unique_id, family_id, assurance_status, draft_text, draft_hash) "
                              "VALUES ('U-RB', 'f', 'verified', 't', 'h')")
                raise RuntimeError("fail after the insert")
        cur.execute("SELECT count(*) FROM email_agent.bp_draft_capture WHERE unique_id = 'U-RB'")
        assert cur.fetchone()[0] == 0
    finally:
        cur.execute("DROP ROLE rw_helper_login2")


def test_without_dedicated_credentials_the_writer_is_the_agents_own_connection(eval_db):
    with connections.writer(nick_for(eval_db)) as w:
        assert w is eval_db


def test_a_half_configured_role_is_not_used(monkeypatch):
    monkeypatch.setenv("EMAIL_AGENT_RO_USER", "someone")           # no password
    assert connections.read_control() == connections.INTERIM
