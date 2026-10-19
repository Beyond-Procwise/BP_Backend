"""The audit tables on the real (test) database: written by the real writer, and append-only because the database says so.

Each probe runs in a transaction that is rolled back, so no audit row is left behind.  Needs PROCWISE_TEST_LIVE_DB=1.
"""
import os

import pytest

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"), pytest.mark.integration]


@pytest.fixture()
def conn():
    from src.services.db import get_conn
    with get_conn() as c:
        c.autocommit = False
        try:
            yield c
        finally:
            c.rollback()


def _call_row(cur):
    cur.execute("INSERT INTO proc.bp_gateway_call (call_id, ts, caller_subject, role, function, ok) "
                "VALUES ('probe-1', now(), 'probe', 'Admin', 'probe_fn', true)")
    return "probe-1"


def test_a_call_row_cannot_be_updated_deleted_or_truncated(conn):
    cur = conn.cursor()
    cid = _call_row(cur)
    for sql in ("UPDATE proc.bp_gateway_call SET role = 'x' WHERE call_id = %s", "DELETE FROM proc.bp_gateway_call WHERE call_id = %s"):
        cur.execute("SAVEPOINT s")
        with pytest.raises(Exception) as exc:
            cur.execute(sql, (cid,))
        assert "append-only" in str(exc.value).lower()
        cur.execute("ROLLBACK TO SAVEPOINT s")
    cur.execute("SAVEPOINT s")
    with pytest.raises(Exception) as exc:
        cur.execute("TRUNCATE proc.bp_gateway_call")
    assert "append-only" in str(exc.value).lower()


def test_a_write_row_is_append_only_and_an_idempotency_key_is_unique_per_function(conn):
    cur = conn.cursor()
    cur.execute("INSERT INTO proc.bp_gateway_write (write_id, ts, actor, function, idempotency_key) VALUES ('w1', now(), 'probe', 'f', 'k1')")
    cur.execute("SAVEPOINT s")
    with pytest.raises(Exception) as exc:
        cur.execute("INSERT INTO proc.bp_gateway_write (write_id, ts, actor, function, idempotency_key) VALUES ('w2', now(), 'probe', 'f', 'k1')")
    assert "unique" in str(exc.value).lower() or "duplicate" in str(exc.value).lower()
    cur.execute("ROLLBACK TO SAVEPOINT s")
    for sql in ("UPDATE proc.bp_gateway_write SET actor = 'x' WHERE write_id = 'w1'", "DELETE FROM proc.bp_gateway_write WHERE write_id = 'w1'"):
        cur.execute("SAVEPOINT s2")
        with pytest.raises(Exception) as exc:
            cur.execute(sql)
        assert "append-only" in str(exc.value).lower()
        cur.execute("ROLLBACK TO SAVEPOINT s2")


def test_the_real_writer_stores_a_gateway_call(conn):
    """The writer is pointed at THIS test's transaction (not an autocommit connection), so the probe row is rolled back:
    the table is append-only, so a committed probe row could never be cleaned up."""
    import contextlib
    from src.services.gateway import AuditSink, Backends, CallerContext, build_default_gateway
    from src.services.gateway.audit import DbAuditWriter
    own = lambda: contextlib.nullcontext(conn)
    g = build_default_gateway(Backends(connect=own), audit=AuditSink(db_writer=DbAuditWriter(own)))
    r = g.call(CallerContext("live-audit-probe", "Admin", None, agent="probe"), "get_inventory_position", {"sku": "A"})
    cur = conn.cursor()
    cur.execute("SELECT function, error_code, caller_subject FROM proc.bp_gateway_call WHERE call_id = %s", (r.call_id,))
    assert cur.fetchone() == ("get_inventory_position", "UNSUPPORTED", "live-audit-probe")
