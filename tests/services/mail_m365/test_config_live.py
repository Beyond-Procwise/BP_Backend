import os

import pytest

psycopg2 = pytest.importorskip("psycopg2")
pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="live DB test; set PROCWISE_TEST_LIVE_DB=1")

from services.mail_m365.config import (CONFIG_KEY, append_nonce, consume_nonce,
                                       merge_integration, read_integration,
                                       write_integration)


@pytest.fixture
def conn():
    from dotenv import load_dotenv
    load_dotenv(".env")
    c = psycopg2.connect(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
                         dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
                         password=os.getenv("DB_PASSWORD"))
    c.autocommit = True
    cur = c.cursor()
    cur.execute("SELECT config_value, last_modified_by, last_modified_date FROM proc.bp_admin_config WHERE config_key = %s",
                (CONFIG_KEY,))
    saved = cur.fetchone()
    cur.close()
    try:
        yield c
    finally:
        cur = c.cursor()
        if saved is None:
            cur.execute("DELETE FROM proc.bp_admin_config WHERE config_key = %s", (CONFIG_KEY,))
        else:
            import json
            v = saved[0]
            cur.execute("UPDATE proc.bp_admin_config SET config_value = %s::jsonb, "
                        "last_modified_by = %s, last_modified_date = %s WHERE config_key = %s",
                        (v if isinstance(v, str) else json.dumps(v), saved[1], saved[2],
                         CONFIG_KEY))
        cur.close()
        c.close()


def test_round_trip_and_nonce_is_spent_once(conn):
    value = {"status": "pending", "pending_nonces": ["a", "b"]}
    write_integration(conn, value, actor="tester")
    assert read_integration(conn) == value
    assert consume_nonce(conn, "a", actor="tester") is True
    assert consume_nonce(conn, "a", actor="tester") is False
    assert read_integration(conn)["pending_nonces"] == ["b"]
    assert consume_nonce(conn, "never-issued", actor="tester") is False


def test_append_and_merge_never_revive_a_spent_nonce(conn):
    write_integration(conn, {"status": "pending", "pending_nonces": []}, actor="tester")
    append_nonce(conn, "n1", actor="tester")
    assert consume_nonce(conn, "n1", actor="tester") is True
    append_nonce(conn, "n2", actor="tester")
    merged = merge_integration(conn, {"status": "connected", "tenant_id": "t"}, actor="tester")
    assert merged["pending_nonces"] == ["n2"] and merged["status"] == "connected"
    assert consume_nonce(conn, "n1", actor="tester") is False


def test_append_keeps_only_the_newest_twenty_and_creates_the_row(conn):
    cur = conn.cursor()
    cur.execute("DELETE FROM proc.bp_admin_config WHERE config_key = %s", (CONFIG_KEY,))
    cur.close()
    for i in range(23):
        append_nonce(conn, f"n{i}", actor="tester")
    assert read_integration(conn)["pending_nonces"] == [f"n{i}" for i in range(3, 23)]
