import os

import pytest

psycopg2 = pytest.importorskip("psycopg2")
pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="live DB test; set PROCWISE_TEST_LIVE_DB=1")

from services.mail_m365.config import (CONFIG_KEY, consume_nonce, read_integration,
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
    cur.execute("SELECT config_value FROM proc.bp_admin_config WHERE config_key = %s",
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
            cur.execute("UPDATE proc.bp_admin_config SET config_value = %s::jsonb "
                        "WHERE config_key = %s",
                        (v if isinstance(v, str) else json.dumps(v), CONFIG_KEY))
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
