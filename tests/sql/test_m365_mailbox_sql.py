"""The shared-mailbox migration, exercised by inserting, not by reading DDL."""
import os
import pytest

psycopg2 = pytest.importorskip("psycopg2")
pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="live DB test; set PROCWISE_TEST_LIVE_DB=1")

ARN = "arn:aws:secretsmanager:eu-west-1:123456789012:secret:m365/app-AbCdEf"


@pytest.fixture
def cur():
    from dotenv import load_dotenv
    load_dotenv(".env")
    conn = psycopg2.connect(host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
                            dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
                            password=os.getenv("DB_PASSWORD"))
    conn.autocommit = False
    c = conn.cursor()
    yield c
    conn.rollback()
    conn.close()


def _insert(cur, address, **extra):
    cols = {"user_ref": "org", "provider": "graph", "mailbox_address": address,
            "role": "send_receive", "credential_ref": ARN, **extra}
    names = ", ".join(cols)
    marks = ", ".join(["%s"] * len(cols))
    cur.execute(f"INSERT INTO proc.bp_mailbox_binding ({names}) VALUES ({marks}) "
                "RETURNING binding_id", list(cols.values()))
    return cur.fetchone()[0]


def test_send_receive_role_is_admitted(cur):
    assert _insert(cur, "m365-test-a@example.test") > 0


def test_default_requires_a_test_send(cur):
    with pytest.raises(psycopg2.errors.CheckViolation):
        _insert(cur, "m365-test-b@example.test", is_default=True,
                is_active=True, scope_verified_at="2026-10-15T00:00:00Z")


def test_only_one_active_default(cur):
    common = dict(is_default=True, is_active=True,
                  scope_verified_at="2026-10-15T00:00:00Z",
                  last_test_send_at="2026-10-15T00:00:00Z")
    _insert(cur, "m365-test-c@example.test", **common)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(cur, "m365-test-d@example.test", **common)


def test_same_org_address_cannot_be_bound_twice_while_active(cur):
    common = dict(is_active=True, scope_verified_at="2026-10-15T00:00:00Z")
    _insert(cur, "m365-test-e@example.test", **common)
    with pytest.raises(psycopg2.errors.UniqueViolation):
        _insert(cur, "m365-test-e@example.test", **common)
