"""proc.bp_demand records who raised a demand, in columns the browser cannot write.

Needs PROCWISE_TEST_LIVE_DB=1; without it pytest uses a fake DB that cannot answer.
"""
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)

DATABASES = ("bp_testdb", "bp_sqldb")


def _connect(dbname):
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), dbname=dbname, connect_timeout=10,
    )
    conn.set_session(readonly=True)
    return conn


@pytest.mark.parametrize("db", DATABASES)
def test_the_raiser_columns_exist_and_are_text(db):
    conn = _connect(db)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='proc' AND table_name='bp_demand' AND column_name LIKE 'raised_by%%'")
        assert dict(cur.fetchall()) == {"raised_by_sub": "text", "raised_by_email": "text"}
    finally:
        conn.close()


def test_the_migration_is_additive_and_idempotent():
    here = os.path.dirname(__file__)
    up = open(os.path.join(here, "..", "..", "deploy", "sql", "2026-10-06_demand_raised_by.sql")).read().upper()
    assert "ADD COLUMN IF NOT EXISTS RAISED_BY_SUB" in up and "ADD COLUMN IF NOT EXISTS RAISED_BY_EMAIL" in up
    assert "DROP" not in up and "DELETE" not in up and "UPDATE " not in up
    down = open(os.path.join(here, "..", "..", "deploy", "sql", "2026-10-06_demand_raised_by_rollback.sql")).read().upper()
    assert "DROP COLUMN IF EXISTS RAISED_BY_SUB" in down
