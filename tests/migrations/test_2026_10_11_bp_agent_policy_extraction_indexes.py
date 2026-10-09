"""Stage 2 final review: the two lookup indexes, in both live databases. Needs PROCWISE_TEST_LIVE_DB=1."""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)
DATABASES = ("bp_testdb", "bp_sqldb")
INDEXES = {
    "ix_bp_policy_extraction_item_policy_key": ("bp_policy_extraction_item", "(policy_key)"),
    "ix_bp_policy_document_version_s3_key": ("bp_policy_document_version", "(s3_key)"),
}


def _connect(dbname):
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        dbname=dbname, connect_timeout=10,
    )


@pytest.mark.parametrize("dbname", DATABASES)
def test_lookup_indexes_exist_on_the_right_columns(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        for name, (table, cols) in INDEXES.items():
            cur.execute("SELECT tablename, indexdef FROM pg_indexes WHERE schemaname = 'proc' AND indexname = %s",
                        (name,))
            row = cur.fetchone()
            assert row is not None, f"{name} missing in {dbname}"
            assert row[0] == table and row[1].endswith(cols), row
