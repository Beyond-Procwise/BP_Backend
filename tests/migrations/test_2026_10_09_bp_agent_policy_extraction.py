"""Stage 2 extraction tables in both live databases. Needs PROCWISE_TEST_LIVE_DB=1."""
import os
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)
DATABASES = ("bp_testdb", "bp_sqldb")


def _connect(dbname):
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        dbname=dbname, connect_timeout=10,
    )


@pytest.mark.parametrize("dbname", DATABASES)
def test_tables_exist(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        for t in ("bp_policy_document", "bp_policy_document_version",
                  "bp_policy_extraction_run", "bp_policy_extraction_item"):
            cur.execute("SELECT to_regclass(%s)", (f"proc.{t}",))
            assert cur.fetchone()[0] is not None, t


@pytest.mark.parametrize("dbname", DATABASES)
def test_source_columns_and_index_exist(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='proc' AND table_name='bp_agent_policy'")
        cols = {r[0] for r in cur.fetchall()}
        assert {"source_document_id", "source_reference", "source_split"} <= cols
        cur.execute("SELECT to_regclass('proc.ix_bp_agent_policy_source')")
        assert cur.fetchone()[0] is not None


@pytest.mark.parametrize("dbname", DATABASES)
def test_same_content_hash_cannot_be_stored_twice_for_a_document(dbname):
    import psycopg2
    conn = _connect(dbname)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO proc.bp_policy_document (title, match_name, created_by)"
                    " VALUES ('t', 't', 'test') RETURNING document_id")
        doc = cur.fetchone()[0]
        h = "a" * 64
        sql = ("INSERT INTO proc.bp_policy_document_version"
               " (document_id, version, filename, s3_key, byte_size, content_hash, uploaded_by)"
               " VALUES (%s, %s, 'f.pdf', 'k', 10, %s, 'test')")
        cur.execute(sql, (doc, 1, h))
        cur.execute("SAVEPOINT s")
        with pytest.raises(psycopg2.errors.UniqueViolation):
            cur.execute(sql, (doc, 2, h))
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_bp_policy_untouched(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cur.execute("SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema='proc' AND table_name='bp_policy' ORDER BY 1")
        cols = [r[0] for r in cur.fetchall()]
    assert cols == sorted(["created_by", "created_date", "last_modified_by", "last_modified_date",
                           "policy_desc", "policy_details", "policy_id", "policy_linked_agents",
                           "policy_name", "policy_status", "policy_type", "version"])
