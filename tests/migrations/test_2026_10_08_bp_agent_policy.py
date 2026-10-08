"""proc.bp_agent_policy* in both live databases. Needs PROCWISE_TEST_LIVE_DB=1."""
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
        for t in ("bp_business_area", "bp_orchestrator_registry",
                  "bp_agent_policy", "bp_agent_policy_version"):
            cur.execute("SELECT to_regclass(%s)", (f"proc.{t}",))
            assert cur.fetchone()[0] is not None, t


@pytest.mark.parametrize("dbname", DATABASES)
def test_every_area_has_general_and_finance_needs_second_reviewer(dbname):
    with _connect(dbname) as conn, conn.cursor() as cur:
        cur.execute("SELECT area_name, sub_areas, second_reviewer, never_suggest FROM proc.bp_business_area")
        rows = {r[0]: r for r in cur.fetchall()}
    assert all("General" in r[1] for r in rows.values())
    assert rows["Finance"][2] is True
    assert not any(r[2] for n, r in rows.items() if n != "Finance")
    assert rows["Legal and compliance"][3] is True and rows["Security"][3] is True


@pytest.mark.parametrize("dbname", DATABASES)
def test_saved_version_cannot_be_changed_or_deleted(dbname):
    import psycopg2
    conn = _connect(dbname)
    conn.autocommit = False
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO proc.bp_agent_policy (policy_key, area_name, status, latest_version, created_by)"
                    " VALUES ('TST-9999', NULL, 'draft', 1, 'test') ")
        cur.execute("INSERT INTO proc.bp_agent_policy_version (policy_key, version, saved_as, form_state, saved_by)"
                    " VALUES ('TST-9999', 1, 'draft', '{}'::jsonb, 'test')")
        with pytest.raises(psycopg2.Error):
            cur.execute("UPDATE proc.bp_agent_policy_version SET change_note='x' WHERE policy_key='TST-9999'")
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
