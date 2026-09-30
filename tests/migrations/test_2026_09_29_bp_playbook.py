"""proc.bp_playbook and proc.bp_playbook_proposal, in both live databases.

Needs PROCWISE_TEST_LIVE_DB=1; without it pytest uses a fake DB that cannot
answer any of these questions.
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


def _connect(dbname, readonly=True):
    import psycopg2

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        dbname=dbname,
        connect_timeout=10,
    )
    if readonly:
        conn.set_session(readonly=True)
    return conn


@pytest.mark.parametrize("dbname", DATABASES)
def test_both_tables_exist(dbname):
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'proc' "
            "AND table_name IN ('bp_playbook', 'bp_playbook_proposal')"
        )
        found = {r[0] for r in cur.fetchall()}
    assert found == {"bp_playbook", "bp_playbook_proposal"}


@pytest.mark.parametrize("dbname", DATABASES)
def test_indexes_exist(dbname):
    """The unique proposal index especially -- without it a sweep duplicates."""
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT indexname FROM pg_indexes WHERE schemaname = 'proc' "
            "AND tablename IN ('bp_playbook', 'bp_playbook_proposal')"
        )
        found = {r[0] for r in cur.fetchall()}
    for name in (
        "ix_bp_playbook_status",
        "ix_bp_playbook_source",
        "ux_bp_playbook_proposal_finding",
        "ix_bp_playbook_proposal_status",
    ):
        assert name in found, f"{name} missing in {dbname}"


@pytest.mark.parametrize("dbname", DATABASES)
def test_active_requires_an_approver(dbname):
    """An active playbook with no approved_by must be rejected by the database,
    not merely discouraged by the endpoint."""
    with _connect(dbname, readonly=False) as conn:
        cur = conn.cursor()
        cur.execute("SELECT workflow_id FROM proc.bp_agent_workflow LIMIT 1")
        row = cur.fetchone()
        assert row, "no workflow to reference"
        with pytest.raises(Exception) as exc:
            cur.execute(
                "INSERT INTO proc.bp_playbook "
                "(playbook_name, trigger_source, agent_workflow_id, "
                " playbook_status, authored_by) "
                "VALUES ('ck probe', 'detection_finding', %s, 'active', 'tester')",
                (row[0],),
            )
        assert "ck_bp_playbook_active_is_approved" in str(exc.value)
        conn.rollback()


@pytest.mark.parametrize("dbname", DATABASES)
def test_finding_id_is_text_so_both_stores_fit(dbname):
    with _connect(dbname) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT data_type FROM information_schema.columns "
            "WHERE table_schema = 'proc' AND table_name = 'bp_playbook_proposal' "
            "AND column_name = 'finding_id'"
        )
        assert cur.fetchone()[0] == "text"
