"""The precedent count as a governed limit (design §3.3), in both live databases.

Needs PROCWISE_TEST_LIVE_DB=1. Read-only against the shared databases, except the tests that run
the migration files inside a transaction that is always rolled back.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1",
    reason="live database required (PROCWISE_TEST_LIVE_DB=1)",
)
DATABASES = ("bp_testdb", "bp_sqldb")
ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "deploy/sql/2026-10-13_agent_policy_conflict_precedent.sql"
DOWN = ROOT / "deploy/sql/2026-10-13_agent_policy_conflict_precedent_rollback.sql"
SLUG = "agent_policy_conflicts"


def _connect(dbname):
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"), password=os.getenv("DB_PASSWORD"),
        dbname=dbname, connect_timeout=10,
    )


def _body(path):
    """The file's statements without its own BEGIN/COMMIT, to run inside a rolled-back transaction."""
    return "\n".join(line for line in path.read_text().splitlines()
                     if line.strip().upper() not in ("BEGIN;", "COMMIT;"))


def _rows(cur):
    cur.execute("SELECT policy_name, policy_type, policy_status, policy_details, created_by "
                "FROM proc.bp_policy WHERE policy_details->>'policy_identifier' = %s ORDER BY policy_id", (SLUG,))
    return cur.fetchall()


def _setting(cur):
    cur.execute("SELECT config_value->>'live_conflict_repeat' FROM proc.bp_admin_config "
                "WHERE config_key = 'agent_policy_settings'")
    row = cur.fetchone()
    return int(row[0]) if row and row[0] and row[0].isdigit() else 5


@pytest.mark.parametrize("db", DATABASES)
def test_the_row_exists_with_the_copied_value(db):
    conn = _connect(db)
    try:
        with conn.cursor() as cur:
            active = [r for r in _rows(cur) if r[2] == 1]
            assert len(active) == 1, f"{db}: expected one active {SLUG} row, found {len(active)}"
            name, ptype, _status, details, created_by = active[0]
            assert (name, ptype, created_by) == ("AgentPolicyConflictPolicy", "limit", "agent_policy_conflicts")
            assert details["policy_identifier"] == SLUG
            assert "applies_to" not in details, "configuration read by name, never an authority statement"
            assert details["rules"] == {"precedent_count": _setting(cur)}
    finally:
        conn.close()


@pytest.mark.parametrize("db", DATABASES)
def test_the_policy_engine_reads_it(db):
    from src.engines.policy_engine import PolicyEngine
    conn = _connect(db)
    try:
        engine = PolicyEngine(connection_factory=lambda: conn)
        row = engine.get_policy(SLUG)
        assert row is not None and row["details"]["rules"]["precedent_count"] >= 0
    finally:
        conn.close()


def test_applying_again_adds_nothing_and_the_rollback_removes_it():
    conn = _connect("bp_testdb")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            assert len([r for r in _rows(cur) if r[2] == 1]) == 1
            cur.execute(_body(UP))
            cur.execute(_body(UP))
            assert len([r for r in _rows(cur) if r[2] == 1]) == 1, "idempotent"
            cur.execute(_body(DOWN))
            assert _rows(cur) == [], "the rollback removes every version of the row"
    finally:
        conn.rollback()
        conn.close()


def test_a_first_apply_copies_the_company_setting():
    conn = _connect("bp_testdb")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(_body(DOWN))
            cur.execute("UPDATE proc.bp_admin_config SET config_value = jsonb_set(config_value, "
                        "'{live_conflict_repeat}', '7') WHERE config_key = 'agent_policy_settings'")
            cur.execute(_body(UP))
            [(_n, _t, _s, details, _c)] = _rows(cur)
            assert details["rules"] == {"precedent_count": 7}
            cur.execute(_body(DOWN))
            cur.execute("UPDATE proc.bp_admin_config SET config_value = config_value - 'live_conflict_repeat' "
                        "WHERE config_key = 'agent_policy_settings'")
            cur.execute(_body(UP))
            [(_n, _t, _s, details, _c)] = _rows(cur)
            assert details["rules"] == {"precedent_count": 5}, "no setting: the ruled default"
    finally:
        conn.rollback()
        conn.close()
