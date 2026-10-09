"""The precedent value range as a governed limit (Task 12), in both live databases.

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
UP = ROOT / "deploy/sql/2026-10-14_agent_policy_conflict_value_range.sql"
DOWN = ROOT / "deploy/sql/2026-10-14_agent_policy_conflict_value_range_rollback.sql"
SLUG = "agent_policy_conflicts"
RULE = "precedent_value_range_pct"


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


def _active(cur):
    cur.execute("SELECT policy_id, policy_details, policy_desc, version FROM proc.bp_policy "
                "WHERE policy_details->>'policy_identifier' = %s AND policy_status = 1 ORDER BY policy_id", (SLUG,))
    return cur.fetchall()


def _set(cur, value_sql):
    cur.execute(f"UPDATE proc.bp_policy SET policy_details = jsonb_set(policy_details, '{{rules,{RULE}}}', "
                f"{value_sql}) WHERE policy_details->>'policy_identifier' = %s AND policy_status = 1", (SLUG,))


@pytest.mark.parametrize("db", DATABASES)
def test_the_row_states_the_range_and_says_what_it_does(db):
    conn = _connect(db)
    try:
        with conn.cursor() as cur:
            [(_id, details, desc, _v)] = _active(cur)
            rules = details["rules"]
            assert RULE in rules, f"{db}: {RULE} is not stated"
            assert rules[RULE] is None or (isinstance(rules[RULE], (int, float)) and rules[RULE] >= 0)
            assert isinstance(rules["precedent_count"], int), "precedent_count is untouched"
            for words in ("0 means not above the largest approved value", "null means no range check"):
                assert words in desc, words
    finally:
        conn.close()


@pytest.mark.parametrize("db", DATABASES)
def test_the_policy_engine_reads_it(db):
    from src.engines.policy_engine import PolicyEngine
    conn = _connect(db)
    try:
        row = PolicyEngine(connection_factory=lambda: conn).get_policy(SLUG)
        assert row is not None and RULE in row["details"]["rules"]
    finally:
        conn.close()


def test_a_first_apply_adds_20_and_a_second_changes_nothing():
    conn = _connect("bp_testdb")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            cur.execute(_body(DOWN))
            [(_id, details, _d, version)] = _active(cur)
            assert RULE not in details["rules"]
            cur.execute(_body(UP))
            [first] = _active(cur)
            assert first[1]["rules"][RULE] == 20 and first[3] == version + 1
            assert {k: v for k, v in first[1]["rules"].items() if k != RULE} == details["rules"]
            cur.execute(_body(UP))
            assert _active(cur) == [first], "idempotent: nothing changes on a second run"
    finally:
        conn.rollback()
        conn.close()


@pytest.mark.parametrize("value_sql,expected", [("'35'::jsonb", 35), ("'null'::jsonb", None), ("'0'::jsonb", 0)])
def test_a_customers_value_is_never_overwritten(value_sql, expected):
    conn = _connect("bp_testdb")
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            _set(cur, value_sql)
            [before] = _active(cur)
            cur.execute(_body(UP))
            [after] = _active(cur)
            assert after == before and after[1]["rules"][RULE] == expected
    finally:
        conn.rollback()
        conn.close()
