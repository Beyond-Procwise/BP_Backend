"""Integration test for 2026-09-24_bp_triage.sql (runs against the .env database)."""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402
from tests.conftest import GOVERNED_LIMIT_SEED  # noqa: E402

MIGRATION = (Path(__file__).resolve().parents[2]
             / "deploy" / "sql" / "2026-09-24_bp_triage.sql")


def _conn():
    s = Settings()
    c = psycopg2.connect(host=s.db_host, dbname=s.db_name,
                         user=s.db_user, password=s.db_password, port=s.db_port)
    c.autocommit = True
    return c


@pytest.fixture(scope="module")
def applied():
    conn = _conn()
    conn.cursor().execute(MIGRATION.read_text())
    conn.cursor().execute(MIGRATION.read_text())   # idempotent: a second apply is harmless
    yield conn
    conn.close()


def test_tables_exist(applied):
    cur = applied.cursor()
    cur.execute("""SELECT table_name FROM information_schema.tables
                   WHERE table_schema='proc' AND table_name LIKE 'bp_triage_%'""")
    assert {r[0] for r in cur.fetchall()} >= {
        "bp_triage_run", "bp_triage_result", "bp_triage_finding"}


def test_policy_row_matches_the_test_seed(applied):
    cur = applied.cursor()
    cur.execute("""SELECT policy_details->'rules' FROM proc.bp_policy
                   WHERE policy_type='limit'
                     AND policy_details->>'policy_identifier'='triage_tolerances'""")
    rows = cur.fetchall()
    assert len(rows) == 1
    assert rows[0][0] == GOVERNED_LIMIT_SEED["triage_tolerances"]


def test_deal_state_table_has_the_scheduler_columns(applied):
    cur = applied.cursor()
    cur.execute("""SELECT column_name, data_type, is_nullable FROM information_schema.columns
                   WHERE table_schema='proc' AND table_name='bp_triage_deal_state'""")
    cols = {name: (dtype, nullable) for name, dtype, nullable in cur.fetchall()}
    assert cols == {
        "deal_id": ("character varying", "NO"),
        "content_hash": ("character varying", "NO"),
        "config_fingerprint": ("character varying", "NO"),
        "last_run_id": ("uuid", "NO"),
        "triaged_at": ("timestamp with time zone", "NO"),
    }


def test_rollback_sql_drops_the_deal_state_table():
    text = (MIGRATION.parent / "2026-09-24_bp_triage_rollback.sql").read_text()
    assert "DROP TABLE IF EXISTS proc.bp_triage_deal_state;" in text


def test_finding_map_records_the_action_centre_mirror_rows(applied):
    cur = applied.cursor()
    cur.execute("""SELECT column_name, data_type FROM information_schema.columns
                   WHERE table_schema='proc' AND table_name='bp_triage_finding'
                     AND column_name IN ('mirror_id', 'replaced_mirror_id')""")
    assert dict(cur.fetchall()) == {"mirror_id": "bigint", "replaced_mirror_id": "bigint"}


def test_rollback_sql_deletes_untouched_mirror_rows_before_dropping_the_map():
    text = (MIGRATION.parent / "2026-09-24_bp_triage_rollback.sql").read_text()
    delete = ("DELETE FROM proc.bp_extraction_discrepancy WHERE source_file LIKE 'triage:%' "
              "AND status='open'\n   AND resolved_by IS NULL AND query_sent_at IS NULL;")
    assert delete in text
    assert text.index(delete) < text.index("DROP TABLE IF EXISTS proc.bp_triage_finding;")


def test_finding_and_mirror_ids_are_uniquely_indexed(applied):
    """One finding <-> at most one mirror: the sync triggers assume it, so a duplicate
    would mean a decision made on one side never fans out to the other."""
    cur = applied.cursor()
    cur.execute("""SELECT indexname, indexdef FROM pg_indexes
                   WHERE schemaname='proc' AND tablename='bp_triage_finding'
                     AND indexname IN ('ix_bp_triage_finding_finding_unique',
                                       'ix_bp_triage_finding_mirror_unique')""")
    defs = dict(cur.fetchall())
    assert set(defs) == {"ix_bp_triage_finding_finding_unique",
                          "ix_bp_triage_finding_mirror_unique"}
    assert "CREATE UNIQUE INDEX" in defs["ix_bp_triage_finding_finding_unique"]
    assert "CREATE UNIQUE INDEX" in defs["ix_bp_triage_finding_mirror_unique"]
    assert "mirror_id IS NOT NULL" in defs["ix_bp_triage_finding_mirror_unique"]
    cur.execute("""SELECT indexname FROM pg_indexes
                   WHERE schemaname='proc' AND tablename='bp_triage_finding'
                     AND indexname IN ('ix_bp_triage_finding_finding', 'ix_bp_triage_finding_mirror')""")
    assert cur.fetchall() == []   # the old plain indexes are gone, not just superseded


def test_decision_sync_triggers_exist(applied):
    cur = applied.cursor()
    cur.execute("""SELECT tgname, tgrelid::regclass::text FROM pg_trigger
                    WHERE tgname IN ('tr_bp_triage_finding_decision_to_mirror',
                                     'tr_bp_triage_mirror_decision_to_finding')
                      AND NOT tgisinternal""")
    assert dict(cur.fetchall()) == {
        "tr_bp_triage_finding_decision_to_mirror": "proc.bp_detection_finding",
        "tr_bp_triage_mirror_decision_to_finding": "proc.bp_extraction_discrepancy",
    }


def test_rollback_sql_drops_the_decision_sync_triggers_first():
    text = (MIGRATION.parent / "2026-09-24_bp_triage_rollback.sql").read_text()
    drops = [
        "DROP TRIGGER IF EXISTS tr_bp_triage_finding_decision_to_mirror "
        "ON proc.bp_detection_finding;",
        "DROP TRIGGER IF EXISTS tr_bp_triage_mirror_decision_to_finding "
        "ON proc.bp_extraction_discrepancy;",
        "DROP FUNCTION IF EXISTS proc.bp_triage_finding_decision_to_mirror();",
        "DROP FUNCTION IF EXISTS proc.bp_triage_mirror_decision_to_finding();",
    ]
    assert all(d in text for d in drops)
    assert max(text.index(d) for d in drops) < text.index("DELETE FROM")
