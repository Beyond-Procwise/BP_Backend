"""Integration test for 2026-05-16-extraction-discrepancy-hitl.sql.

Asserts:
- New columns are added (additive)
- Existing 1,543 discrepancy rows are preserved
- resolution_action CHECK enforces the enum
- status CHECK now accepts 'superseded'
- Trigger fires NOTIFY on the right transition AND only when the row had
  blocks_promotion=TRUE
- Trigger does NOT fire when blocks_promotion=FALSE (protects legacy rows)
"""
from __future__ import annotations

import json
import select
import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402

MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "scripts" / "migrations" / "2026-05-16-extraction-discrepancy-hitl.sql"
)


def _conn(autocommit=True):
    s = Settings()
    c = psycopg2.connect(
        host=s.db_host, dbname=s.db_name,
        user=s.db_user, password=s.db_password, port=s.db_port,
    )
    c.autocommit = autocommit
    return c


def _columns(cur, t):
    cur.execute(
        """SELECT column_name FROM information_schema.columns
            WHERE table_schema='proc' AND table_name=%s""", (t,))
    return {r[0] for r in cur.fetchall()}


@pytest.fixture(scope="module")
def applied():
    c = _conn()
    c.cursor().execute(MIGRATION.read_text())
    yield c
    c.close()


def test_new_columns_added(applied):
    cols = _columns(applied.cursor(), "bp_extraction_discrepancy")
    for c in ("resolved_value", "resolution_action", "resolved_by",
              "blocks_promotion", "evidence_page", "evidence_bbox",
              "evidence_text"):
        assert c in cols, f"missing column {c}"


def test_existing_rows_preserved(applied):
    cur = applied.cursor()
    cur.execute("SELECT COUNT(*) FROM proc.bp_extraction_discrepancy")
    n = cur.fetchone()[0]
    # baseline pre-migration: 1543
    assert n >= 1543, f"row loss: {n}"


def test_existing_rows_default_blocks_false(applied):
    """Legacy rows must default to blocks_promotion=FALSE to avoid spurious NOTIFY."""
    cur = applied.cursor()
    cur.execute("SELECT COUNT(*) FROM proc.bp_extraction_discrepancy WHERE blocks_promotion = TRUE")
    assert cur.fetchone()[0] == 0, "legacy rows must default to blocks_promotion=FALSE"


def test_status_enum_accepts_superseded(applied):
    cur = applied.cursor()
    # build a fresh _raw row to attach to
    cur.execute("""
        INSERT INTO proc.bp_invoice_raw (source_file, pipeline_version, raw_payload)
        VALUES ('/tmp/x', 'test-superseded', '{}'::jsonb) RETURNING raw_id""")
    raw_id = cur.fetchone()[0]
    cur.execute("""
        INSERT INTO proc.bp_extraction_discrepancy
          (doc_type, raw_id, source_file, field_name, issue_type, severity, status)
        VALUES ('invoice', %s, '/tmp/x', 'f', 'invariant_failed', 'critical', 'superseded')
        RETURNING discrepancy_id""", (raw_id,))
    disc_id = cur.fetchone()[0]
    cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s", (disc_id,))
    cur.execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id=%s", (raw_id,))


def test_trigger_fires_on_blocking_resolved(applied):
    cur = applied.cursor()
    cur.execute("""
        INSERT INTO proc.bp_invoice_raw (source_file, pipeline_version, raw_payload)
        VALUES ('/tmp/trg', 'test-trg', '{}'::jsonb) RETURNING raw_id""")
    raw_id = cur.fetchone()[0]
    cur.execute("""
        INSERT INTO proc.bp_extraction_discrepancy
          (doc_type, raw_id, source_file, field_name, issue_type, severity,
           blocks_promotion)
        VALUES ('invoice', %s, '/tmp/trg', 'invoice_amount',
                'invariant_failed', 'critical', TRUE)
        RETURNING discrepancy_id""", (raw_id,))
    disc_id = cur.fetchone()[0]

    # New connection for LISTEN — psycopg2's LISTEN/NOTIFY needs its own conn
    listener = _conn(autocommit=True)
    lcur = listener.cursor()
    lcur.execute("LISTEN extraction_raw_ready_for_promotion;")

    cur.execute("""
        UPDATE proc.bp_extraction_discrepancy
           SET status='resolved', resolved_value='150.00',
               resolution_action='apply_value', resolved_by='test'
         WHERE discrepancy_id=%s""", (disc_id,))

    # Wait up to 3s for the notification
    got = None
    if select.select([listener], [], [], 3.0)[0]:
        listener.poll()
        if listener.notifies:
            got = listener.notifies.pop(0)

    # Cleanup
    cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s", (disc_id,))
    cur.execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id=%s", (raw_id,))
    listener.close()

    assert got is not None, "expected NOTIFY on extraction_raw_ready_for_promotion"
    payload = json.loads(got.payload)
    assert payload["raw_id"] == raw_id
    assert payload["doc_type"] == "invoice"


def test_trigger_silent_when_blocks_false(applied):
    """Resolving a NON-blocking discrepancy must NOT fire NOTIFY."""
    cur = applied.cursor()
    cur.execute("""
        INSERT INTO proc.bp_invoice_raw (source_file, pipeline_version, raw_payload)
        VALUES ('/tmp/sil', 'test-silent', '{}'::jsonb) RETURNING raw_id""")
    raw_id = cur.fetchone()[0]
    cur.execute("""
        INSERT INTO proc.bp_extraction_discrepancy
          (doc_type, raw_id, source_file, field_name, issue_type, severity,
           blocks_promotion)
        VALUES ('invoice', %s, '/tmp/sil', 'tax_amount',
                'value_out_of_range', 'warning', FALSE)
        RETURNING discrepancy_id""", (raw_id,))
    disc_id = cur.fetchone()[0]

    listener = _conn(autocommit=True)
    lcur = listener.cursor()
    lcur.execute("LISTEN extraction_raw_ready_for_promotion;")

    cur.execute("""UPDATE proc.bp_extraction_discrepancy
                      SET status='resolved', resolution_action='dismiss',
                          resolved_by='test'
                    WHERE discrepancy_id=%s""", (disc_id,))

    got = None
    if select.select([listener], [], [], 2.0)[0]:
        listener.poll()
        if listener.notifies:
            got = listener.notifies.pop(0)

    cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s", (disc_id,))
    cur.execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id=%s", (raw_id,))
    listener.close()

    assert got is None, "non-blocking resolution must not NOTIFY"


def test_trigger_silent_when_other_blocking_still_open(applied):
    """If a blocking row resolves but another blocking row is still open,
       NOTIFY must NOT fire."""
    cur = applied.cursor()
    cur.execute("""
        INSERT INTO proc.bp_invoice_raw (source_file, pipeline_version, raw_payload)
        VALUES ('/tmp/two', 'test-two', '{}'::jsonb) RETURNING raw_id""")
    raw_id = cur.fetchone()[0]
    cur.execute("""
        INSERT INTO proc.bp_extraction_discrepancy
          (doc_type, raw_id, source_file, field_name, issue_type, severity, blocks_promotion)
        VALUES ('invoice', %s, '/tmp/two', 'a', 'invariant_failed', 'critical', TRUE),
               ('invoice', %s, '/tmp/two', 'b', 'invariant_failed', 'critical', TRUE)
        RETURNING discrepancy_id""", (raw_id, raw_id))
    rows = [r[0] for r in cur.fetchall()]
    first = rows[0]
    other = rows[1] if len(rows) > 1 else None
    # If RETURNING returned only one (driver behaviour), fetch the rest
    if other is None:
        cur.execute("SELECT discrepancy_id FROM proc.bp_extraction_discrepancy WHERE raw_id=%s ORDER BY discrepancy_id", (raw_id,))
        rows = [r[0] for r in cur.fetchall()]
        first, other = rows[0], rows[1]

    listener = _conn(autocommit=True)
    lcur = listener.cursor()
    lcur.execute("LISTEN extraction_raw_ready_for_promotion;")

    cur.execute("""UPDATE proc.bp_extraction_discrepancy
                      SET status='resolved', resolution_action='dismiss',
                          resolved_by='test'
                    WHERE discrepancy_id=%s""", (first,))

    got = None
    if select.select([listener], [], [], 2.0)[0]:
        listener.poll()
        if listener.notifies:
            got = listener.notifies.pop(0)

    cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE raw_id=%s", (raw_id,))
    cur.execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id=%s", (raw_id,))
    listener.close()

    assert got is None, "NOTIFY fired prematurely while a sibling blocking-open remained"


def test_resolution_action_check(applied):
    cur = applied.cursor()
    cur.execute("""
        INSERT INTO proc.bp_invoice_raw (source_file, pipeline_version, raw_payload)
        VALUES ('/tmp/ra', 'test-ra', '{}'::jsonb) RETURNING raw_id""")
    raw_id = cur.fetchone()[0]
    # 'apply_value' / 'keep_null' / 'dismiss' allowed; others must fail
    cur.execute("""
        INSERT INTO proc.bp_extraction_discrepancy
          (doc_type, raw_id, source_file, field_name, issue_type, severity,
           resolution_action)
        VALUES ('invoice', %s, '/tmp/ra', 'f', 'invariant_failed',
                'critical', 'apply_value') RETURNING discrepancy_id""", (raw_id,))
    ok_id = cur.fetchone()[0]
    cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE discrepancy_id=%s", (ok_id,))
    with pytest.raises(psycopg2.errors.CheckViolation):
        cur.execute("""
            INSERT INTO proc.bp_extraction_discrepancy
              (doc_type, raw_id, source_file, field_name, issue_type, severity,
               resolution_action)
            VALUES ('invoice', %s, '/tmp/ra', 'f', 'invariant_failed',
                    'critical', 'invalid_action')""", (raw_id,))
    # rollback the failed statement (psycopg2 marks the connection as needing rollback)
    applied.rollback()
    cur = applied.cursor()
    cur.execute("DELETE FROM proc.bp_invoice_raw WHERE raw_id=%s", (raw_id,))
