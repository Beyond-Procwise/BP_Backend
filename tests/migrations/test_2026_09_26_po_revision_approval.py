"""Integration test for 2026-09-26_po_revision_approval.sql (runs against the .env database)."""
from __future__ import annotations

import sys
from pathlib import Path

import psycopg2
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import Settings  # noqa: E402

MIGRATION = (Path(__file__).resolve().parents[2]
             / "deploy" / "sql" / "2026-09-26_po_revision_approval.sql")


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


@pytest.mark.parametrize("layer", ["raw", "stg", "trgt"])
def test_every_layer_carries_revision_and_approval(applied, layer):
    cur = applied.cursor()
    cur.execute("""SELECT column_name, data_type FROM information_schema.columns
                   WHERE table_schema='proc' AND table_name=%s
                     AND column_name IN ('po_revision','approval_status')""", (f"bp_purchase_order_{layer}",))
    assert dict(cur.fetchall()) == {"po_revision": "integer", "approval_status": "text"}


def test_existing_rows_read_as_unstated(applied):
    cur = applied.cursor()
    cur.execute("SELECT count(*) FROM proc.bp_purchase_order_trgt WHERE approval_status IS NOT NULL OR po_revision IS NOT NULL")
    assert cur.fetchone()[0] == 0   # nothing is back-filled: NULL means "the document does not say"


def test_approval_values_are_closed(applied):
    cur = applied.cursor()
    cur.execute("""SELECT pg_get_constraintdef(oid) FROM pg_constraint
                   WHERE conname = 'bp_purchase_order_trgt_approval_chk'""")
    assert "approved" in cur.fetchone()[0]
