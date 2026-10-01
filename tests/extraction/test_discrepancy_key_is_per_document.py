"""Two documents sharing one invoice number are two findings, not one.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/extraction/test_discrepancy_key_is_per_document.py

The open-findings key was (doc_type, doc_pk_candidate, issue_type, field_name).
`doc_pk_candidate` is the invoice number read OUT of the document, not an
identifier OF the document -- so two DIFFERENT documents carrying the same
invoice number collided, and the later one's ON CONFLICT DO UPDATE overwrote the
earlier one's finding in place, with nothing recording that it had existed.

This is not hypothetical and not confined to seeded data. bp_testdb holds two
documents under invoice 01-2024-002 -- PO1_IT_Invoice_LowerCost.pdf (total
15,877.50) and MASTER Invoice for PO1.pdf (17,065.50) -- and findings for only
the second. And the canonical real case for one invoice number arriving on two
documents is the same invoice submitted twice, which is exactly what
duplicate_invoice_detector.py exists to catch: the findings layer was keeping
only the last of them.

The fix adds coalesce(source_file, '') to the key. It must NOT cost the property
the key was built for -- re-reading the SAME document (new raw_id, same
source_file) still refreshes rather than stacks -- so both directions are
asserted here.

Runs inside a transaction that is rolled back: nothing is left behind.
"""

from __future__ import annotations

import os
import uuid
from contextlib import contextmanager

import pytest

from src.services.db import get_conn
from src.services.extraction import persistence
from src.services.extraction.persistence import Discrepancy, write_discrepancies

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

DOC_A = "documents/invoice/PO1_IT_Invoice_LowerCost.pdf"
DOC_B = "documents/invoice/MASTER Invoice for PO1.pdf"


def _mismatch(total: str, expected: str) -> Discrepancy:
    return Discrepancy(field_name="invoice_total_incl_tax", issue_type="sum_mismatch",
                       severity="warning", blocks_promotion=False,
                       raw_value=total, expected_value=expected, notes="t")


class _NoCommit:
    """The live connection with `commit` neutered, everything else passed through.

    psycopg2 connection attributes are read-only, so commit cannot be monkeypatched
    onto the real object -- it has to be wrapped. Neutering commit rather than
    deferring the rollback is deliberate: write_discrepancies commits on its own, and
    a real commit here would leave test rows in the live table.
    """

    def __init__(self, conn): self._conn = conn
    def commit(self): pass
    def __getattr__(self, name): return getattr(self._conn, name)


@contextmanager
def _in_rolled_back_transaction(monkeypatch):
    """Run the REAL writer against the REAL database, then undo it.

    Only the connection is substituted. The SQL, the index it conflicts against and
    the triggers on the table are all production.
    """
    with get_conn() as conn:
        conn.autocommit = False
        wrapped = _NoCommit(conn)

        @contextmanager
        def _conn():
            yield wrapped

        monkeypatch.setattr(persistence, "get_conn", _conn)
        monkeypatch.setattr(persistence, "bulk_record", lambda *a, **k: None)
        try:
            yield conn
        finally:
            conn.rollback()


def _rows(conn, pk):
    cur = conn.cursor()
    cur.execute(
        "SELECT source_file, raw_value FROM proc.bp_extraction_discrepancy "
        " WHERE doc_pk_candidate = %s AND issue_type = 'sum_mismatch' "
        " AND coalesce(status,'open') <> 'resolved' ORDER BY source_file", (pk,))
    return cur.fetchall()


def test_a_second_document_does_not_overwrite_the_first_ones_finding(monkeypatch):
    pk = f"TEST-KEY-{uuid.uuid4().hex[:8]}"
    with _in_rolled_back_transaction(monkeypatch) as conn:
        write_discrepancies(doc_type="invoice", raw_id=900001, source_file=DOC_A,
                            doc_pk_candidate=pk,
                            discrepancies=[_mismatch("15877.50", "15757.50")])
        write_discrepancies(doc_type="invoice", raw_id=900002, source_file=DOC_B,
                            doc_pk_candidate=pk,
                            discrepancies=[_mismatch("17065.50", "16945.50")])

        rows = _rows(conn, pk)

    assert len(rows) == 2, f"one document's finding was overwritten: {rows}"
    assert {r[0] for r in rows} == {DOC_A, DOC_B}
    # The figures are what makes them different findings rather than one restated.
    assert {r[1].strip() for r in rows} == {"15877.50", "17065.50"}


def test_re_reading_the_same_document_still_refreshes_rather_than_stacks(monkeypatch):
    """The property the key was built for. Widening it must not cost this."""
    pk = f"TEST-KEY-{uuid.uuid4().hex[:8]}"
    with _in_rolled_back_transaction(monkeypatch) as conn:
        write_discrepancies(doc_type="invoice", raw_id=900003, source_file=DOC_A,
                            doc_pk_candidate=pk,
                            discrepancies=[_mismatch("15877.50", "15757.50")])
        # Same document, re-extracted: a NEW raw_id, the SAME source_file.
        write_discrepancies(doc_type="invoice", raw_id=900004, source_file=DOC_A,
                            doc_pk_candidate=pk,
                            discrepancies=[_mismatch("15999.99", "15757.50")])

        rows = _rows(conn, pk)

    assert len(rows) == 1, f"a re-read stacked a duplicate: {rows}"
    assert rows[0][1].strip() == "15999.99", "the re-read did not refresh the figure"


def test_the_upsert_clause_names_the_document(monkeypatch):
    """Cheap guard, no database: the clause and the index must agree on the key.

    They are matched by SHAPE. If this clause and ix_bp_extraction_discrepancy_open_key
    ever disagree, Postgres does not warn -- it rejects every write with "there is no
    unique or exclusion constraint matching the ON CONFLICT specification", which is
    how bp_sqldb silently recorded nothing for two months.
    """
    captured = {}

    class _Cur:
        def executemany(self, sql, rows): captured["sql"] = sql
        def execute(self, sql, params=()): pass
        def fetchone(self): return None

    class _Conn:
        autocommit = False
        def cursor(self): return _Cur()
        def commit(self): pass
        def rollback(self): pass

    @contextmanager
    def _conn():
        yield _Conn()

    monkeypatch.setattr(persistence, "get_conn", _conn)
    monkeypatch.setattr(persistence, "bulk_record", lambda *a, **k: None)
    write_discrepancies(doc_type="invoice", raw_id=1, source_file="f.pdf",
                        doc_pk_candidate="INV-1",
                        discrepancies=[_mismatch("1", "2")])

    clause = " ".join(captured["sql"].lower().split())
    conflict = clause.split("on conflict", 1)[1].split("do update", 1)[0]
    assert "source_file" in conflict, f"the conflict target omits the document: {conflict}"
