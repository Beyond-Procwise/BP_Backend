"""The type findings, written through the real write path into the real queue.

write_discrepancies commits on its own connection, so the probe queue rows are
deleted on exit. The proc.bp_agent_actions audit rows it also writes are
append-only (DELETE is refused by trigger) and stay: doc_pk 'PROBE-TYPE-*'.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = [pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1"),
              pytest.mark.integration]

from src.services.concepts.vocabulary import SEED_VOCABULARY as V  # noqa: E402
from src.services.extraction import persistence  # noqa: E402
from src.services.extraction.type_resolver import (  # noqa: E402
    resolve_document_type, type_resolution_discrepancies,
)

INVOICE_PAGE = "TAX INVOICE\nInvoice No: INV-1\nAmount due: 10.00\n"


@pytest.fixture()
def pk():
    from src.services.db import get_conn
    key = f"PROBE-TYPE-{uuid.uuid4().hex[:10]}"
    yield key
    with get_conn() as c:
        cur = c.cursor()
        cur.execute("DELETE FROM proc.bp_extraction_discrepancy WHERE doc_pk_candidate=%s", (key,))


def _write(pk, resolution, raw_id):
    return persistence.write_discrepancies(
        doc_type="invoice", raw_id=raw_id, source_file=f"probe_{raw_id}.pdf",
        doc_pk_candidate=pk, discrepancies=type_resolution_discrepancies(resolution))


def _rows(pk):
    from src.services.db import get_conn
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(
            "SELECT issue_type, field_name, blocks_promotion, raw_value, expected_value, "
            "evidence_text, source_file, status FROM proc.bp_extraction_discrepancy "
            "WHERE doc_pk_candidate=%s ORDER BY issue_type", (pk,))
        return cur.fetchall()


def test_a_disagreement_is_recorded_faithfully_as_one_non_blocking_row(pk):
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V)
    _write(pk, r, 1)
    (row,) = _rows(pk)
    assert row[0] == "document_type_disagreement" and row[1] == "document_type"
    assert row[2] is False
    assert (row[3], row[4]) == ("doctype.quote", "doctype.invoice")
    assert row[5] and row[5] in INVOICE_PAGE


def test_re_extraction_refreshes_the_open_finding_instead_of_stacking(pk):
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V)
    _write(pk, r, 1)
    _write(pk, r, 2)
    _write(pk, r, 3)
    rows = _rows(pk)
    assert len(rows) == 1, rows
    assert rows[0][6] == "probe_3.pdf", "the surviving row must carry the latest run"


def test_two_different_type_findings_are_two_rows_not_one(pk):
    """Distinct issue types stay apart under the same field_name."""
    a = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V)
    b = resolve_document_type(
        declared_concept="doctype.sow", full_text="INVOICE / QUOTE\n", vocabulary=V)
    _write(pk, a, 1)
    _write(pk, b, 2)
    assert [r[0] for r in _rows(pk)] == [
        "document_type_disagreement", "unresolved_document_type"]
