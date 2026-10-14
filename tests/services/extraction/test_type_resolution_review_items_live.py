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


def _write(pk, resolution, raw_id, source_file=None):
    """One write of a document's type findings.

    ``source_file`` defaults to a value fixed per ``pk`` because it is part of the
    open-row key (`2026-10-01_discrepancy_open_key_per_document.sql`): a finding
    belongs to a DOCUMENT, and `doc_pk_candidate` is a number read out of the page
    rather than an identifier of it. A re-read of the same document keeps the same
    source_file and therefore refreshes; two different files are two documents and
    must not collide. Pass ``source_file`` explicitly to exercise that second case.
    """
    return persistence.write_discrepancies(
        doc_type="invoice", raw_id=raw_id,
        source_file=source_file or f"{pk}.pdf",
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


def _raw_ids(pk):
    from src.services.db import get_conn
    with get_conn() as c:
        cur = c.cursor()
        cur.execute(
            "SELECT raw_id FROM proc.bp_extraction_discrepancy "
            "WHERE doc_pk_candidate=%s ORDER BY raw_id", (pk,))
        return [r[0] for r in cur.fetchall()]


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
    assert rows[0][6] == f"{pk}.pdf", rows
    # The surviving row must carry the LATEST run, not the first: raw_id is the
    # per-run identity, and it changes on every re-read.
    assert _raw_ids(pk) == [3], "the surviving row must carry the latest run"


def test_two_different_files_sharing_a_read_value_are_two_findings(pk):
    """The collision the source_file half of the key exists to stop.

    Two different documents can carry the same invoice number — that is the case
    duplicate_invoice_detector exists for — and before source_file joined the key
    the later one silently overwrote the earlier.
    """
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V)
    _write(pk, r, 1, source_file=f"{pk}_first.pdf")
    _write(pk, r, 2, source_file=f"{pk}_second.pdf")
    rows = _rows(pk)
    assert len(rows) == 2, rows
    assert {row[6] for row in rows} == {f"{pk}_first.pdf", f"{pk}_second.pdf"}


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
