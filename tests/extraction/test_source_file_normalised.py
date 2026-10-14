"""One document, one spelling — because source_file is now part of the findings key.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/extraction/test_source_file_normalised.py

099b123 put coalesce(source_file,'') into the open-findings key, which closed the bug where
two documents sharing an invoice number overwrote each other's findings. It also made
source_file load-bearing: from then on, the SAME document arriving under two spellings of
its path would split into two findings instead of refreshing one -- the stacking bug the
key was originally built to prevent (Test Data_300726, 94 duplicated keys).

So the value is normalised at every point it enters a findings row.

WHAT THIS DELIBERATELY DOES NOT DO. It does not reduce the value to a basename. That would
currently be lossless -- measured: 38,511 distinct document references across the invoice,
PO and quote raw tables yield 38,511 distinct basenames -- but it would make the key blind
to the directory, so the day two folders each hold `invoice.pdf` two DIFFERENT documents
would merge and one's findings would be silently discarded. That is exactly the bug
099b123 fixed, re-introduced wearing a different hat. The normaliser fixes SPELLING, never
identity, and `test_normalising_never_merges_two_different_documents` is what holds that
line.

source_file is also not uniformly a path: 4,539 of 5,373 findings rows on bp_testdb carry
`triage:<ref>` synthetic references and 300 carry `s3://` URIs. A normaliser that assumed
"filesystem path" would corrupt those, so both are asserted below.
"""

from __future__ import annotations

import os

import pytest

from src.services.extraction.persistence import normalise_source_file as norm

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")


# --- the spellings that must collapse ------------------------------------------------
@pytest.mark.parametrize("spelled,canonical", [
    ("  documents/invoice/X.pdf  ", "documents/invoice/X.pdf"),   # stray whitespace
    ("documents//invoice/X.pdf", "documents/invoice/X.pdf"),      # doubled separator
    ("./documents/invoice/X.pdf", "documents/invoice/X.pdf"),     # leading ./
    ("/documents/invoice/X.pdf", "documents/invoice/X.pdf"),      # leading /
    ("documents///invoice//X.pdf", "documents/invoice/X.pdf"),    # several of both
])
def test_a_cosmetic_difference_in_spelling_collapses(spelled, canonical):
    assert norm(spelled) == canonical


# --- the shapes that must survive untouched ------------------------------------------
@pytest.mark.parametrize("ref", [
    "triage:DEALV2-000001",                        # a synthetic reference, not a path
    "process_monitor:4821",                        # ditto
    "file:documents/invoice/X.pdf",                # ditto
    "s3://bp-testdata/invoice/INV000001-1A.pdf",   # the scheme separator is not a doubled /
    "documents/invoice/CBC-APP-001_APPLICATION.docx",
    "MASTER Invoice for PO1.pdf",                  # spaces INSIDE a name are part of it
])
def test_a_reference_that_is_already_canonical_is_returned_unchanged(ref):
    assert norm(ref) == ref


def test_an_s3_uri_keeps_its_double_slash_but_still_loses_a_doubled_key_separator():
    assert norm("s3://bucket//invoice//X.pdf") == "s3://bucket/invoice/X.pdf"


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_nothing_is_still_nothing(empty):
    """A blank must not become a path, and must not raise. 8 unresolved rows on bp_sqldb
    have a blank source_file; they fold under coalesce and must keep doing so."""
    assert norm(empty) in (None, "")


def test_normalising_never_merges_two_different_documents():
    """The line this must not cross. Same filename, different directories = different docs."""
    a = norm("documents/2024/invoice.pdf")
    b = norm("documents/2025/invoice.pdf")
    assert a != b, "normalisation collapsed two different documents into one key"


def test_normalising_is_idempotent():
    """It runs on every write, so f(f(x)) must equal f(x) or the key drifts over time."""
    for v in ["./a//b.pdf", "  /x/y.pdf ", "s3://b//k.pdf", "triage:X"]:
        assert norm(norm(v)) == norm(v)


# --- the invariant that says no backfill was needed ----------------------------------
@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_no_value_already_in_the_table_is_changed_by_normalising():
    """If this goes red, the normaliser became lossier and existing rows need a backfill.

    It was true when the normaliser shipped -- 3,797 distinct values on bp_testdb, 292 on
    bp_sqldb, none altered -- which is why there is no data migration. Reducing to a
    basename, for instance, would fail here loudly rather than silently splitting every
    document's findings on its next write.
    """
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT source_file FROM proc.bp_extraction_discrepancy "
                    " WHERE source_file IS NOT NULL")
        values = [r[0] for r in cur.fetchall()]

    assert values, "no rows to check — the guard would pass vacuously"
    changed = [(v, norm(v)) for v in values if norm(v) != v]
    assert changed == [], f"{len(changed)} stored value(s) would change, e.g. {changed[:3]}"


# --- the wire-in: a correct normaliser nobody calls protects nothing -----------------
@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_two_spellings_of_one_document_refresh_one_finding(monkeypatch):
    """Through the real writer, against the live index. One document, one row.

    Without normalisation at the write boundary these two spellings are different key
    values, so the second write INSERTS instead of refreshing and the document's findings
    stack -- exactly the Test Data_300726 regression.
    """
    import uuid
    from tests.extraction.test_discrepancy_key_is_per_document import (
        _in_rolled_back_transaction, _mismatch, _rows)
    from src.services.extraction.persistence import write_discrepancies

    pk = f"TEST-NORM-{uuid.uuid4().hex[:8]}"
    with _in_rolled_back_transaction(monkeypatch) as conn:
        write_discrepancies(doc_type="invoice", raw_id=910001,
                            source_file="documents/invoice/Same.pdf", doc_pk_candidate=pk,
                            discrepancies=[_mismatch("100.00", "90.00")])
        # The same document, spelled differently by a different caller.
        write_discrepancies(doc_type="invoice", raw_id=910002,
                            source_file="./documents//invoice/Same.pdf", doc_pk_candidate=pk,
                            discrepancies=[_mismatch("111.00", "90.00")])

        rows = _rows(conn, pk)

    assert len(rows) == 1, f"one document's findings stacked under two spellings: {rows}"
    assert rows[0][0] == "documents/invoice/Same.pdf", "the stored value was not canonical"
    assert rows[0][1].strip() == "111.00", "the second write did not refresh the first"


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_promotions_own_upsert_normalises_too():
    """promotion._DISCREPANCY_UPSERT is the SECOND writer and needs its own guard.

    The existing rerun test passes an already-canonical 'test.xlsx', so it cannot catch a
    promotion site that stopped normalising. This one spells the same document two ways
    across two reads, which is what a second writer disagreeing with the first looks like.
    """
    import uuid
    from src.services.db import get_conn
    from src.services.extraction.promotion import _check_tax_total_consistency

    pk = f"TEST-PNORM-{uuid.uuid4().hex[:8]}"
    base = {"quote_id": pk, "total_amount": "1116000", "tax_amount": "223200",
            "total_amount_incl_tax": "1000000"}
    with get_conn() as conn:
        conn.autocommit = False
        try:
            cur = conn.cursor()
            _check_tax_total_consistency(
                cur, "quote", 1, dict(base, source_file="documents/quote/Q.xlsx"))
            _check_tax_total_consistency(
                cur, "quote", 2, dict(base, source_file="./documents//quote/Q.xlsx"))
            cur.execute(
                "SELECT source_file FROM proc.bp_extraction_discrepancy "
                " WHERE doc_pk_candidate = %s AND issue_type = 'sum_mismatch'", (pk,))
            rows = cur.fetchall()
        finally:
            conn.rollback()

    assert len(rows) == 1, f"promotion stacked one document under two spellings: {rows}"
    assert rows[0][0] == "documents/quote/Q.xlsx"
