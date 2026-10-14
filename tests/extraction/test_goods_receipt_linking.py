"""A goods receipt reaches _trgt by its purchase order's deal, or not at all.

The receipt's whole value is that it attaches to the order: a receipt sitting in
`_stg` with no `deal_id` is a document nobody will ever find, and a receipt in
`_trgt` on the wrong deal is worse than one that never promoted. So the two
cases are tested together -- the PO exists and the receipt lands on its deal,
and the PO does not exist and the receipt STAYS in `_stg`.

The PO is resolved with `linking_engine._pick_po`, the same function the
two-way match uses, so a receipt and an invoice citing the same order in
different formats ("PO-4500018832", "4500018832", "po 4500018832") reach the
same row. That shared normalisation is asserted, not assumed.

Live-only. Run with:
    set -a && . ./.env && set +a
    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/extraction/test_goods_receipt_linking.py
"""
from __future__ import annotations

import os
from uuid import uuid4

import pytest

from src.services.db import get_conn

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


def _seed_po(cur, po_id: str, *, deal_id: str) -> None:
    cur.execute(
        "INSERT INTO proc.bp_purchase_order_trgt (po_id, deal_id, deal_name, "
        "supplier_name, expected_delivery_date) "
        "VALUES (%s, %s, %s, 'Northwind Trading Ltd', DATE '2026-03-31')",
        (po_id, deal_id, f"deal for {po_id}"),
    )


def _seed_receipt_in_stg(cur, grn_id: str, *, po_id: str) -> None:
    cur.execute(
        "INSERT INTO proc.bp_goods_receipt_stg (grn_id, po_id, supplier_name) "
        "VALUES (%s, %s, 'Northwind Trading Ltd')", (grn_id, po_id))
    cur.execute(
        "INSERT INTO proc.bp_goods_receipt_line_items_stg "
        "(goods_receipt_line_id, grn_id, line_no, item_description, "
        " quantity_received, unit_of_measure, po_id) "
        "VALUES (%s, %s, 1, 'Widget A', 10, 'each', %s)",
        (f"{grn_id}-L1", grn_id, po_id))


def _cleanup(grn_ids, po_ids) -> None:
    with get_conn() as conn, conn.cursor() as cur:
        for g in grn_ids:
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id=%s", (g,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (g,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_stg WHERE grn_id=%s", (g,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_stg WHERE grn_id=%s", (g,))
        for p in po_ids:
            cur.execute("DELETE FROM proc.bp_purchase_order_trgt WHERE po_id=%s", (p,))


@pytest.fixture()
def tag():
    """Digits only, and PO ids below are built as "45<tag>" -- NOT "POTEST<tag>".

    `_norm_po` strips a leading "po" from the stored id as well as from the
    citation, so a purchase order whose own number genuinely begins "PO" cannot
    be found by a citation that adds the prefix again ("PO-POTEST1" normalises
    to "potest1", the row to "test1"). That is pre-existing shared behaviour of
    the two-way match's normalisation, not something a goods receipt changes,
    and it is not this test's subject -- so the fixture avoids the collision
    rather than pretending it is not there.
    """
    yield f"{uuid4().int % 100000000:08d}"


def test_a_receipt_citing_a_po_lands_on_that_pos_deal(tag):
    from src.services.extraction.goods_receipt_link import link_receipt_to_po

    grn, po, deal = f"GRN-{tag}", f"45{tag}", f"DEALTEST-{tag}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            _seed_po(cur, po, deal_id=deal)
            _seed_receipt_in_stg(cur, grn, po_id=po)

        assert link_receipt_to_po(grn)["linked"] is True

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT po_id, deal_id, deal_name, document_id, deal_date "
                        "FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            row = cur.fetchone()
            assert row is not None, "the receipt never reached _trgt"
            assert row[0] == po
            assert row[1] == deal
            assert row[2] == f"deal for {po}"
            assert row[3] == f"{deal}::goods_receipt::{grn}"
            assert str(row[4]) == "2026-03-31"

            cur.execute("SELECT quantity_received, unit_of_measure, deal_id "
                        "FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id=%s", (grn,))
            assert cur.fetchall() == [(10, "each", deal)]
    finally:
        _cleanup([grn], [po])


def test_a_receipt_whose_po_does_not_exist_stays_in_stg(tag):
    """Absence is not failure and it is not promotion either. A receipt whose
    order we do not hold keeps its _stg row -- it is real, captured evidence --
    and does not invent a deal to sit on."""
    from src.services.extraction.goods_receipt_link import link_receipt_to_po

    grn = f"GRN-ORPHAN-{tag}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            _seed_receipt_in_stg(cur, grn, po_id=f"99{tag}")

        result = link_receipt_to_po(grn)
        assert result["linked"] is False
        assert result["reason"] == "no_matching_po"

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            assert cur.fetchone() is None, "an orphan receipt reached _trgt"
            cur.execute("SELECT 1 FROM proc.bp_goods_receipt_stg WHERE grn_id=%s", (grn,))
            assert cur.fetchone() is not None, "the _stg evidence was discarded"
    finally:
        _cleanup([grn], [])


def test_a_receipt_whose_po_has_no_deal_yet_stays_in_stg(tag):
    """The PO exists but has not been grouped into a deal. There is nothing to
    attach to yet, so the receipt waits rather than landing deal-less in _trgt,
    where every reader keys on deal_id and would simply never see it."""
    from src.services.extraction.goods_receipt_link import link_receipt_to_po

    grn, po = f"GRN-NODEAL-{tag}", f"46{tag}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_purchase_order_trgt (po_id, supplier_name) "
                        "VALUES (%s, 'Northwind Trading Ltd')", (po,))
            _seed_receipt_in_stg(cur, grn, po_id=po)

        result = link_receipt_to_po(grn)
        assert result["linked"] is False
        assert result["reason"] == "po_has_no_deal"

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1 FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            assert cur.fetchone() is None
    finally:
        _cleanup([grn], [po])


@pytest.mark.parametrize("cited", ["{po}", "PO-{po}", "po {po}", "{po} (Rev 1)"])
def test_the_receipt_reaches_the_same_po_an_invoice_would(tag, cited):
    """_pick_po's normalisation, shared with the two-way match: prefix,
    separators and a revision suffix must not change which order is meant."""
    from src.services.extraction.goods_receipt_link import link_receipt_to_po

    grn, po, deal = f"GRN-NORM-{tag}", f"45{tag}", f"DEALTEST-{tag}"
    try:
        with get_conn() as conn, conn.cursor() as cur:
            _seed_po(cur, po, deal_id=deal)
            _seed_receipt_in_stg(cur, grn, po_id=cited.format(po=po))

        assert link_receipt_to_po(grn)["linked"] is True
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT deal_id FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s", (grn,))
            assert cur.fetchone()[0] == deal
            # The CANONICAL po_id must be written back over the spelling the
            # document used, on the header, the lines and the _trgt copies.
            # Without this the three-way match joins receipt lines to PO lines
            # on "PO-4512345678" and finds nothing, and the match reads as
            # "no receipt" -- a silent pass, which is the one failure mode this
            # whole feature must not have.
            for table in ("proc.bp_goods_receipt_stg", "proc.bp_goods_receipt_trgt",
                          "proc.bp_goods_receipt_line_items_stg",
                          "proc.bp_goods_receipt_line_items_trgt"):
                cur.execute(f"SELECT DISTINCT po_id FROM {table} WHERE grn_id=%s", (grn,))
                assert [r[0] for r in cur.fetchall()] == [po], table
    finally:
        _cleanup([grn], [po])


def test_a_receipt_that_could_not_link_is_retried_when_its_po_arrives(tag):
    """A note arriving BEFORE its purchase order waited for ever.

    `link_receipt_to_po` is one best-effort call at ingestion, and every sweep
    in deal_assignment_service iterates a hard-coded ("invoice","quote","po") --
    so nothing ever woke a receipt that reported `no_matching_po` or
    `po_has_no_deal`. It stayed in _stg, invisible to the match and to every
    reader. Found in the whole-branch review of 2026-10-05.
    """
    from src.services.extraction.goods_receipt_link import (
        link_pending_receipts, link_receipt_to_po,
    )

    grn, po, deal = f"GRN-LATE-{tag}", f"44{tag}", f"DEALTEST-{tag}"
    try:
        # The note lands first. Its order does not exist yet.
        with get_conn() as conn, conn.cursor() as cur:
            _seed_receipt_in_stg(cur, grn, po_id=po)
        assert link_receipt_to_po(grn)["reason"] == "no_matching_po"

        # A sweep now changes nothing, and says so rather than failing.
        first = link_pending_receipts()
        assert first["by_reason"].get("no_matching_po", 0) >= 1

        # The order is extracted and grouped.
        with get_conn() as conn, conn.cursor() as cur:
            _seed_po(cur, po, deal_id=deal)

        after = link_pending_receipts()
        assert after["linked"] >= 1, after
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT deal_id FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s",
                        (grn,))
            row = cur.fetchone()
            assert row is not None, "the sweep did not promote the waiting receipt"
            assert row[0] == deal

        # And it does not promote it twice.
        link_pending_receipts()
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM proc.bp_goods_receipt_trgt WHERE grn_id=%s",
                        (grn,))
            assert cur.fetchone()[0] == 1
    finally:
        _cleanup([grn], [po])
