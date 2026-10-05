"""The deal overview separates "the amounts reconcile" from "the goods arrived".

The column called `three_way_match` has never been a three-way match. It checks
that a quote, a PO and an invoice all exist and that the invoice total is within
10% of the PO total -- a VALUE reconciliation across three document types, which
is a real and useful thing, and not evidence that anything was delivered.

So it becomes two columns:
  value_reconciled   exactly what three_way_match computes today, renamed to
                     what it is. Not one deal's answer may change.
  three_way_matched  did what was billed actually arrive? NULL when the deal
                     has no goods receipt -- not assessed, which is the honest
                     answer and not `false`. Missing paperwork must not
                     manufacture a failure rate.

Live-only.
"""
from __future__ import annotations

import os

import pytest

from src.services.db import get_conn

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live db")


def test_value_reconciled_reproduces_todays_three_way_match_exactly():
    """The rename must not change a single deal's answer."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT count(*) FROM proc.bp_deal_overview
                        WHERE value_reconciled IS DISTINCT FROM three_way_match""")
        assert cur.fetchone()[0] == 0


def test_three_way_matched_is_null_where_no_receipt_exists():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT count(*) FROM proc.bp_deal_overview d
                        WHERE d.three_way_matched IS NOT NULL
                          AND NOT EXISTS (SELECT 1 FROM proc.bp_goods_receipt_trgt g
                                           WHERE g.deal_id = d.deal_id)""")
        assert cur.fetchone()[0] == 0, "a deal with no receipt was given a verdict"


def test_the_whole_corpus_is_not_assessed_because_it_holds_no_receipts():
    """Today every deal must read NULL, because there are zero goods receipts.
    Stated as a test so that the day one arrives, this goes red and somebody
    reads the next test instead of assuming the column is dead."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_goods_receipt_trgt")
        receipts = cur.fetchone()[0]
        cur.execute("""SELECT count(*) FROM proc.bp_deal_overview
                        WHERE three_way_matched IS NOT NULL""")
        verdicts = cur.fetchone()[0]
        if receipts == 0:
            assert verdicts == 0
        else:
            assert verdicts > 0, (
                f"{receipts} receipt rows exist but no deal has a verdict -- "
                f"the deal_id join is broken")


def test_the_verdict_moves_null_to_true_to_false_on_one_real_deal():
    """NULL with no receipt, true once one arrives with no open gap, false once
    a gap is raised. The deal needs a PO and an invoice in _trgt, because
    bp_deal_documents -- what the overview aggregates -- is built from the
    quote, PO and invoice _trgt tables. A goods receipt is deliberately NOT one
    of them: it carries no amount and would distort every total. Its presence
    reaches the view through the `receipted` CTE instead.

    An earlier version of this test guarded every assertion with
    `if row is not None` and therefore asserted NOTHING, because a receipt
    alone never puts a deal in the overview. That is the shape of guard this
    project keeps shipping green; it is written out this time.
    """
    from uuid import uuid4

    tag = f"{uuid4().int % 100000000:08d}"
    deal = f"DEALTEST-{tag}"
    po, inv, grn = f"48{tag}", f"INVOV-{tag}", f"GRN-OV-{tag}"

    def verdict(cur):
        cur.execute("SELECT three_way_matched FROM proc.bp_deal_overview "
                    "WHERE deal_id = %s", (deal,))
        row = cur.fetchone()
        assert row is not None, f"{deal} is not in bp_deal_overview at all"
        return row[0]

    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_purchase_order_trgt "
                        "(po_id, deal_id, deal_name, total_amount, order_date) "
                        "VALUES (%s, %s, 'overview probe', 1000, DATE '2026-01-01')",
                        (po, deal))
            cur.execute("INSERT INTO proc.bp_invoice_trgt "
                        "(invoice_id, deal_id, deal_name, invoice_amount, invoice_date) "
                        "VALUES (%s, %s, 'overview probe', 1000, DATE '2026-02-01')",
                        (inv, deal))

            assert verdict(cur) is None, "no receipt yet -- must be NOT ASSESSED"

            cur.execute("INSERT INTO proc.bp_goods_receipt_trgt (grn_id, po_id, deal_id) "
                        "VALUES (%s, %s, %s)", (grn, po, deal))
            assert verdict(cur) is True, "a receipt with no open gap must pass"

            cur.execute(
                "INSERT INTO proc.bp_extraction_discrepancy "
                "(doc_type, source_file, doc_pk_candidate, field_name, issue_type, "
                " severity, status, blocks_promotion) "
                "VALUES ('goods_receipt', %s, %s, 'po_line[1]', 'billed_not_received', "
                "        'critical', 'open', FALSE)",
                (f"s3://test/{grn}.pdf", grn))
            assert verdict(cur) is False, "an open billed_not_received must read false"

            # And from the invoice side, which joins through a different table.
            cur.execute("UPDATE proc.bp_extraction_discrepancy SET status = 'resolved' "
                        "WHERE doc_pk_candidate = %s", (grn,))
            assert verdict(cur) is True, "a resolved gap must stop counting"
            cur.execute(
                "INSERT INTO proc.bp_extraction_discrepancy "
                "(doc_type, source_file, doc_pk_candidate, field_name, issue_type, "
                " severity, status, blocks_promotion) "
                "VALUES ('invoice', %s, %s, 'po_line[1]', 'nothing_received', "
                "        'critical', 'open', FALSE)",
                (f"s3://test/{inv}.pdf", inv))
            assert verdict(cur) is False, "an invoice-side gap must read false too"
    finally:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_extraction_discrepancy "
                        "WHERE doc_pk_candidate IN (%s, %s)", (grn, inv))
            cur.execute("DELETE FROM proc.bp_goods_receipt_trgt WHERE grn_id = %s", (grn,))
            cur.execute("DELETE FROM proc.bp_invoice_trgt WHERE invoice_id = %s", (inv,))
            cur.execute("DELETE FROM proc.bp_purchase_order_trgt WHERE po_id = %s", (po,))
