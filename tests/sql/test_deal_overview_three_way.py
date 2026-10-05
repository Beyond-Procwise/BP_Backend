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
import pathlib
import re

import pytest

from src.services.db import get_conn

pytestmark = pytest.mark.skipif(
    os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live db")


#: The expression three_way_match carried, lifted verbatim from the migration
#: that defined it. Whitespace-normalised so a reflow is not a failure.
_OLD_MIGRATION = pathlib.Path("deploy/sql/2026-07-29_bp_deal_overview_value_reconciliation.sql")
_NEW_MIGRATION = pathlib.Path("deploy/sql/2026-10-04_deal_overview_drop_three_way_match.sql")


def _expression_for(path: pathlib.Path, column: str) -> str:
    """The boolean expression a migration binds to `column`, whitespace-normalised."""
    text = path.read_text()
    end = text.index(f") AS {column}")
    start = text.rindex("(quote_count > 0", 0, end)
    return re.sub(r"\s+", " ", text[start:end + 1]).strip()


def test_value_reconciled_computes_exactly_what_three_way_match_computed():
    """The rename must not change a single deal's answer.

    While both columns existed this compared them row by row across all 5,042
    deals and found zero differences -- that run is the evidence, recorded in
    the commit. `three_way_match` is now dropped, so the comparison is no
    longer expressible in SQL; what remains checkable forever is that the two
    migrations bind the SAME expression, which is what made the row-by-row run
    come out empty. A tolerance or a predicate edited into one and not the
    other turns this red.
    """
    assert _expression_for(_OLD_MIGRATION, "three_way_match") == \
        _expression_for(_NEW_MIGRATION, "value_reconciled")


def test_the_old_column_is_really_gone():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT 1 FROM information_schema.columns
                        WHERE table_schema='proc' AND table_name='bp_deal_overview'
                          AND column_name='three_way_match'""")
        assert cur.fetchone() is None


def test_three_way_matched_is_null_where_no_receipt_exists():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT count(*) FROM proc.bp_deal_overview d
                        WHERE d.three_way_matched IS NOT NULL
                          AND NOT EXISTS (SELECT 1 FROM proc.bp_goods_receipt_trgt g
                                           WHERE g.deal_id = d.deal_id)""")
        assert cur.fetchone()[0] == 0, "a deal with no receipt was given a verdict"


def test_a_verdict_appears_exactly_when_a_readable_receipt_does():
    """Counted the same way the view counts: a receipt with LINES, on a deal.

    A receipt whose lines were never read is not evidence of delivery -- see
    test_a_receipt_with_no_lines_gives_the_deal_no_verdict. So "receipts exist"
    is not the right precondition for "some deal has a verdict"; "a receipt
    with lines, on a deal that is in the overview" is.

    The corpus held zero receipt-like documents when this was written, so the
    usual answer is 0 and 0. The day a real one lands, this says whether the
    deal_id join actually reached a deal.
    """
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""
            SELECT count(*) FROM proc.bp_goods_receipt_trgt g
             WHERE g.deal_id IS NOT NULL AND g.deal_id <> ''
               AND EXISTS (SELECT 1 FROM proc.bp_goods_receipt_line_items_trgt l
                            WHERE l.grn_id = g.grn_id)
               AND EXISTS (SELECT 1 FROM proc.bp_deal_overview d
                            WHERE d.deal_id = g.deal_id)""")
        readable = cur.fetchone()[0]
        cur.execute("""SELECT count(*) FROM proc.bp_deal_overview
                        WHERE three_way_matched IS NOT NULL""")
        verdicts = cur.fetchone()[0]
        if readable == 0:
            assert verdicts == 0
        else:
            assert verdicts > 0, (
                f"{readable} readable receipts are on deals in the overview but "
                f"no deal has a verdict -- the deal_id join is broken")


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

            # A LINE and a RECORDED OUTCOME, both, because that is what the
            # pipeline leaves: promote() writes the row, the linking copies it
            # to _trgt, and run_match_for_receipt stamps how many lines it could
            # actually compare. A receipt with lines but nothing comparable is
            # not evidence of delivery -- see
            # test_a_receipt_whose_every_line_is_refused... in
            # tests/extraction/test_three_way_match_live.py.
            cur.execute("INSERT INTO proc.bp_goods_receipt_trgt "
                        "(grn_id, po_id, deal_id, lines_assessed, lines_unverifiable) "
                        "VALUES (%s, %s, %s, 1, 0)", (grn, po, deal))
            cur.execute(
                "INSERT INTO proc.bp_goods_receipt_line_items_trgt "
                "(goods_receipt_line_id, grn_id, line_no, item_description, "
                " quantity_received, unit_of_measure, po_id, deal_id) "
                "VALUES (%s, %s, 1, 'Widget A', 10, 'each', %s, %s)",
                (f"{grn}-L1", grn, po, deal))
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
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_trgt WHERE grn_id = %s", (grn,))
            cur.execute("DELETE FROM proc.bp_invoice_trgt WHERE invoice_id = %s", (inv,))
            cur.execute("DELETE FROM proc.bp_purchase_order_trgt WHERE po_id = %s", (po,))


def test_a_receipt_with_no_lines_gives_the_deal_no_verdict():
    """A receipt row whose lines were never read must not read as "goods
    received".

    Found on the live run of 2026-10-05: the first delivery note extracted its
    header, linked to its PO and reached _trgt with ZERO lines (the table
    extractor's field-name defect). The match skips a receipt with no lines --
    correctly, there is nothing to compare -- so it raised no finding, and a
    `receipted` CTE that counted the HEADER turned that silence into
    `three_way_matched = true`. A control that passes because it looked at
    nothing is the one failure this feature must not have.
    """
    from uuid import uuid4

    tag = f"{uuid4().int % 100000000:08d}"
    deal = f"DEALTEST-{tag}"
    po, inv, grn = f"49{tag}", f"INVNL-{tag}", f"GRN-NOLINES-{tag}"

    def verdict(cur):
        cur.execute("SELECT three_way_matched FROM proc.bp_deal_overview "
                    "WHERE deal_id = %s", (deal,))
        row = cur.fetchone()
        assert row is not None, f"{deal} is not in bp_deal_overview"
        return row[0]

    try:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("INSERT INTO proc.bp_purchase_order_trgt "
                        "(po_id, deal_id, deal_name, total_amount, order_date) "
                        "VALUES (%s, %s, 'no-lines probe', 1000, DATE '2026-01-01')",
                        (po, deal))
            cur.execute("INSERT INTO proc.bp_invoice_trgt "
                        "(invoice_id, deal_id, deal_name, invoice_amount, invoice_date) "
                        "VALUES (%s, %s, 'no-lines probe', 1000, DATE '2026-02-01')",
                        (inv, deal))
            cur.execute("INSERT INTO proc.bp_goods_receipt_trgt "
                        "(grn_id, po_id, deal_id, lines_assessed) "
                        "VALUES (%s, %s, %s, 1)", (grn, po, deal))

            assert verdict(cur) is None, (
                "a receipt with no lines must leave the deal NOT ASSESSED")

            cur.execute(
                "INSERT INTO proc.bp_goods_receipt_line_items_trgt "
                "(goods_receipt_line_id, grn_id, line_no, item_description, "
                " quantity_received, unit_of_measure, po_id, deal_id) "
                "VALUES (%s, %s, 1, 'Widget A', 10, 'each', %s, %s)",
                (f"{grn}-L1", grn, po, deal))
            assert verdict(cur) is True, "one line is enough to have looked"
    finally:
        with get_conn() as c, c.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_trgt WHERE grn_id=%s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_trgt WHERE grn_id = %s", (grn,))
            cur.execute("DELETE FROM proc.bp_invoice_trgt WHERE invoice_id = %s", (inv,))
            cur.execute("DELETE FROM proc.bp_purchase_order_trgt WHERE po_id = %s", (po,))
