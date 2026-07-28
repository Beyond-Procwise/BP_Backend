"""Provenance declarations at promotion: values the pipeline supplied rather than read.

Two things were being written to _stg with no record that they were inferred, so on screen
they were indistinguishable from figures the document actually printed:

  * a money field computed as subtotal + tax (or subtotal x tax_percent) when the page
    never stated it
  * a date whose DAY we supplied because the page gave only a month ("Nov 2024")

The money check previously compared the row before and after _compute_derived. That never
fired once across the whole corpus, because _compute_derived runs TWICE — context_layer
derives and persists before _raw is written, so the "before" snapshot is already derived.
It now asks the document instead, which is the only thing that can answer the question.
"""
import datetime
from decimal import Decimal

from src.services.extraction.promotion import (
    _log_derived_money,
    _log_imprecise_dates,
    _money_on_page,
)


class FakeCur:
    """Records the parameters of every INSERT the function attempts."""

    def __init__(self):
        self.rows = []

    def execute(self, sql, params=None):
        if params:
            self.rows.append(params)


# INSERT parameter order, so the indices below are readable rather than magic:
#  0 doc_type   1 raw_id      2 source_file     3 doc_pk_candidate
#  4 field_name 5 raw_value   6 expected_value  7 computed_value
#  8 issue_type 9 severity   10 status         11 notes  12 blocks_promotion
FIELD, EXPECTED, ISSUE, NOTES = 4, 6, 8, 11


def issues(cur):
    return [(p[FIELD], p[ISSUE]) for p in cur.rows]


# ---------------------------------------------------------------- money on page
def test_money_on_page_tolerates_how_the_page_writes_it():
    page = "Subtotal 1,318.40  Tax 263.68  Total 1582.08"
    assert _money_on_page(Decimal("1318.40"), page)
    assert _money_on_page(Decimal("1582.08"), page)
    assert _money_on_page(Decimal("263.68"), page)
    assert not _money_on_page(Decimal("9999.99"), page)


def test_money_on_page_is_false_without_text():
    # No text means we cannot say the figure is on the page, so nothing is declared
    # rather than everything being declared. Guarded by the caller.
    assert not _money_on_page(Decimal("10.00"), "")


# ---------------------------------------------------------------- derived money
def test_declares_a_net_that_is_total_minus_tax_and_absent_from_the_page():
    # The real shape of invoice INV-792631: the page prints a subtotal, a tax and a
    # total, and the stored net is total - tax, matching none of them.
    page = "Subtotal: 26,580.00\nTax: 2,723.00\nTotal: 30,403.00"
    row = {
        "invoice_id": "INV-792631",
        "invoice_amount": Decimal("27680.00"),      # 30,403.00 - 2,723.00
        "tax_amount": Decimal("2723.00"),
        "invoice_total_incl_tax": Decimal("30403.00"),
    }
    cur = FakeCur()
    assert _log_derived_money(cur, "invoice", 1, page, row) == 1
    assert issues(cur) == [("invoice_amount", "value_derived")]
    assert "does not appear" in cur.rows[0][NOTES]


def test_declares_nothing_when_every_figure_is_printed():
    page = "Subtotal: £1318.40\nTax 20%: £263.68\nTotal: £1,582.08"
    row = {
        "invoice_id": "X",
        "invoice_amount": Decimal("1318.40"),
        "tax_amount": Decimal("263.68"),
        "invoice_total_incl_tax": Decimal("1582.08"),
    }
    cur = FakeCur()
    assert _log_derived_money(cur, "invoice", 1, page, row) == 0


def test_declares_nothing_when_there_is_no_page_text():
    # A scan with no text layer must not have every one of its fields declared
    # inferred — that would be an accusation this check cannot support.
    row = {"invoice_id": "X", "invoice_amount": Decimal("10.00")}
    cur = FakeCur()
    assert _log_derived_money(cur, "invoice", 1, "", row) == 0


def test_derived_money_covers_quotes_and_purchase_orders():
    page = "Grand total 5,000.00"
    row = {"po_id": "P1", "total_amount": Decimal("4166.67"),
           "total_amount_incl_tax": Decimal("5000.00")}
    cur = FakeCur()
    assert _log_derived_money(cur, "purchase_order", 1, page, row) == 1
    assert issues(cur) == [("total_amount", "value_derived")]


# ---------------------------------------------------------------- imprecise dates
def test_declares_a_day_we_supplied_for_a_month_only_document():
    # The real shape of invoice INV600784: the page says "Nov 2024" and nothing more.
    page = "INVOICE\nInvoice Number INV600784\nNov 2024\nPay by : 31 Dec 2024"
    row = {"invoice_id": "INV600784",
           "invoice_date": datetime.date(2024, 11, 1),
           "due_date": datetime.date(2024, 12, 31)}
    cur = FakeCur()
    assert _log_imprecise_dates(cur, "invoice", 1, page, row) == 1
    field, issue = issues(cur)[0]
    assert (field, issue) == ("invoice_date", "date_precision_inferred")
    assert "no day" in cur.rows[0][NOTES]
    # The month IS what the document states, and is recorded as the expected value.
    assert cur.rows[0][EXPECTED] == "November 2024"


def test_leaves_alone_a_first_of_month_the_document_actually_states():
    page = "Invoice date: 1 Nov 2024"
    row = {"invoice_id": "Y", "invoice_date": datetime.date(2024, 11, 1)}
    cur = FakeCur()
    assert _log_imprecise_dates(cur, "invoice", 1, page, row) == 0


def test_leaves_alone_a_real_day():
    page = "Nov 2024"
    row = {"invoice_id": "Z", "invoice_date": datetime.date(2024, 11, 15)}
    cur = FakeCur()
    assert _log_imprecise_dates(cur, "invoice", 1, page, row) == 0


def test_leaves_alone_a_month_the_page_does_not_mention():
    # Day 1 alone is not evidence; the page must actually carry that month-year.
    page = "Nov 2024"
    row = {"invoice_id": "Z", "invoice_date": datetime.date(2023, 4, 1)}
    cur = FakeCur()
    assert _log_imprecise_dates(cur, "invoice", 1, page, row) == 0


def test_imprecise_dates_cover_quotes_and_purchase_orders():
    page = "Purchase Order\nMar 2023"
    cur = FakeCur()
    assert _log_imprecise_dates(
        cur, "purchase_order", 1, page,
        {"po_id": "P1", "order_date": datetime.date(2023, 3, 1)},
    ) == 1
    assert issues(cur) == [("order_date", "date_precision_inferred")]
