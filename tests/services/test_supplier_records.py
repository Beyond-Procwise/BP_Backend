"""supplier_items, open_orders, invoice_status: one supplier's own records, with what is absent said plainly."""
import datetime as dt
from decimal import Decimal

from src.services import supplier_records as SR
from src.services.contract_tools import store

TODAY = dt.date(2026, 10, 22)
SUPPLIERS = [("S1", "Kestrel Supplies 8")]
D = dt.date(2026, 3, 1)
# src, item_key, item_id, description, uom, lines, qty, price_min, price_max, currency, last_date
ITEMS = [("po", "ITM1", "ITM1", "Pallets", "each", 3, Decimal("300"), Decimal("9.5"), Decimal("10"), "GBP", D),
         ("invoice", "ITM1", "ITM1", "Pallets", "each", 2, Decimal("200"), Decimal("10"), Decimal("10"), "GBP", D),
         ("po", "Straps", None, "Straps", None, 1, Decimal("50"), Decimal("2"), Decimal("2"), "GBP", D)]
# po_id, order_date, expected, status, currency, total, contract_id, buyer_id, invoiced_same_ccy, invoices, other_ccy, receipts
ORDERS = [("PO1", D, None, None, "GBP", Decimal("1000"), "C1", "B7", Decimal("600"), 2, 0, 1),
          ("PO2", D, dt.date(2026, 4, 1), "closed", "GBP", Decimal("500"), None, None, Decimal("500"), 1, 1, 0),
          ("PO3", D, None, None, "EUR", None, None, "B7", Decimal("0"), 0, 0, 0)]
# invoice_id, date, due, paid, status, ccy, amount, incl_tax, po_id, terms, open_findings
INVOICES = [("INV1", D, dt.date(2026, 4, 1), None, None, "GBP", Decimal("600"), Decimal("720"), "PO1", "Net 30", 1),
            ("INV2", D, None, None, None, "GBP", Decimal("500"), None, None, None, 0),
            ("INV3", D, dt.date(2026, 12, 1), None, "approved", "GBP", Decimal("50"), None, "PO2", "Net 30", 0),
            ("INV4", D, dt.date(2026, 4, 1), dt.date(2026, 4, 2), "paid", "EUR", Decimal("70"), None, "PO2", None, 0)]


def _rows(items=ITEMS, orders=ORDERS, invoices=INVOICES, suppliers=SUPPLIERS, outage=False):
    seen = []

    def rows(sql, params):
        seen.append((sql, list(params)))
        if "bp_supplier s" in sql:
            return list(suppliers)
        if outage:
            raise store.StoreUnavailable("down")
        if "UNION ALL" in sql:
            return list(items)
        if "invoiced_same_currency" in sql:
            return list(orders)
        if "open_findings" in sql:
            inv = params[1]
            return [r for r in invoices if inv is None or r[0] == inv]
        raise AssertionError(sql)
    rows.seen = seen
    return rows


def test_items_lists_each_item_per_source_with_counts_and_price_range_and_points_to_discovery():
    out = SR.items("kestrel", rows=_rows(), today=TODAY)
    assert out["found"] and out["written"] is False and out["distinct_items"] == 2
    assert out["summary"] == "Kestrel Supplies 8: 2 distinct item(s) on 6 order and invoice line(s) in the period."
    first = out["items"][0]
    assert first == {"source": "po", "item_id": "ITM1", "description": "Pallets", "unit_of_measure": "each", "lines": 3,
                     "quantity": 300.0, "unit_price_min": 9.5, "unit_price_max": 10.0, "currency": "GBP", "last_date": "2026-03-01"}
    assert out["items"][2]["item_id"] is None and out["items"][2]["description"] == "Straps"
    assert "supplier_discovery" in out["note"] and "not normalised" in out["note"]


def test_open_orders_balance_is_total_minus_same_currency_invoices_and_absent_fields_are_said():
    out = SR.open_orders("S1", rows=_rows(), today=TODAY)
    by = {o["po_id"]: o for o in out["orders"]}
    assert by["PO1"]["balance"] == 400.0 and by["PO1"]["open_by_balance"] is True and by["PO1"]["receipts"] == 1
    assert by["PO1"]["status"] == "not recorded" and by["PO1"]["expected_delivery_date"] == "not recorded"
    assert by["PO1"]["buyer_id"] == "B7" and by["PO2"]["buyer_id"] == "not recorded"
    assert by["PO2"]["balance"] == 0.0 and by["PO2"]["open_by_balance"] is False and by["PO2"]["invoices_in_another_currency"] == 1
    assert by["PO3"]["order_total"] is None and by["PO3"]["balance"] is None and by["PO3"]["open_by_balance"] is None
    assert out["totals_by_currency"]["GBP"] == {"order_total": 1500.0, "invoiced": 1100.0, "balance": 400.0, "orders": 2}
    assert out["open_by_balance"] == 1 and out["status_not_recorded"] == 2
    assert out["summary"] == "Kestrel Supplies 8: 3 order(s) in the period, 1 with an invoiced balance still open."
    assert "tax basis" in out["note"] and "not closed or on time" in out["note"]


def test_invoices_report_status_due_and_paid_as_stored_and_overdue_only_from_a_stored_due_date():
    out = SR.invoices("S1", rows=_rows(), today=TODAY)
    by = {i["invoice_id"]: i for i in out["invoices"]}
    assert by["INV1"]["status"] == "not recorded" and by["INV1"]["overdue"] is True and by["INV1"]["days_past_due"] == 204
    assert by["INV1"]["open_findings"] == 1 and by["INV1"]["total_incl_tax"] == 720.0
    assert by["INV2"]["due_date"] == "not recorded" and by["INV2"]["overdue"] is None and by["INV2"]["po_id"] == "not recorded"
    assert by["INV3"]["overdue"] is False and by["INV3"]["days_past_due"] == 0
    assert by["INV4"]["overdue"] is None and by["INV4"]["paid_date"] == "2026-04-02"
    assert out["counts"] == {"status_recorded": 2, "status_not_recorded": 2, "paid_date_recorded": 1,
                             "due_date_recorded": 3, "overdue": 1, "with_open_findings": 1, "without_po": 1}
    assert out["without_po_by_currency"] == {"GBP": {"amount": 500.0, "invoices": 1}} and "1 with no order recorded" in out["summary"]
    assert out["totals_by_currency"] == {"GBP": {"amount": 1150.0, "invoices": 3}, "EUR": {"amount": 70.0, "invoices": 1}}
    assert "never computed from payment terms" in out["note"] and "not 'unpaid' and not 'paid'" in out["note"]


def test_one_invoice_can_be_asked_for_and_the_filter_reaches_the_query():
    rows = _rows()
    out = SR.invoices("S1", invoice_id="INV3", rows=rows, today=TODAY)
    assert [i["invoice_id"] for i in out["invoices"]] == ["INV3"]
    assert rows.seen[-1][1][1] == "INV3"


def test_an_unknown_supplier_an_outage_and_a_bad_period_are_replies_not_results():
    assert SR.items("nobody", rows=_rows(suppliers=[]), today=TODAY)["found"] is False
    out = SR.open_orders("S1", rows=_rows(outage=True), today=TODAY)
    assert out["found"] is False and out["available"] is False
    assert "date like" in SR.invoices("S1", start="last year", rows=_rows(), today=TODAY)["note"]
    assert SR.items("S1", start="2026-05-01", end="2026-01-01", rows=_rows(), today=TODAY)["note"] == "start is after end."


def test_the_default_period_is_three_years_back_and_a_given_period_is_passed_through():
    rows = _rows()
    SR.items("S1", rows=rows, today=TODAY)
    assert rows.seen[-1][1][1] == TODAY - dt.timedelta(days=1095) and rows.seen[-1][1][2] == TODAY
    SR.items("S1", start="2026-01-01", end="2026-06-30", rows=rows, today=TODAY)
    assert rows.seen[-1][1][1:3] == [dt.date(2026, 1, 1), dt.date(2026, 6, 30)]


def test_no_rows_is_an_honest_empty_result_not_an_error():
    out = SR.open_orders("S1", rows=_rows(orders=[]), today=TODAY)
    assert out["found"] and out["orders"] == [] and out["summary"].startswith("Kestrel Supplies 8: no purchase orders recorded in the period")
    assert "not that none were placed" in out["note"]


# invoice_a, date_a, po_a, invoice_b, date_b, po_b, currency, amount
PAIRS = [("INV1", dt.date(2026, 3, 1), "PO1", "INV7", dt.date(2026, 3, 9), "PO1", "GBP", Decimal("600")),
         ("INV2", dt.date(2026, 3, 1), None, "INV9", dt.date(2026, 3, 12), "PO4", "GBP", Decimal("500"))]
PARENT = [("S1", "Kestrel Supplies 8", "P1", "Kestrel Group", "GB")]
CHILDREN = [("S2", "Kestrel Supplies 9", "IE"), ("S3", "Kestrel Supplies 10", None)]


def _rows2(pairs=PAIRS, parent=PARENT, children=CHILDREN, suppliers=SUPPLIERS):
    seen = []

    def rows(sql, params):
        seen.append((sql, list(params)))
        if "bp_supplier s" in sql and "parent_company_id" not in sql:
            return list(suppliers)
        if "b.invoice_id > a.invoice_id" in sql:
            return list(pairs)
        if "LEFT JOIN proc.bp_supplier p" in sql:
            return list(parent)
        if "c.parent_company_id = %s" in sql:
            return list(children)
        raise AssertionError(sql)
    rows.seen = seen
    return rows


def test_duplicate_invoices_lists_same_amount_pairs_within_the_window_as_suspicions():
    rows = _rows2()
    out = SR.duplicate_invoices("S1", window_days=10, rows=rows, today=TODAY)
    assert out["found"] and out["window_days"] == 10 and rows.seen[-1][1][:2] == [10, 10]
    a, b = out["pairs"]
    assert a == {"invoice_a": "INV1", "date_a": "2026-03-01", "po_a": "PO1", "invoice_b": "INV7", "date_b": "2026-03-09", "po_b": "PO1",
                 "currency": "GBP", "amount": 600.0, "days_apart": 8, "same_po": True}
    assert b["po_a"] == "not recorded" and b["same_po"] is False
    assert "2 pair(s)" in out["summary"] and "not a duplicate" in out["summary"] and "finance confirms" in out["note"]


def test_duplicate_window_defaults_and_a_bad_window_falls_back():
    rows = _rows2(pairs=[])
    out = SR.duplicate_invoices("S1", window_days="soon", rows=rows, today=TODAY)
    assert out["window_days"] == SR.DEFAULT_DUPLICATE_WINDOW_DAYS and out["pairs"] == [] and "no two invoices" in out["summary"]


def test_supplier_parents_reads_one_level_and_says_deeper_tiers_are_not_held():
    out = SR.supplier_parents("kestrel", rows=_rows2())
    assert out["found"] and out["parent"] == {"supplier_id": "P1", "supplier": "Kestrel Group"}
    assert [c["supplier"] for c in out["subsidiaries"]] == ["Kestrel Supplies 9", "Kestrel Supplies 10"] and out["subsidiaries"][1]["country"] == "not recorded"
    assert out["summary"] == "Kestrel Supplies 8: parent on record is Kestrel Group; 2 subsidiaries on record."
    assert "not held" in out["note"] and "not independent" in out["note"]
    no_parent = SR.supplier_parents("S1", rows=_rows2(parent=[("S1", "Kestrel Supplies 8", None, None, "GB")], children=[]))
    assert no_parent["parent"] is None and "no parent on record; 0 subsidiaries" in no_parent["summary"]
