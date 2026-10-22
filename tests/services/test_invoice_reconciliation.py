import datetime as dt
from decimal import Decimal

from src.services import invoice_reconciliation as IR
from src.services.contract_tools import store

TODAY = dt.date(2026, 10, 22)
SUPPLIERS = [("S1", "Clearwater Logistics 14")]
D = dt.date(2026, 3, 1)
# invoice_id, invoice_date, po_id, item_id, description, uom, ordered_qty, po_unit_price, po_currency, n,
# invoiced_qty, invoiced_unit_price, currency, line_amount
LINES = [
    ("INV1", D, "PO1", "ITM1", "Pallets", "each", Decimal("100"), Decimal("10"), "GBP", 1, Decimal("120"), Decimal("10"), "GBP", Decimal("1200")),
    ("INV1", D, "PO1", "ITM2", "Straps", "each", Decimal("50"), Decimal("2"), "GBP", 1, Decimal("50"), Decimal("2.5"), "GBP", Decimal("125")),
    ("INV2", D, "PO2", "ITM9", "Install", None, None, None, None, None, Decimal("1"), Decimal("300"), "GBP", Decimal("300")),
    ("INV2", D, "PO2", "ITM3", "Wrap", "roll", Decimal("10"), Decimal("5"), "EUR", 1, Decimal("10"), Decimal("5"), "GBP", Decimal("50")),
    ("INV3", D, "PO3", "ITM4", "Tape", "roll", Decimal("40"), Decimal("1"), "GBP", 2, Decimal("40"), Decimal("1"), "GBP", Decimal("40")),
]
RECEIPTS = [("PO1", 2, dt.date(2026, 2, 20), dt.date(2026, 3, 2))]
FINDINGS = [(9, "quantity", "line", "high", "invoice", "INV1", "quantity", "120", "100", "20", "qty above PO", "open", dt.datetime(2026, 3, 3, 9))]


def _rows(lines=LINES, receipts=RECEIPTS, findings=FINDINGS, suppliers=SUPPLIERS):
    seen = []

    def rows(sql, params):
        seen.append((sql, list(params)))
        if "bp_supplier s" in sql:
            return list(suppliers)
        if "WITH pol" in sql:
            inv, po = params[6], params[8]
            return [r for r in lines if (inv is None or r[0] == inv) and (po is None or r[2] == po)]
        if "bp_goods_receipt_trgt" in sql:
            return [r for r in receipts if r[0] in params[0]]
        if "bp_detection_finding" in sql:
            return list(findings)
        raise AssertionError(sql)
    rows.seen = seen
    return rows


def test_lines_are_matched_and_differences_are_by_kind_and_currency():
    out = IR.reconcile("clearwater", rows=_rows(), today=TODAY)
    assert out["found"] and out["written"] is False and out["invoice_lines"] == 5
    by = {(l["invoice_id"], l["item_id"]): l for l in out["lines"]}
    qty = by[("INV1", "ITM1")]
    assert qty["matched"] and qty["kind"] == "quantity_above_ordered" and qty["quantity_difference_at_po_price"] == 200.0 and qty["price_difference"] == 0.0
    price = by[("INV1", "ITM2")]
    assert price["kind"] == "price_above_po" and price["price_difference"] == 25.0 and price["quantity_difference_at_po_price"] == 0.0
    assert by[("INV2", "ITM9")]["kind"] == "no_po_line" and by[("INV2", "ITM9")]["matched"] is False
    assert by[("INV2", "ITM3")]["kind"] == "currency_differs" and by[("INV3", "ITM4")]["kind"] == "ambiguous_po_lines"
    assert out["differences_by_currency"] == {"GBP": {"price_above_po": 25.0, "price_below_po": 0.0,
                                                      "quantity_above_ordered_at_po_price": 200.0,
                                                      "quantity_below_ordered_at_po_price": 0.0, "lines": 2}}
    assert out["kinds"] == {"quantity_above_ordered": 1, "price_above_po": 1, "no_po_line": 1, "currency_differs": 1, "ambiguous_po_lines": 1}


def test_receipts_are_per_po_without_quantities_and_a_missing_receipt_is_a_gap_not_non_delivery():
    out = IR.reconcile("S1", rows=_rows(), today=TODAY)
    by = {(l["invoice_id"], l["item_id"]): l for l in out["lines"]}
    assert by[("INV1", "ITM1")]["receipts_on_po"] == 2 and by[("INV1", "ITM1")]["received_qty"] is None
    assert by[("INV2", "ITM9")]["receipts_on_po"] == 0
    assert out["pos_without_receipt"] == ["PO2", "PO3"] and out["receipts_by_po"]["PO1"]["last"] == "2026-03-02"
    assert "not evidence that nothing was delivered" in out["note"]


def test_findings_are_carried_as_recorded_and_are_called_catches_not_verdicts():
    out = IR.reconcile("S1", rows=_rows(), today=TODAY)
    f = out["open_findings"][0]
    assert f["rule"] == "quantity" and f["observed"] == "120" and f["expected"] == "100" and f["delta"] == "20"
    assert out["open_findings_count"] == 1 and "not a verdict" in out["note"]


def test_invoice_and_po_filters_reach_the_queries():
    rows = _rows()
    out = IR.reconcile("S1", invoice_id="INV1", rows=rows, today=TODAY)
    assert out["invoice_lines"] == 2 and out["invoice_id"] == "INV1"
    findings_params = [p for s, p in rows.seen if "bp_detection_finding" in s][0]
    assert findings_params[1] == "INV1"
    out = IR.reconcile("S1", po_id="PO2", rows=_rows(), today=TODAY)
    assert {l["po_id"] for l in out["lines"]} == {"PO2"}


def test_period_defaults_and_bad_dates():
    out = IR.reconcile("S1", rows=_rows(), today=TODAY)
    assert out["period"]["label"] != "custom"
    assert "date like" in IR.reconcile("S1", start="soon", rows=_rows(), today=TODAY)["note"]
    assert IR.reconcile("S1", start="2026-06-01", end="2026-01-01", rows=_rows(), today=TODAY)["note"] == "start is after end."


def test_no_lines_is_said_and_outage_is_not_correctness():
    out = IR.reconcile("S1", rows=_rows(lines=[], receipts=[], findings=[]), today=TODAY)
    assert out["found"] and out["invoice_lines"] == 0 and out["note"].startswith("No invoice line")

    def down(sql, params):
        raise store.StoreUnavailable("db down")
    out = IR.reconcile("S1", rows=down, today=TODAY)
    assert out["found"] is False and out["available"] is False and "correct or incorrect" in out["note"]
    assert IR.reconcile("", rows=_rows())["found"] is False
