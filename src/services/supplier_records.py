"""Three read-only views of one supplier's own records: what we buy from them, their open orders, their invoices.

Built after the 2026-10-22 content review of the skills library: six skills had told the agent to find
"what a supplier sells" or "which invoices are on hold" with tools that return neither (similar_items
gives counts, invoice_reconciliation gives line matches). Each function here answers one of those
questions from the order, invoice and receipt tables as stored, says plainly what the records do not
hold (the corpus carries no order status, no invoice status and almost no due dates), and writes
nothing. Sums are done here, in the open, the way invoice_reconciliation sums its differences; nothing
is estimated.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services import supplier_resolve
from src.services.analytics.period import Period
from src.services.contract_tools import store

log = logging.getLogger(__name__)

MAX_ROWS = 60
DEFAULT_WINDOW_DAYS = 1095
NOT_RECORDED = "not recorded"

Rows = Callable[[str, Sequence[Any]], List[tuple]]

_ITEMS_SQL = """
SELECT src, item_key, item_id, description, uom, lines, qty, price_min, price_max, currency, last_date FROM (
    SELECT 'po' AS src, COALESCE(l.item_id, l.item_description) AS item_key, MIN(l.item_id) AS item_id,
           MIN(l.item_description) AS description, MIN(l.unit_of_measure) AS uom, COUNT(*)::int AS lines,
           SUM(l.quantity) AS qty, MIN(l.unit_price) AS price_min, MAX(l.unit_price) AS price_max,
           MIN(COALESCE(l.currency, h.currency)) AS currency, MAX(h.order_date) AS last_date
      FROM proc.bp_po_line_items_trgt l
      JOIN proc.bp_purchase_order_trgt h USING (po_id)
     WHERE h.supplier_id = %s AND h.order_date BETWEEN %s AND %s
     GROUP BY COALESCE(l.item_id, l.item_description)
    UNION ALL
    SELECT 'invoice', COALESCE(il.item_id, il.item_description), MIN(il.item_id), MIN(il.item_description),
           MIN(il.unit_of_measure), COUNT(*)::int, SUM(il.quantity), MIN(il.unit_price), MAX(il.unit_price),
           MIN(ih.currency), MAX(ih.invoice_date)
      FROM proc.bp_invoice_line_items_trgt il
      JOIN proc.bp_invoice_trgt ih ON ih.invoice_id = il.invoice_id
     WHERE ih.supplier_id = %s AND ih.invoice_date BETWEEN %s AND %s
     GROUP BY COALESCE(il.item_id, il.item_description)
) x
ORDER BY lines DESC, item_key
LIMIT %s
"""

_ORDERS_SQL = """
SELECT h.po_id, h.order_date, h.expected_delivery_date, h.po_status, h.currency, h.total_amount, h.contract_id,
       (SELECT COALESCE(SUM(i.invoice_amount), 0) FROM proc.bp_invoice_trgt i
         WHERE i.po_id = h.po_id AND i.currency = h.currency) AS invoiced_same_currency,
       (SELECT COUNT(*) FROM proc.bp_invoice_trgt i WHERE i.po_id = h.po_id) AS invoices,
       (SELECT COUNT(*) FROM proc.bp_invoice_trgt i WHERE i.po_id = h.po_id AND i.currency <> h.currency) AS invoices_other_currency,
       (SELECT COUNT(*) FROM proc.bp_goods_receipt_trgt g WHERE g.po_id = h.po_id) AS receipts
  FROM proc.bp_purchase_order_trgt h
 WHERE h.supplier_id = %s AND h.order_date BETWEEN %s AND %s
 ORDER BY h.order_date DESC, h.po_id
 LIMIT %s
"""

_INVOICES_SQL = """
SELECT i.invoice_id, i.invoice_date, i.due_date, i.invoice_paid_date, i.invoice_status, i.currency,
       i.invoice_amount, i.invoice_total_incl_tax, i.po_id, i.payment_terms,
       (SELECT COUNT(*) FROM proc.bp_detection_finding f
         WHERE f.status = 'open' AND f.doc_type = 'invoice' AND f.doc_pk = i.invoice_id) AS open_findings
  FROM proc.bp_invoice_trgt i
 WHERE i.supplier_id = %s AND (%s IS NULL OR i.invoice_id = %s) AND i.invoice_date BETWEEN %s AND %s
 ORDER BY i.invoice_date DESC, i.invoice_id
 LIMIT %s
"""

OUTAGE = {"found": False, "available": False,
          "note": "The order and invoice records could not be read just now. This says nothing about what they hold."}


def _plain(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, date):
        return v.isoformat()
    return v


def _money(v: Any) -> Optional[float]:
    return None if v is None else float(round(Decimal(str(v)), 2))


def _day(value: Any, default: date, name: str) -> date:
    if value in (None, ""):
        return default
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{name} must be a date like 2026-01-31.") from None


def _frame(supplier: Any, start: Any, end: Any, rows: Optional[Rows], today: Optional[date]):
    """Resolve the supplier and the period, or return the reply to hand back as is."""
    read = rows or store.rows
    resolved, reply = supplier_resolve.resolve(supplier, rows=read)
    if reply is not None:
        return None, reply
    now = today or date.today()
    default = Period(now - timedelta(days=DEFAULT_WINDOW_DAYS), now, f"last {DEFAULT_WINDOW_DAYS} days")
    try:
        period = Period(_day(start, default.start, "start"), _day(end, default.end, "end"),
                        default.label if start is None and end is None else "custom")
    except ValueError as exc:
        return None, {"found": False, "note": str(exc)}
    if period.start > period.end:
        return None, {"found": False, "note": "start is after end."}
    return (read, resolved, period, now), None


def _period(period: Period) -> Dict[str, str]:
    return {"label": period.label, "start": period.start.isoformat(), "end": period.end.isoformat()}


def items(supplier: Any = None, start: Any = None, end: Any = None, *, rows: Optional[Rows] = None,
          today: Optional[date] = None) -> Dict[str, Any]:
    """What we buy from one supplier: each item on their order and invoice lines with line count, quantity,
    unit-price range as billed and last date, per source. Prices are not normalised across units."""
    frame, reply = _frame(supplier, start, end, rows, today)
    if reply is not None:
        return reply
    read, (sid, sname), period, _ = frame
    try:
        raw = read(_ITEMS_SQL, [sid, period.start, period.end, sid, period.start, period.end, MAX_ROWS])
    except store.StoreUnavailable:
        return dict(OUTAGE)
    out_items = [{"source": src, "item_id": item_id, "description": desc, "unit_of_measure": uom, "lines": int(lines or 0),
                  "quantity": _plain(qty), "unit_price_min": _plain(pmin), "unit_price_max": _plain(pmax),
                  "currency": ccy, "last_date": _plain(last)}
                 for src, _key, item_id, desc, uom, lines, qty, pmin, pmax, ccy, last in raw]
    distinct = len({(r[1]) for r in raw})
    summary = (f"{sname}: {distinct} distinct item(s) on {sum(i['lines'] for i in out_items)} order and invoice line(s) "
               f"in the period." if raw else f"{sname}: no order or invoice lines in the period.")
    return {"found": True, "supplier_id": sid, "supplier": sname, "period": _period(period), "summary": summary,
            "distinct_items": distinct, "items": out_items, "written": False,
            "note": "Lines as stored, per source (po, invoice). Unit prices are as billed and not normalised across "
                    "units of measure. To find other suppliers for one of these items, pass its description to "
                    "supplier_discovery."}


def open_orders(supplier: Any = None, start: Any = None, end: Any = None, *, rows: Optional[Rows] = None,
                today: Optional[date] = None) -> Dict[str, Any]:
    """One supplier's purchase orders with what has been invoiced against each and the balance still open."""
    frame, reply = _frame(supplier, start, end, rows, today)
    if reply is not None:
        return reply
    read, (sid, sname), period, _ = frame
    try:
        raw = read(_ORDERS_SQL, [sid, period.start, period.end, MAX_ROWS])
    except store.StoreUnavailable:
        return dict(OUTAGE)
    orders: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, float]] = {}
    status_missing = 0
    for po_id, odate, expected, status, ccy, total, contract_id, invoiced, n_inv, n_other, n_rec in raw:
        total_d = Decimal(str(total)) if total is not None else None
        invoiced_d = Decimal(str(invoiced or 0))
        balance = None if total_d is None else total_d - invoiced_d
        if status in (None, ""):
            status_missing += 1
        orders.append({
            "po_id": po_id, "order_date": _plain(odate), "expected_delivery_date": _plain(expected) or NOT_RECORDED,
            "status": status or NOT_RECORDED, "currency": ccy, "order_total": _money(total_d),
            "invoiced_same_currency": _money(invoiced_d), "balance": _money(balance),
            "open_by_balance": None if balance is None else balance > 0,
            "invoices": int(n_inv or 0), "invoices_in_another_currency": int(n_other or 0),
            "receipts": int(n_rec or 0), "contract_id": contract_id,
        })
        t = totals.setdefault(ccy or "unknown", {"order_total": 0.0, "invoiced": 0.0, "balance": 0.0, "orders": 0})
        t["orders"] += 1
        if total_d is not None:
            t["order_total"] = float(round(Decimal(str(t["order_total"])) + total_d, 2))
            t["balance"] = float(round(Decimal(str(t["balance"])) + balance, 2))
        t["invoiced"] = float(round(Decimal(str(t["invoiced"])) + invoiced_d, 2))
    open_count = sum(1 for o in orders if o["open_by_balance"])
    summary = (f"{sname}: {len(orders)} order(s) in the period, {open_count} with an invoiced balance still open."
               if orders else f"{sname}: no purchase orders recorded in the period (which says nothing about orders placed outside the records or the contract).")
    return {"found": True, "supplier_id": sid, "supplier": sname, "period": _period(period), "summary": summary,
            "orders": orders, "totals_by_currency": totals, "open_by_balance": open_count,
            "status_not_recorded": status_missing, "written": False,
            "note": "Balance = order total minus invoices on the order in the same currency, both as stored (the "
                    "tax basis of the two may differ: check before quoting a balance as money owed). Invoices in "
                    "another currency are counted, not summed. An order status or expected delivery date that is "
                    f"'{NOT_RECORDED}' is absent from the records, not closed or on time. No orders in the period means none are "
                    "recorded, not that none were placed and not that no agreement exists."}


def invoices(supplier: Any = None, invoice_id: Any = None, start: Any = None, end: Any = None, *,
             rows: Optional[Rows] = None, today: Optional[date] = None) -> Dict[str, Any]:
    """One supplier's invoices with the status, due and paid dates as stored, open findings per invoice, and an
    overdue flag only where a due date exists and no paid date does."""
    frame, reply = _frame(supplier, start, end, rows, today)
    if reply is not None:
        return reply
    read, (sid, sname), period, now = frame
    inv = str(invoice_id).strip() if invoice_id not in (None, "") else None
    try:
        raw = read(_INVOICES_SQL, [sid, inv, inv, period.start, period.end, MAX_ROWS])
    except store.StoreUnavailable:
        return dict(OUTAGE)
    out: List[Dict[str, Any]] = []
    totals: Dict[str, Dict[str, float]] = {}
    counts = {"status_recorded": 0, "status_not_recorded": 0, "paid_date_recorded": 0, "due_date_recorded": 0,
              "overdue": 0, "with_open_findings": 0}
    for inv_id, idate, due, paid, status, ccy, amount, incl_tax, po_id, terms, findings in raw:
        overdue = None
        days_past_due = None
        if due is not None:
            counts["due_date_recorded"] += 1
            if paid is None:
                overdue = due < now
                days_past_due = (now - due).days if overdue else 0
        if status not in (None, ""):
            counts["status_recorded"] += 1
        else:
            counts["status_not_recorded"] += 1
        if paid is not None:
            counts["paid_date_recorded"] += 1
        if overdue:
            counts["overdue"] += 1
        if int(findings or 0):
            counts["with_open_findings"] += 1
        out.append({
            "invoice_id": inv_id, "invoice_date": _plain(idate), "due_date": _plain(due) or NOT_RECORDED,
            "paid_date": _plain(paid) or NOT_RECORDED, "status": status or NOT_RECORDED, "currency": ccy,
            "amount": _money(amount), "total_incl_tax": _money(incl_tax), "po_id": po_id or NOT_RECORDED,
            "payment_terms": terms or NOT_RECORDED, "overdue": overdue, "days_past_due": days_past_due,
            "open_findings": int(findings or 0),
        })
        t = totals.setdefault(ccy or "unknown", {"amount": 0.0, "invoices": 0})
        t["invoices"] += 1
        if amount is not None:
            t["amount"] = float(round(Decimal(str(t["amount"])) + Decimal(str(amount)), 2))
    summary = (f"{sname}: {len(out)} invoice(s) in the period; a payment status is recorded for "
               f"{counts['status_recorded']} of them, a due date for {counts['due_date_recorded']}, a paid date for "
               f"{counts['paid_date_recorded']}; {counts['overdue']} overdue by due date; {counts['with_open_findings']} "
               f"with open findings." if out else f"{sname}: no invoices in the period.")
    return {"found": True, "supplier_id": sid, "supplier": sname, "period": _period(period), "summary": summary,
            "invoices": out, "totals_by_currency": totals, "counts": counts, "written": False,
            "note": f"Status, due and paid dates are as stored; '{NOT_RECORDED}' means the record holds none, which "
                    "is not 'unpaid' and not 'paid'. Overdue is judged only from a stored due date with no paid "
                    "date, never computed from payment terms. An invoice's hold reason, if any, is in its open "
                    "findings (invoice_reconciliation shows them in full)."}
