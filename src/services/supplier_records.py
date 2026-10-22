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
SELECT h.po_id, h.order_date, h.expected_delivery_date, h.po_status, h.currency, h.total_amount, h.contract_id, h.buyer_id,
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
    for po_id, odate, expected, status, ccy, total, contract_id, buyer_id, invoiced, n_inv, n_other, n_rec in raw:
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
            "receipts": int(n_rec or 0), "contract_id": contract_id, "buyer_id": buyer_id or NOT_RECORDED,
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
              "overdue": 0, "with_open_findings": 0, "without_po": 0}
    without_po: Dict[str, Dict[str, float]] = {}
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
        if po_id in (None, ""):
            counts["without_po"] += 1
            w = without_po.setdefault(ccy or "unknown", {"amount": 0.0, "invoices": 0})
            w["invoices"] += 1
            if amount is not None:
                w["amount"] = float(round(Decimal(str(w["amount"])) + Decimal(str(amount)), 2))
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
               f"with open findings; {counts['without_po']} with no order recorded." if out else f"{sname}: no invoices in the period.")
    return {"found": True, "supplier_id": sid, "supplier": sname, "period": _period(period), "summary": summary,
            "invoices": out, "totals_by_currency": totals, "without_po_by_currency": without_po, "counts": counts, "written": False,
            "note": f"Status, due and paid dates are as stored; '{NOT_RECORDED}' means the record holds none, which "
                    "is not 'unpaid' and not 'paid'. Overdue is judged only from a stored due date with no paid "
                    "date, never computed from payment terms. An invoice's hold reason, if any, is in its open "
                    "findings (invoice_reconciliation shows them in full)."}


_DUPLICATES_SQL = """
SELECT a.invoice_id, a.invoice_date, a.po_id, b.invoice_id, b.invoice_date, b.po_id, a.currency, a.invoice_amount
  FROM proc.bp_invoice_trgt a
  JOIN proc.bp_invoice_trgt b
    ON b.supplier_id = a.supplier_id AND b.invoice_id > a.invoice_id
   AND b.currency IS NOT DISTINCT FROM a.currency AND b.invoice_amount = a.invoice_amount
   AND b.invoice_date BETWEEN a.invoice_date - %s AND a.invoice_date + %s
 WHERE a.supplier_id = %s AND a.invoice_date BETWEEN %s AND %s AND a.invoice_amount IS NOT NULL AND a.invoice_amount <> 0
 ORDER BY a.invoice_date DESC, a.invoice_id
 LIMIT %s
"""

DEFAULT_DUPLICATE_WINDOW_DAYS = 14


def duplicate_invoices(supplier: Any = None, start: Any = None, end: Any = None, window_days: Any = None, *,
                       rows: Optional[Rows] = None, today: Optional[date] = None) -> Dict[str, Any]:
    """Pairs of one supplier's invoices with the same amount and currency within a window of days: suspected
    duplicates for a person to check, never a verdict (library row 045)."""
    frame, reply = _frame(supplier, start, end, rows, today)
    if reply is not None:
        return reply
    read, (sid, sname), period, _ = frame
    try:
        window = max(0, int(window_days)) if window_days not in (None, "") else DEFAULT_DUPLICATE_WINDOW_DAYS
    except (TypeError, ValueError):
        window = DEFAULT_DUPLICATE_WINDOW_DAYS
    try:
        raw = read(_DUPLICATES_SQL, [window, window, sid, period.start, period.end, MAX_ROWS])
    except store.StoreUnavailable:
        return dict(OUTAGE)
    pairs = []
    for a_id, a_date, a_po, b_id, b_date, b_po, ccy, amount in raw:
        days = abs((b_date - a_date).days) if a_date and b_date else None
        pairs.append({"invoice_a": a_id, "date_a": _plain(a_date), "po_a": a_po or NOT_RECORDED,
                      "invoice_b": b_id, "date_b": _plain(b_date), "po_b": b_po or NOT_RECORDED,
                      "currency": ccy, "amount": _money(amount), "days_apart": days,
                      "same_po": bool(a_po and b_po and a_po == b_po)})
    summary = (f"{sname}: {len(pairs)} pair(s) of invoices with the same amount within {window} days in the period; "
               "each is a suspicion for a person to check, not a duplicate." if pairs
               else f"{sname}: no two invoices share an amount within {window} days in the period.")
    return {"found": True, "supplier_id": sid, "supplier": sname, "period": _period(period), "window_days": window,
            "summary": summary, "pairs": pairs, "written": False,
            "note": "A pair is the same supplier, currency and amount within the window. Two genuine invoices can match "
                    "(a repeat order at a fixed price); a pair on the same order is the stronger signal. Nothing is "
                    "blocked or marked here; finance confirms with the delivery notes."}


_PARENT_SQL = """
SELECT s.supplier_id, s.supplier_name, s.parent_company_id, p.supplier_name, s.registered_country
  FROM proc.bp_supplier s
  LEFT JOIN proc.bp_supplier p ON p.supplier_id = s.parent_company_id
 WHERE s.supplier_id = %s
"""
_CHILDREN_SQL = """
SELECT c.supplier_id, c.supplier_name, c.registered_country
  FROM proc.bp_supplier c
 WHERE c.parent_company_id = %s
 ORDER BY c.supplier_name
 LIMIT %s
"""


def supplier_parents(supplier: Any = None, *, rows: Optional[Rows] = None) -> Dict[str, Any]:
    """The parent and the subsidiaries recorded on the supplier master for one supplier (library row 056). The
    master holds one level: a parent id per record. Anything beyond that is not held and is said so."""
    read = rows or store.rows
    resolved, reply = supplier_resolve.resolve(supplier, rows=read)
    if reply is not None:
        return reply
    sid, sname = resolved
    try:
        head = read(_PARENT_SQL, [sid])
        kids = read(_CHILDREN_SQL, [sid, MAX_ROWS])
    except store.StoreUnavailable:
        return {"found": False, "available": False, "note": "The supplier master could not be read just now."}
    if not head:
        return {"found": False, "note": f"{sname} has no master record to read a parent from."}
    _, _, parent_id, parent_name, country = head[0]
    parent = {"supplier_id": parent_id, "supplier": parent_name or NOT_RECORDED} if parent_id else None
    children = [{"supplier_id": c_id, "supplier": c_name, "country": c_country or NOT_RECORDED} for c_id, c_name, c_country in kids]
    summary = (f"{sname}: " + (f"parent on record is {parent['supplier']}" if parent else "no parent on record")
               + f"; {len(children)} subsidiar{'y' if len(children) == 1 else 'ies'} on record.")
    return {"found": True, "supplier_id": sid, "supplier": sname, "country": country or NOT_RECORDED, "parent": parent,
            "subsidiaries": children, "summary": summary, "written": False,
            "note": "One level only, as the supplier master records it: a parent id per supplier. Sub-tier suppliers "
                    "(who supplies our supplier) and deeper ownership are not held; ask the supplier or a provider. "
                    "Two bidders sharing a parent here are not independent for a minimum-bidder rule."}
