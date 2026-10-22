"""invoice_reconciliation: what a supplier invoiced against what we ordered and received, line by line. Read-only.

Built because the overbilling skill, given only corpus-wide facts, spent its whole round budget
looking for lines (live, 2026-10-22). One call returns, for a supplier (optionally one invoice or
one PO) in a period: every invoice line matched to a PO line on (po_id, item_id) with ordered and
invoiced quantity and unit price and the differences by kind, the invoice lines with no PO line to
match, the goods receipts recorded per PO (header level: the records carry no received quantities),
and the open triage findings on the supplier's documents with their observed, expected and delta
values as the triage engine recorded them. Differences are summed per currency in this module (the
same arithmetic the triage checks do); nothing is estimated, and a missing receipt is reported as
"no receipt recorded", never as "not delivered".
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services import supplier_resolve
from src.services.analytics.period import Period
from src.services.contract_tools import store

log = logging.getLogger(__name__)

MAX_LINES = 60
DEFAULT_WINDOW_DAYS = 1095  # three years back when no period is given: a dispute is rarely about this quarter
MAX_FINDINGS = 40
KINDS = ("price_above_po", "price_below_po", "quantity_above_ordered", "quantity_below_ordered", "matched")

_MATCHED_SQL = """
WITH pol AS (
    SELECT l.po_id, l.item_id, MIN(l.item_description) AS description, MIN(l.unit_of_measure) AS uom,
           SUM(l.quantity) AS ordered_qty, MIN(l.unit_price) AS po_unit_price, MIN(UPPER(h.currency)) AS po_currency,
           COUNT(*) AS n
      FROM proc.bp_po_line_items_trgt l
      JOIN proc.bp_purchase_order_trgt h USING (po_id)
     WHERE l.item_id IS NOT NULL AND l.unit_price IS NOT NULL
       AND l.po_id IN (SELECT DISTINCT il.po_id FROM proc.bp_invoice_line_items_trgt il
                        JOIN proc.bp_invoice_trgt ih ON ih.invoice_id = il.invoice_id
                       WHERE ih.supplier_id = %s AND ih.invoice_date BETWEEN %s AND %s AND il.po_id IS NOT NULL)
     GROUP BY l.po_id, l.item_id
)
SELECT ih.invoice_id, ih.invoice_date, il.po_id, il.item_id, COALESCE(il.item_description, pol.description),
       pol.uom, pol.ordered_qty, pol.po_unit_price, pol.po_currency, pol.n,
       il.quantity, il.unit_price, UPPER(COALESCE(il.currency, ih.currency)), il.line_amount
  FROM proc.bp_invoice_line_items_trgt il
  JOIN proc.bp_invoice_trgt ih ON ih.invoice_id = il.invoice_id
  LEFT JOIN pol ON pol.po_id = il.po_id AND pol.item_id = il.item_id
 WHERE ih.supplier_id = %s AND ih.invoice_date BETWEEN %s AND %s
   AND (%s IS NULL OR ih.invoice_id = %s) AND (%s IS NULL OR il.po_id = %s)
 ORDER BY ih.invoice_date, ih.invoice_id, il.line_no
"""
_RECEIPTS_SQL = """
    SELECT g.po_id, COUNT(*)::int, MIN(g.receipt_date), MAX(g.receipt_date)
      FROM proc.bp_goods_receipt_trgt g
     WHERE g.po_id = ANY(%s)
     GROUP BY g.po_id
"""
_FINDINGS_SQL = """
    SELECT f.finding_id, f.rule_id, f.category, f.severity, f.doc_type, f.doc_pk, f.field_name,
           f.observed_value, f.expected_value, f.delta, f.notes, f.status, f.detected_at
      FROM proc.bp_detection_finding f
      JOIN proc.bp_deal_overview o ON o.deal_id = f.deal_id
     WHERE f.status = 'open' AND o.supplier_id = %s
       AND (%s IS NULL OR f.doc_pk = %s)
     ORDER BY f.detected_at DESC
     LIMIT %s
"""


def _plain(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _dec(v: Any) -> Optional[Decimal]:
    if v is None or isinstance(v, bool):
        return None
    try:
        return Decimal(str(v))
    except Exception:  # noqa: BLE001
        return None


def _day(value: Any, default: date, name: str) -> date:
    if value is None or value == "":
        return default
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{name} must be a date like 2026-01-31.") from None


def _line(r: tuple) -> Dict[str, Any]:
    (invoice_id, invoice_date, po_id, item_id, description, uom, ordered_qty, po_price, po_ccy, n,
     inv_qty, inv_price, inv_ccy, line_amount) = r
    out: Dict[str, Any] = {"invoice_id": invoice_id, "invoice_date": _plain(invoice_date), "po_id": po_id,
                           "item_id": item_id, "description": description, "unit_of_measure": uom,
                           "invoiced_qty": _plain(inv_qty), "invoiced_unit_price": _plain(inv_price),
                           "currency": inv_ccy, "line_amount": _plain(line_amount)}
    if po_price is None or n is None:
        out.update({"matched": False, "kind": "no_po_line", "ordered_qty": None, "po_unit_price": None,
                    "note": "no PO line with this item on this PO; the agreed basis is not on record"})
        return out
    if int(n) > 1:
        out.update({"matched": False, "kind": "ambiguous_po_lines", "ordered_qty": _plain(ordered_qty),
                    "po_unit_price": _plain(po_price), "note": "several PO lines carry this item; not matched"})
        return out
    if po_ccy and inv_ccy and po_ccy != inv_ccy:
        out.update({"matched": False, "kind": "currency_differs", "ordered_qty": _plain(ordered_qty),
                    "po_unit_price": _plain(po_price), "po_currency": po_ccy,
                    "note": "invoice and PO are in different currencies; not compared"})
        return out
    out.update({"matched": True, "ordered_qty": _plain(ordered_qty), "po_unit_price": _plain(po_price)})
    kinds: List[str] = []
    price_diff = qty_diff = Decimal(0)
    ip, pp, iq, oq = _dec(inv_price), _dec(po_price), _dec(inv_qty), _dec(ordered_qty)
    if ip is not None and pp is not None and iq is not None:
        price_diff = (ip - pp) * iq
        if ip > pp:
            kinds.append("price_above_po")
        elif ip < pp:
            kinds.append("price_below_po")
    if iq is not None and oq is not None and pp is not None:
        qty_diff = (iq - oq) * pp
        if iq > oq:
            kinds.append("quantity_above_ordered")
        elif iq < oq:
            kinds.append("quantity_below_ordered")
    out.update({"kind": "+".join(kinds) if kinds else "matched",
                "price_difference": float(round(price_diff, 2)), "quantity_difference_at_po_price": float(round(qty_diff, 2))})
    return out


def reconcile(supplier: Any = None, invoice_id: Any = None, po_id: Any = None, start: Any = None, end: Any = None, *,
              rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
              today: Optional[date] = None) -> Dict[str, Any]:
    read = rows or store.rows
    resolved, reply = supplier_resolve.resolve(supplier, rows=read)
    if reply is not None:
        if reply.get("available") is False:
            reply["note"] += " Nothing could be reconciled, which says nothing about whether the invoices are correct or incorrect."
        return reply
    sid, sname = resolved
    now = today or date.today()
    default = Period(now - timedelta(days=DEFAULT_WINDOW_DAYS), now, f"last {DEFAULT_WINDOW_DAYS} days")
    try:
        period = Period(_day(start, default.start, "start"), _day(end, default.end, "end"),
                        default.label if start is None and end is None else "custom")
    except ValueError as exc:
        return {"found": False, "note": str(exc)}
    if period.start > period.end:
        return {"found": False, "note": "start is after end."}
    inv = str(invoice_id).strip() if invoice_id not in (None, "") else None
    po = str(po_id).strip() if po_id not in (None, "") else None
    try:
        lines_raw = read(_MATCHED_SQL, [sid, period.start, period.end, sid, period.start, period.end, inv, inv, po, po])
        po_ids = sorted({r[2] for r in lines_raw if r[2]})
        receipts_raw = read(_RECEIPTS_SQL, [po_ids]) if po_ids else []
        findings_raw = read(_FINDINGS_SQL, [sid, inv, inv, MAX_FINDINGS])
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The invoice, order and receipt records could not be read just now. This does not mean "
                        "the invoices are correct or incorrect."}
    lines = [_line(r) for r in lines_raw]
    receipts = {po_: {"receipts": int(n), "first": _plain(a), "last": _plain(b)} for po_, n, a, b in receipts_raw}
    for ln in lines:
        rec = receipts.get(ln["po_id"])
        ln["receipts_on_po"] = rec["receipts"] if rec else 0
        ln["received_qty"] = None  # the receipt records carry no quantities
    totals: Dict[str, Dict[str, float]] = {}
    kinds_count: Dict[str, int] = {}
    for ln in lines:
        kinds_count[ln["kind"]] = kinds_count.get(ln["kind"], 0) + 1
        if ln.get("matched"):
            t = totals.setdefault(ln["currency"] or "unknown", {"price_above_po": 0.0, "price_below_po": 0.0,
                                                                   "quantity_above_ordered_at_po_price": 0.0,
                                                                   "quantity_below_ordered_at_po_price": 0.0, "lines": 0})
            t["lines"] += 1
            pd_, qd = ln["price_difference"], ln["quantity_difference_at_po_price"]
            t["price_above_po" if pd_ > 0 else "price_below_po"] += pd_ if pd_ != 0 else 0.0
            t["quantity_above_ordered_at_po_price" if qd > 0 else "quantity_below_ordered_at_po_price"] += qd if qd != 0 else 0.0
    for t in totals.values():
        for k in list(t):
            if k != "lines":
                t[k] = float(round(t[k], 2))
    findings = [{"finding_id": fid, "rule": rule, "category": cat, "severity": sev, "document_type": dt, "document": pk,
                 "field": field, "observed": obs, "expected": exp, "delta": delta, "notes": notes, "status": status,
                 "detected_at": _plain(at)} for fid, rule, cat, sev, dt, pk, field, obs, exp, delta, notes, status, at in findings_raw]
    out: Dict[str, Any] = {"found": True, "supplier": {"supplier_id": sid, "name": sname},
                           "period": {"label": period.label, "start": str(period.start), "end": str(period.end)},
                           "invoice_id": inv, "po_id": po, "invoice_lines": len(lines), "lines": lines[:MAX_LINES],
                           "kinds": kinds_count, "differences_by_currency": totals,
                           "receipts_by_po": receipts, "pos_without_receipt": [p for p in po_ids if p not in receipts],
                           "open_findings": findings, "open_findings_count": len(findings),
                           "written": False}
    notes = ["Each invoice line is matched to the PO line with the same item on the same PO; ordered quantity is the "
             "PO's, invoiced is the invoice's, and the differences are (invoiced unit price - PO unit price) x "
             "invoiced quantity, and (invoiced quantity - ordered quantity) x PO unit price, summed per currency.",
             "Goods receipts are recorded per PO without quantities, so 'received' is never known here: a PO with no "
             "receipt recorded is our gap in records, not evidence that nothing was delivered.",
             "Findings are what the triage engine recorded, with the values it observed and expected; a finding is a "
             "catch to examine, not a verdict against the supplier."]
    if not lines:
        notes.insert(0, "No invoice line from this supplier in the period" + (" for that invoice" if inv else "")
                     + (" on that PO" if po else "") + ".")
    if len(lines) > MAX_LINES:
        notes.append(f"Showing the first {MAX_LINES} of {len(lines)} lines; narrow by invoice_id or po_id.")
    out["note"] = " ".join(notes)
    return out
