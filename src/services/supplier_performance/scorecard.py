"""calc_supplier_scorecard: one supplier's performance from the organisation's own records. Read-only.

Dimensions, each measured only where the records allow and otherwise reported as "not assessed":

  spend              invoiced amounts in the period, converted with the dashboards' currency rules
  price_variance     ``ppv.purchase_price_variance`` over invoice lines matched one-to-one to a PO line
  on_time_delivery   ``otif.delivery_performance`` over POs in the period: the PO's expected delivery
                     date against the last goods receipt recorded for it. Received quantities are not
                     recorded, so "in full" is never assessed here; only the on-time rate can be.
  open_findings      triage findings still open on this supplier's deals, counted now

The on-time tolerances come from the governed ``supplier_scorecard`` policy (keys
``otif_late_tolerance_days``, ``otif_in_full_tolerance``, ``otif_early_tolerance_days``). Without that
policy the on-time rate is not assessed and the reason is stated: a tolerance is a rule, not a default.
No composite is formed here: weights belong to the category owner, who can run
``supplier.composite_score`` on these dimensions.
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services.analytics.currency import DEFAULT_CURRENCY, DisplayCurrency
from src.services.analytics.period import Period, fiscal_year_to_date
from src.services.contract_tools import store
from src.services.supplier_performance import otif as _otif
from src.services.supplier_performance import ppv as _ppv

log = logging.getLogger(__name__)

POLICY = "supplier_scorecard"

_SUPPLIER_SQL = """
    SELECT s.supplier_id::text, s.supplier_name
      FROM proc.bp_supplier s
     WHERE s.supplier_id::text = %s OR s.supplier_name ILIKE %s OR s.trading_name ILIKE %s
     ORDER BY s.supplier_name
     LIMIT 6
"""
_SPEND_SQL = """
    SELECT UPPER(COALESCE(i.currency, '')), SUM(i.invoice_amount)::numeric, COUNT(*)::int
      FROM proc.bp_invoice_trgt i
     WHERE i.supplier_id = %s AND i.invoice_date BETWEEN %s AND %s AND i.invoice_amount IS NOT NULL
     GROUP BY 1
"""
# Invoice lines matched to exactly one PO line on (po_id, item_id), billed in the period, with both
# header currencies so a pair priced in different currencies is left unassessed, never taken 1:1.
_PRICE_LINES_SQL = """
WITH pol AS (
    SELECT l.po_id, l.item_id, MIN(l.unit_price) AS up, MIN(h.currency) AS cur, COUNT(*) AS n
      FROM proc.bp_po_line_items_trgt l
      JOIN proc.bp_purchase_order_trgt h USING (po_id)
     WHERE l.item_id IS NOT NULL AND l.unit_price IS NOT NULL
     GROUP BY l.po_id, l.item_id
)
SELECT pol.up, il.unit_price, il.quantity, h.currency, pol.cur
  FROM proc.bp_invoice_line_items_trgt il
  JOIN proc.bp_invoice_trgt h ON h.invoice_id = il.invoice_id
  JOIN pol ON pol.po_id = il.po_id AND pol.item_id = il.item_id
 WHERE pol.n = 1
   AND h.supplier_id = %s
   AND h.invoice_date BETWEEN %s AND %s
   AND il.unit_price IS NOT NULL AND il.quantity IS NOT NULL
"""
# One row per PO: the promised date and the date of the LAST receipt against it (a delivery is
# complete when its last receipt lands). A PO with no receipt row is unassessed, not late: an
# unrecorded receipt and a missing delivery look the same in our records.
_DELIVERIES_SQL = """
    SELECT p.po_id, p.expected_delivery_date, MAX(g.receipt_date)
      FROM proc.bp_purchase_order_trgt p
      LEFT JOIN proc.bp_goods_receipt_trgt g ON g.po_id = p.po_id
     WHERE p.supplier_id = %s AND p.order_date BETWEEN %s AND %s
     GROUP BY p.po_id, p.expected_delivery_date
"""
_FINDINGS_SQL = """
    SELECT COUNT(*)::int
      FROM proc.bp_detection_finding f
      JOIN proc.bp_deal_overview o ON o.deal_id = f.deal_id
     WHERE f.status = 'open' AND o.supplier_id = %s
"""


def _day(value: Any, default: date, name: str) -> date:
    if value is None or value == "":
        return default
    try:
        return date.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"{name} must be a date like 2026-01-31.") from None


def _governed_tolerances() -> Dict[str, Any]:
    from src.services import governed_limits

    return governed_limits.rules(POLICY)


def _on_time(deliveries: List[Dict[str, Any]], limits: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
    try:
        rules = limits()
        late = float(rules["otif_late_tolerance_days"])
        in_full = float(rules["otif_in_full_tolerance"])
        early = rules.get("otif_early_tolerance_days")
        early = None if early is None else float(early)
    except Exception as exc:  # noqa: BLE001 - a missing or malformed rule is "not assessed", never a default
        return {"assessed": False, "formula": "otif.delivery_performance", "purchase_orders": len(deliveries),
                "reason": f"the governed {POLICY} policy does not state usable on-time tolerances ({exc}); "
                          "a tolerance is a rule, not a default, so the on-time rate is not assessed"}
    try:
        result = _otif.otif(deliveries=deliveries, late_tolerance_days=late, in_full_tolerance=in_full,
                            early_tolerance_days=early)
    except ValueError as exc:
        return {"assessed": False, "formula": "otif.delivery_performance", "purchase_orders": len(deliveries),
                "reason": f"the governed {POLICY} tolerances are not usable: {exc}"}
    if result is None or not result.get("on_time_assessed"):
        return {"assessed": False, "formula": "otif.delivery_performance", "purchase_orders": len(deliveries),
                "reason": "no PO in the period has both an expected delivery date and a goods receipt recorded"}
    return {"assessed": True, "formula": "otif.delivery_performance",
            "on_time_rate": result["on_time_rate"], "on_time_assessed": result["on_time_assessed"],
            "purchase_orders": len(deliveries), "unassessed": len(deliveries) - result["on_time_assessed"],
            "in_full": "not assessed: received quantities are not recorded",
            "tolerances": {"late_days": late, "early_days": early, "source": f"policy {POLICY}"}}


def _price_variance(lines_raw: Sequence[tuple]) -> Dict[str, Any]:
    lines, mixed = [], 0
    for po_up, inv_up, qty, inv_ccy, po_ccy in lines_raw:
        a, b = str(inv_ccy or "").strip().upper(), str(po_ccy or "").strip().upper()
        if a and b and a != b:
            mixed += 1
            continue
        lines.append({"po_unit_price": po_up, "invoice_unit_price": inv_up, "quantity": qty, "currency": a or b})
    result = _ppv.purchase_price_variance(lines=lines) if lines else None
    if result is None:
        return {"assessed": False, "formula": "ppv.purchase_price_variance", "lines_matched": len(lines_raw),
                "reason": "no invoice line in the period matches exactly one PO line with a usable price and quantity"
                          + (f"; {mixed} matched in a different currency from the PO" if mixed else "")}
    out = {"assessed": True, "formula": "ppv.purchase_price_variance", "lines_matched": len(lines_raw)}
    out.update({k: v for k, v in result.items()})
    if mixed:
        out["mixed_currency_lines"] = mixed
    return out


def build(supplier: Any = None, start: Any = None, end: Any = None, currency: Any = None, *,
          rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
          limits: Optional[Callable[[], Dict[str, Any]]] = None,
          display: Optional[DisplayCurrency] = None, today: Optional[date] = None) -> Dict[str, Any]:
    name = str(supplier).strip() if supplier is not None else ""
    if not name:
        return {"found": False, "note": "Give the supplier's name or supplier_id."}
    default = fiscal_year_to_date(today or date.today())
    try:
        period = Period(_day(start, default.start, "start"), _day(end, default.end, "end"),
                        default.label if start is None and end is None else "custom")
    except ValueError as exc:
        return {"found": False, "note": str(exc)}
    if period.start > period.end:
        return {"found": False, "note": "start is after end."}
    read = rows or store.rows
    outage = {"found": False, "available": False,
              "note": "The supplier records could not be read just now. This does not mean the supplier has no history."}
    try:
        matches = read(_SUPPLIER_SQL, [name, f"%{name}%", f"%{name}%"])
    except store.StoreUnavailable:
        return outage
    if not matches:
        return {"found": False, "note": f"No supplier on record matches '{name}'. Suppliers not on record cannot be scored."}
    if len(matches) > 1:
        return {"found": False, "ambiguous": True,
                "candidates": [{"supplier_id": sid, "name": nm} for sid, nm in matches],
                "note": "Several suppliers match; ask the user which one, then pass its supplier_id."}
    sid, sname = matches[0]
    try:
        spend_rows = read(_SPEND_SQL, [sid, period.start, period.end])
        price_rows = read(_PRICE_LINES_SQL, [sid, period.start, period.end])
        delivery_rows = read(_DELIVERIES_SQL, [sid, period.start, period.end])
        findings_rows = read(_FINDINGS_SQL, [sid])
        if display is None:
            from src.services.analytics.repository import display_currency_from_request
            display = display_currency_from_request(str(currency or DEFAULT_CURRENCY))
    except Exception as exc:  # noqa: BLE001 - an unreadable store is an outage, not "no history"
        log.warning("calc_supplier_scorecard: unreadable: %s", exc)
        return outage

    conv = display.total([(Decimal(str(amt or 0)), ccy) for ccy, amt, _n in spend_rows])
    spend = {"assessed": bool(spend_rows), "invoices": sum(int(n or 0) for _c, _a, n in spend_rows),
             "currency": display.target, "amount": None if conv.value is None else float(round(conv.value, 2)),
             "as_billed": {ccy or "unknown": float(round(Decimal(str(amt or 0)), 2)) for ccy, amt, _n in spend_rows},
             "excluded_currencies": list(conv.excluded_currencies)}
    if not spend_rows:
        spend["reason"] = "no invoice from this supplier in the period"
    deliveries = [{"promised_date": promised, "received_date": received, "ordered_qty": None, "received_qty": None}
                  for _po, promised, received in delivery_rows]
    dims = {"spend": spend, "price_variance": _price_variance(price_rows),
            "on_time_delivery": _on_time(deliveries, limits or _governed_tolerances),
            "open_findings": {"assessed": True, "count": int(findings_rows[0][0]) if findings_rows else 0,
                              "note": "open triage findings on this supplier's deals, counted now, not by period"}}
    not_assessed = [k for k, v in dims.items() if not v.get("assessed")]
    note = ("Measured from this organisation's own invoices, PO lines, goods receipts and findings; each dimension "
            "says how many records it rests on. Quality, responsiveness, ESG and SLA credits are not recorded, so "
            "they are not scored: ask the user. No composite is formed; the category owner's weights would go into "
            "supplier.composite_score.")
    if not_assessed:
        note += " Not assessed: " + ", ".join(not_assessed) + " (reasons given)."
    return {"found": True, "supplier": {"supplier_id": sid, "name": sname},
            "period": {"label": period.label, "start": str(period.start), "end": str(period.end)},
            "dimensions": dims, "not_assessed": not_assessed, "composite": None, "note": note}
