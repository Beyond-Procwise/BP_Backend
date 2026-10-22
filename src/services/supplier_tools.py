"""supplier_discovery and supplier_lookup: what the platform itself knows about suppliers.

Both read only the supplier master and the PO, invoice and quote lines. They find suppliers the
organisation already deals with: they are not a market scan, and a supplier they do not return may
still exist. Bank details and personal contacts are never returned.
"""
from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

log = logging.getLogger(__name__)

DEFAULT_LIMIT = 10
MAX_LIMIT = 25
MIN_TERM = 3

_DISCOVER_SQL = """
WITH hits AS (
  SELECT p.supplier_id::text AS sid, 'po' AS kind FROM proc.bp_po_line_items_trgt l
    JOIN proc.bp_purchase_order_trgt p ON p.po_id = l.po_id WHERE l.item_description ILIKE %s
  UNION ALL
  SELECT i.supplier_id::text, 'invoice' FROM proc.bp_invoice_line_items_trgt l
    JOIN proc.bp_invoice_trgt i ON i.invoice_id = l.invoice_id WHERE l.item_description ILIKE %s
  UNION ALL
  SELECT q.supplier_id::text, 'quote' FROM proc.bp_quote_line_items_trgt l
    JOIN proc.bp_quote_trgt q ON q.quote_id = l.quote_id WHERE l.item_description ILIKE %s
)
SELECT s.supplier_id::text, s.supplier_name, s.is_preferred_supplier, s.country, s.registered_country,
       s.esg_cert_iso14001, s.esg_cert_sa8000, s.esg_cert_ecovadis, s.insurance_expiry_date,
       COUNT(*) FILTER (WHERE h.kind = 'po')      AS po_lines,
       COUNT(*) FILTER (WHERE h.kind = 'invoice') AS invoice_lines,
       COUNT(*) FILTER (WHERE h.kind = 'quote')   AS quote_lines
  FROM hits h JOIN proc.bp_supplier s ON s.supplier_id::text = h.sid
 {where}
 GROUP BY s.supplier_id, s.supplier_name, s.is_preferred_supplier, s.country, s.registered_country,
          s.esg_cert_iso14001, s.esg_cert_sa8000, s.esg_cert_ecovadis, s.insurance_expiry_date
 ORDER BY COUNT(*) DESC, s.supplier_name
 LIMIT %s
"""

_LOOKUP_SQL = """
SELECT s.supplier_id::text, s.supplier_name, s.trading_name, s.supplier_type, s.legal_structure,
       s.registered_country, s.registration_number, s.is_preferred_supplier, s.risk_score,
       s.credit_limit_amount, s.esg_cert_iso14001, s.esg_cert_sa8000, s.esg_cert_ecovadis,
       s.insurance_coverage_type, s.insurance_coverage_amount, s.insurance_expiry_date,
       s.default_currency, s.incoterms, s.delivery_lead_time_days, s.country, s.website_url,
       (SELECT COUNT(*) FROM proc.bp_invoice_trgt i WHERE i.supplier_id = s.supplier_id) AS invoices
  FROM proc.bp_supplier s
 WHERE {where}
 ORDER BY s.supplier_name
 LIMIT 6
"""


def _read(sql: str, params: Sequence[Any]) -> List[tuple]:
    from src.services.db import get_conn

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(sql, tuple(params))
        return list(cur.fetchall())


def _num(value: Any) -> Any:
    """Stored amounts arrive as Decimal; the tool result must be plain JSON."""
    return float(value) if isinstance(value, Decimal) else value


def _like(text: str) -> str:
    return "%" + text.replace("%", "\\%").replace("_", "\\_") + "%"


def _limit(value: Any) -> int:
    try:
        return max(1, min(int(value), MAX_LIMIT))
    except (TypeError, ValueError):
        return DEFAULT_LIMIT


def _certs(iso14001: Any, sa8000: Any, ecovadis: Any) -> List[str]:
    return [name for name, on in (("ISO 14001", iso14001), ("SA8000", sa8000), ("EcoVadis", ecovadis)) if on]


def _insurance(expiry: Any, today: date) -> Dict[str, Any]:
    if expiry is None:
        return {"insurance_expiry": None, "insurance_status": "not on file"}
    d = expiry.date() if hasattr(expiry, "date") else expiry
    return {"insurance_expiry": str(d), "insurance_status": "expired" if d < today else "in date"}


def discover(item: Any = None, country: Any = None, preferred_only: Any = None, limit: Any = None, *,
             rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
             today: Optional[date] = None) -> Dict[str, Any]:
    if not isinstance(item, str) or len(item.strip()) < MIN_TERM:
        return {"found": False, "note": f"Give the item or service to look for, at least {MIN_TERM} letters."}
    n = _limit(limit)
    where, extra = [], []
    if isinstance(country, str) and country.strip():
        where.append("(s.country ILIKE %s OR s.registered_country ILIKE %s)")
        extra += [country.strip(), country.strip()]
    if preferred_only in (True, "true", "True"):
        where.append("s.is_preferred_supplier IS TRUE")
    sql = _DISCOVER_SQL.format(where=("WHERE " + " AND ".join(where)) if where else "")
    pat = _like(item.strip())
    try:
        found = (rows or _read)(sql, [pat, pat, pat, *extra, n + 1])
    except Exception as exc:  # noqa: BLE001 - an unreadable store is an outage, not "no suppliers"
        log.warning("supplier_discovery: unreadable: %s", exc)
        return {"found": False, "available": False,
                "note": "Supplier records could not be read just now. This does not mean there are no suppliers."}
    today = today or date.today()
    suppliers = []
    for r in found[:n]:
        sid, name, pref, ctry, reg, c1, c2, c3, ins, po, inv, qt = r
        suppliers.append({"supplier_id": sid, "name": name, "preferred": bool(pref), "country": ctry or reg,
                          "certifications_on_file": _certs(c1, c2, c3), **_insurance(ins, today),
                          "evidence": {"po_lines": po, "invoice_lines": inv, "quote_lines": qt}})
    out: Dict[str, Any] = {"found": bool(suppliers), "item": item.strip(), "suppliers": suppliers,
                           "note": ("Suppliers this organisation has already ordered, been invoiced or quoted by for "
                                    "something matching the item. This is not a market scan: other suppliers exist. "
                                    "Certifications are as recorded, not verified.")}
    if len(found) > n:
        out["note"] += f" More than {n} match; narrow by country or preferred_only."
    if not suppliers:
        out["note"] = ("No known supplier has a line matching that wording. Try another word for the item; "
                       "the market scan is not covered by this tool.")
    return out


def lookup(supplier: Any = None, *, rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None,
           today: Optional[date] = None) -> Dict[str, Any]:
    if not isinstance(supplier, str) or len(supplier.strip()) < 2:
        return {"found": False, "note": "Give the supplier's name or supplier_id."}
    text = supplier.strip()
    sql = _LOOKUP_SQL.format(where="s.supplier_id::text = %s OR s.supplier_name ILIKE %s OR s.trading_name ILIKE %s")
    try:
        found = (rows or _read)(sql, [text, _like(text), _like(text)])
    except Exception as exc:  # noqa: BLE001
        log.warning("supplier_lookup: unreadable: %s", exc)
        return {"found": False, "available": False,
                "note": "Supplier records could not be read just now. This does not mean the supplier is unknown."}
    if not found:
        return {"found": False, "note": "No supplier on file by that name or id. Do not assume its status; ask the user."}
    today = today or date.today()
    profiles = []
    asked = text.lower()
    for r in found[:5]:
        (sid, name, trading, stype, legal, reg_ctry, reg_no, pref, risk, credit, c1, c2, c3, ins_type, ins_amt,
         ins_exp, ccy, inco, lead, ctry, site, invoices) = r
        # How this row was found: a partial name is a guess the model made, not a supplier the user named.
        match = ("exact id" if str(sid).lower() == asked else
                 "exact name" if asked in (str(name or "").lower(), str(trading or "").lower()) else "partial name")
        profiles.append({
            "supplier_id": sid, "name": name, "trading_name": trading, "match": match, "type": stype, "legal_structure": legal,
            "registered_country": reg_ctry, "registration_number": reg_no, "country": ctry,
            "preferred": bool(pref), "risk_score": risk if risk is not None else "not scored",
            "credit_limit": _num(credit), "default_currency": ccy, "incoterms": inco, "lead_time_days": lead,
            "certifications_on_file": _certs(c1, c2, c3),
            "insurance": {"type": ins_type, "amount": _num(ins_amt), **_insurance(ins_exp, today)},
            "website": site, "invoices_on_file": invoices})
    out: Dict[str, Any] = {"found": True, "suppliers": profiles,
                           "note": "As recorded in the supplier master; nothing here is verified against an outside source."}
    if len(found) > 5:
        out["note"] += " More than five suppliers match that name; ask which one."
    elif len(found) > 1:
        out["note"] += " Several suppliers match that name; confirm which one before relying on it."
    elif profiles[0]["match"] == "partial name":
        out["note"] += (f" Matched on part of the name only ('{text}' is not this supplier's full name or id): confirm "
                        "with the user that this is the supplier they mean before relying on it.")
    return out
