"""Opportunity buckets: one supplier and item, with every lever found on it.

A unit price and a volume describe one piece of spend. Several detectors can each
find a saving on that same spend -- the price is above benchmark, the volume earns
a discount, the supplier could be consolidated -- and each used to reach the user
as its own separate opportunity. They are not separate opportunities: they are
different levers on one thing.

A bucket is read from proc.bp_opportunity; nothing is stored. It exists where a
supplier and item carry a unit price and a quantity. Each stored finding on that
supplier and item is one lever inside it. Everything else stays an individual
opportunity.

The bucket's saving is its largest open lever, never the sum: levers on the same
spend overlap, and adding them would count the same money more than once.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.services.db import get_conn

log = logging.getLogger(__name__)

_OPEN = ("identified", "negotiation", "agreed")
SAVING_BASIS = ("largest open lever; levers on the same spend overlap and are "
                "not added together")

_UNIT_PRICE_KEYS = ("actual_price", "unit_price")
_QUANTITY_KEYS = ("quantity", "total_quantity")


def bucket_id(supplier_id: Any, item_id: Any) -> Optional[str]:
    """The bucket a supplier and item belong to, or None when either is missing."""
    supplier = str(supplier_id).strip() if supplier_id is not None else ""
    item = str(item_id).strip() if item_id is not None else ""
    if not supplier or not item:
        return None
    return f"{supplier}|{item}".lower()


def _num(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _mapping(value: Any) -> dict:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return {}
    return value if isinstance(value, dict) else {}


def _listing(value: Any) -> list:
    # A JSONB column normally arrives parsed. It arrives as text when the driver has
    # been told not to parse it (scripts/testdata/db.py does, process-wide).
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            return []
    return list(value) if isinstance(value, (list, tuple)) else []


def _first(calc: dict, keys: tuple) -> Optional[float]:
    for key in keys:
        found = _num(calc.get(key))
        if found is not None:
            return found
    return None


def _iso(value: Any) -> Optional[str]:
    return value.isoformat() if hasattr(value, "isoformat") else (str(value) if value else None)


def _lever(row: dict) -> dict:
    calc = _mapping(row.get("calculation_details"))
    review = _mapping(row.get("review_detail"))
    asked = row.get("review_requested_at")
    return {
        "opportunityId": row.get("opportunity_id"),
        "lever": row.get("detector_type") or "",
        "saving": _num(row.get("financial_impact_gbp")),
        "stage": row.get("stage"),
        "unitPrice": _num(row.get("unit_price")) if row.get("unit_price") is not None
        else _first(calc, _UNIT_PRICE_KEYS),
        "quantity": _num(row.get("quantity")) if row.get("quantity") is not None
        else _first(calc, _QUANTITY_KEYS),
        "evidence": _listing(row.get("source_records")),
        "detectedOn": _iso(row.get("detected_on")),
        # New information has arrived on a lever a person rejected.
        "reviewRequested": asked is not None,
        "reviewRequestedAt": _iso(asked),
        "whatChanged": review if asked is not None else None,
        "rejectionReason": row.get("rejection_reason"),
    }


def group(rows: list) -> dict:
    """Sort stored findings into buckets and individual opportunities.

    ``rows`` are dicts of proc.bp_opportunity columns. Returns
    ``{"buckets": [...], "individual": [...]}`` where each individual entry is a
    lever on its own.
    """
    by_bucket: dict = {}
    individual = []
    for row in rows:
        key = bucket_id(row.get("supplier_id"), row.get("item_id"))
        if key is None:
            individual.append(row)
        else:
            by_bucket.setdefault(key, []).append(row)

    buckets = []
    for key, members in by_bucket.items():
        levers = [_lever(m) for m in members]
        priced = next((l for l in levers
                       if l["unitPrice"] is not None and l["quantity"] is not None), None)
        if priced is None:
            # No unit price and volume: nothing ties these to one piece of spend.
            individual.extend(members)
            continue
        levers.sort(key=lambda l: (l["saving"] is None, -(l["saving"] or 0.0)))
        open_levers = [l for l in levers if l["stage"] in _OPEN]
        savings = [l["saving"] for l in open_levers if l["saving"] is not None]
        head = members[0]
        buckets.append({
            "bucketId": key,
            "supplierId": head.get("supplier_id"),
            "supplier": head.get("supplier_name") or head.get("supplier_id"),
            "itemId": head.get("item_id"),
            "item": head.get("item_description") or head.get("item_id"),
            "unitPrice": priced["unitPrice"],
            "quantity": priced["quantity"],
            "spend": round(priced["unitPrice"] * priced["quantity"], 2),
            "levers": levers,
            "leverCount": len(levers),
            "openLeverCount": len(open_levers),
            "potentialSaving": max(savings) if savings else None,
            "savingBasis": SAVING_BASIS,
            "reviewRequested": any(l["reviewRequested"] for l in levers),
        })
    buckets.sort(key=lambda b: (b["potentialSaving"] is None, -(b["potentialSaving"] or 0.0)))
    return {"buckets": buckets, "individual": [_lever(r) | {
        "supplier": r.get("supplier_name") or r.get("supplier_id"),
        "item": r.get("item_description") or r.get("item_id"),
    } for r in individual]}


_COLUMNS = ("opportunity_id", "detector_type", "supplier_id", "supplier_name", "item_id",
            "item_description", "financial_impact_gbp", "stage", "unit_price", "quantity",
            "calculation_details", "source_records", "detected_on",
            "review_requested_at", "review_detail")


def _fetch(cur, where: str = "", params: tuple = ()) -> list:
    # proc.opportunity_feedback is created on the first rejection, so it may not
    # exist yet; a missing table is "no reasons recorded", not an error.
    cur.execute("select to_regclass('proc.opportunity_feedback') is not null")
    has_feedback = bool((cur.fetchone() or [False])[0])
    reason = ("(select f.reason from proc.opportunity_feedback f "
              "where f.opportunity_id = o.opportunity_id and f.status = 'rejected')"
              if has_feedback else "null")
    cur.execute(
        f"select {', '.join('o.' + c for c in _COLUMNS)}, {reason} as rejection_reason "
        "from proc.bp_opportunity o "
        # Retired by a mining run: no longer detected, so not a lever on anything.
        f"where o.retired_at is null {where} "
        "order by o.financial_impact_gbp desc nulls last", params)
    names = list(_COLUMNS) + ["rejection_reason"]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def list_buckets(conn: Any = None) -> dict:
    """Every bucket, plus the opportunities that stand alone."""
    if conn is None:
        with get_conn() as own:
            return group(_fetch(own.cursor()))
    return group(_fetch(conn.cursor()))


def get_bucket(supplier_id: str, item_id: str, conn: Any = None) -> Optional[dict]:
    """One supplier and item's bucket, or None when it has no priced lever."""
    where = "and lower(btrim(o.supplier_id)) = %s and lower(btrim(o.item_id)) = %s"
    params = (str(supplier_id).strip().lower(), str(item_id).strip().lower())
    if conn is None:
        with get_conn() as own:
            found = group(_fetch(own.cursor(), where, params))
    else:
        found = group(_fetch(conn.cursor(), where, params))
    return found["buckets"][0] if found["buckets"] else None
