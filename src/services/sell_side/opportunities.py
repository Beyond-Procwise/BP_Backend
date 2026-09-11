"""A sell-side opportunity: one catalog SKU one account should be buying.

win_probability is not a parameter of anything here. Calibration writes it,
from closed outcomes, or nothing does (spec §4.5).
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from src.services.sell_side import ladder
from src.services.sell_side._db import NotFound, dict_cursor
from src.services.sell_side.costing import cost_at
from src.services.sell_side.money import iso_currency, price_line, q2

OPPORTUNITY_TYPES = frozenset({"upsell", "cross_sell", "upgrade", "refill", "switch_supplier"})
JUSTIFICATION_KINDS = frozenset(
    {"price_gap", "benchmark", "end_of_life", "usage_cadence", "coverage_gap"})


def create_opportunity(
    conn: Any, *, account_id: str, opportunity_type: str,
    catalog_item_id: Optional[int] = None, currency: Optional[str] = None,
    expected_quantity: Optional[Decimal] = None,
    expected_unit_price: Optional[Decimal] = None,
    phase_id: Optional[str] = "sales.opportunity",
    subprocess_id: Optional[str] = "sales.opportunity.qualified",
    detector_type: Optional[str] = None, reason_codes: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if opportunity_type not in OPPORTUNITY_TYPES:
        raise ValueError(f"opportunity_type must be one of {sorted(OPPORTUNITY_TYPES)}")
    ladder.check(phase_id, subprocess_id)
    if expected_quantity is not None and expected_quantity <= 0:
        raise ValueError("expected_quantity must be greater than zero")
    if expected_unit_price is not None and expected_unit_price < 0:
        raise ValueError("expected_unit_price cannot be negative")
    if expected_unit_price is not None:
        exponent = expected_unit_price.as_tuple().exponent
        if isinstance(exponent, int) and exponent < -4:
            raise ValueError("expected_unit_price has more than 4 decimal places")
    wanted = iso_currency(currency) if currency is not None else None

    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT 1 FROM proc.bp_account WHERE account_id = %s", (account_id,))
        if cur.fetchone() is None:
            raise NotFound(f"account {account_id!r} does not exist")

        revenue = cost = margin = margin_pct = None
        if catalog_item_id is not None:
            c = cost_at(cur, catalog_item_id, expected_quantity or Decimal(1))
            if wanted is not None and wanted != c.currency:
                raise ValueError(f"catalog item is priced in {c.currency}, not {wanted}; "
                                 "no FX conversion is performed")
            wanted = c.currency
            if expected_quantity is not None:
                if expected_unit_price is not None:
                    p = price_line(expected_quantity, expected_unit_price, c.unit_cost, c.list_price)
                    revenue, cost, margin, margin_pct = (
                        p.line_total, p.line_cost, p.line_margin, p.line_margin_pct)
                elif c.unit_cost is not None:
                    cost = q2(expected_quantity * c.unit_cost)
        elif expected_quantity is not None and expected_unit_price is not None:
            revenue = q2(expected_quantity * expected_unit_price)
        if margin_pct is not None and abs(margin_pct) >= 1000:
            raise ValueError(f"margin_pct {margin_pct} is out of range")
        if wanted is None:
            raise ValueError("currency is required when no catalog item is named")

        cur.execute(
            "INSERT INTO proc.bp_sales_opportunity (account_id, catalog_item_id, opportunity_type, "
            "currency, expected_quantity, expected_revenue, expected_cost, expected_margin, "
            "margin_pct, phase_id, subprocess_id, detector_type, reason_codes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
            (account_id, catalog_item_id, opportunity_type, wanted, expected_quantity,
             revenue, cost, margin, margin_pct, phase_id, subprocess_id, detector_type,
             list(reason_codes) if reason_codes else None))
        row = dict(cur.fetchone())
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row


def get_opportunity(conn: Any, sales_opportunity_id: int) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    cur.execute("SELECT * FROM proc.bp_sales_opportunity WHERE sales_opportunity_id = %s",
                (sales_opportunity_id,))
    row = cur.fetchone()
    if row is None:
        raise NotFound(f"opportunity {sales_opportunity_id} does not exist")
    out = dict(row)
    cur.execute("SELECT * FROM proc.bp_sales_justification WHERE sales_opportunity_id = %s "
                "ORDER BY justification_id", (sales_opportunity_id,))
    out["justifications"] = [dict(r) for r in cur.fetchall()]
    return out


def list_opportunities(conn: Any, *, account_id: Optional[str] = None,
                       outcome: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT * FROM proc.bp_sales_opportunity WHERE (%s IS NULL OR account_id = %s) "
        "AND (%s IS NULL OR outcome = %s) ORDER BY sales_opportunity_id DESC LIMIT %s",
        (account_id, account_id, outcome, outcome, max(1, min(limit, 500))))
    return [dict(r) for r in cur.fetchall()]


def add_justification(conn: Any, sales_opportunity_id: int, *, kind: str, claim: str,
                      evidence_ref: Optional[str] = None,
                      evidence_value: Optional[Decimal] = None,
                      customer_safe: bool = True) -> Dict[str, Any]:
    if kind not in JUSTIFICATION_KINDS:
        raise ValueError(f"kind must be one of {sorted(JUSTIFICATION_KINDS)}")
    if not (claim or "").strip():
        raise ValueError("claim is empty")
    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT 1 FROM proc.bp_sales_opportunity WHERE sales_opportunity_id = %s",
                    (sales_opportunity_id,))
        if cur.fetchone() is None:
            raise NotFound(f"opportunity {sales_opportunity_id} does not exist")
        cur.execute(
            "INSERT INTO proc.bp_sales_justification (sales_opportunity_id, kind, claim, "
            "evidence_ref, evidence_value, customer_safe) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
            (sales_opportunity_id, kind, claim.strip(), evidence_ref, evidence_value,
             bool(customer_safe)))
        row = dict(cur.fetchone())
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row


def set_stage(conn: Any, sales_opportunity_id: int, *, phase_id: str,
              subprocess_id: Optional[str]) -> Dict[str, Any]:
    ladder.check(phase_id, subprocess_id)
    cur = dict_cursor(conn)
    try:
        cur.execute(
            "UPDATE proc.bp_sales_opportunity SET phase_id = %s, subprocess_id = %s, "
            "last_modified_date = now() WHERE sales_opportunity_id = %s RETURNING *",
            (phase_id, subprocess_id, sales_opportunity_id))
        row = cur.fetchone()
        if row is None:
            raise NotFound(f"opportunity {sales_opportunity_id} does not exist")
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return dict(row)
