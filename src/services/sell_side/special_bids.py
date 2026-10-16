"""A distributor's deal cost for one customer: the special bid.

A bid is registered by the distributor for a named customer, with a reference, a cost, an
approved quantity band and an expiry (a bid always expires). It is keyed by
(distributor_id, distributor_sku) so a catalog reprice does not orphan it. It applies to a
line when it is active, in its window, in the quote's currency, for that customer, and the
quantity is inside its band. Of those, the cheapest cost wins.

Its cost is internal: it lives in the internal tables and never reaches a customer view.
A quote that rests on a bid must not be approved or issued after the bid has ended or been
withdrawn (`stale_lines`), because the cost it was margined on no longer holds.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional

from src.services.sell_side._db import NotFound, dict_cursor

_COLS = ("special_bid_id, account_id, distributor_id, distributor_sku, bid_reference, "
         "cost_price, currency, min_quantity, max_quantity, valid_from, valid_to, status, "
         "created_by, created_date")


@dataclass(frozen=True)
class Bid:
    special_bid_id: int
    cost_price: Decimal
    bid_reference: str


def resolve(cur: Any, *, account_id: str, distributor_id: str, distributor_sku: str,
            currency: str, quantity: Decimal, on_date: Optional[dt.date] = None) -> Optional[Bid]:
    """``cur`` must be a RealDictCursor. None when no bid applies."""
    on_date = on_date or dt.date.today()
    cur.execute(
        "SELECT special_bid_id, cost_price, bid_reference FROM proc.bp_catalog_special_bid "
        "WHERE account_id = %s AND distributor_id = %s AND distributor_sku = %s "
        "AND status = 'active' AND currency = %s AND min_quantity <= %s "
        "AND (max_quantity IS NULL OR max_quantity >= %s) AND valid_from <= %s AND valid_to >= %s "
        "ORDER BY cost_price ASC, special_bid_id DESC LIMIT 1",
        (account_id, distributor_id, distributor_sku, currency, quantity, quantity,
         on_date, on_date))
    row = cur.fetchone()
    return Bid(row["special_bid_id"], row["cost_price"], row["bid_reference"]) if row else None


def register(conn: Any, *, account_id: str, distributor_id: str, distributor_sku: str,
             bid_reference: str, cost_price: Decimal, currency: str, valid_from: dt.date,
             valid_to: Optional[dt.date], min_quantity: Decimal = Decimal(0),
             max_quantity: Optional[Decimal] = None, created_by: Optional[str] = None
             ) -> Dict[str, Any]:
    if not (bid_reference or "").strip():
        raise ValueError("a special bid needs the distributor's bid reference")
    if valid_to is None:
        raise ValueError("a special bid always expires; valid_to is required")
    if valid_to < valid_from:
        raise ValueError("valid_to is before valid_from")
    if valid_to < dt.date.today():
        raise ValueError("valid_to is already in the past")
    if Decimal(cost_price) < 0:
        raise ValueError("cost_price cannot be negative")
    if Decimal(min_quantity) < 0 or (max_quantity is not None
                                     and Decimal(max_quantity) < Decimal(min_quantity)):
        raise ValueError("the quantity band is not valid")
    if len((currency or "").strip()) != 3:
        raise ValueError("currency must be a three-letter code")
    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT 1 FROM proc.bp_account WHERE account_id = %s", (account_id,))
        if cur.fetchone() is None:
            raise NotFound(f"account {account_id!r} does not exist")
        cur.execute(
            "INSERT INTO proc.bp_catalog_special_bid (account_id, distributor_id, "
            "distributor_sku, bid_reference, cost_price, currency, min_quantity, max_quantity, "
            f"valid_from, valid_to, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            f"RETURNING {_COLS}",
            (account_id, distributor_id, distributor_sku, bid_reference.strip(), cost_price,
             currency.strip().upper(), min_quantity, max_quantity, valid_from, valid_to, created_by))
        row = dict(cur.fetchone())
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row


def withdraw(conn: Any, special_bid_id: int, *, actor: Optional[str] = None) -> Dict[str, Any]:
    cur = dict_cursor(conn)
    try:
        cur.execute(f"UPDATE proc.bp_catalog_special_bid SET status = 'withdrawn' "
                    f"WHERE special_bid_id = %s RETURNING {_COLS}", (special_bid_id,))
        row = cur.fetchone()
        if row is None:
            raise NotFound(f"special bid {special_bid_id} does not exist")
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return dict(row)


def list_for_account(conn: Any, account_id: str) -> List[Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(f"SELECT {_COLS} FROM proc.bp_catalog_special_bid WHERE account_id = %s "
                "ORDER BY distributor_id, distributor_sku, special_bid_id", (account_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.rollback()
    return rows


def stale_lines(cur: Any, sales_quote_id: int,
                on_date: Optional[dt.date] = None) -> List[Dict[str, Any]]:
    """Lines whose cost rests on a bid that has since ended or been withdrawn."""
    on_date = on_date or dt.date.today()
    cur.execute(
        "SELECT l.line_no, b.special_bid_id, b.bid_reference, b.status, b.valid_to "
        "FROM proc.bp_sales_quote_line_internal i "
        "JOIN proc.bp_sales_quote_line l ON l.sales_quote_line_id = i.sales_quote_line_id "
        "JOIN proc.bp_catalog_special_bid b ON b.special_bid_id = i.special_bid_id "
        "WHERE i.sales_quote_id = %s AND (b.status <> 'active' OR b.valid_to < %s) "
        "ORDER BY l.line_no", (sales_quote_id, on_date))
    return [dict(r) for r in cur.fetchall()]
