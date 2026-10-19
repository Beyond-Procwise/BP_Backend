"""What one customer has agreed to pay for one catalog SKU.

An agreement is keyed by (distributor_id, distributor_sku), not by catalog_item_id: a
catalog reprice closes a version and opens a new row with a new id, which would orphan
every agreement pointing at the old one.

An agreement either states a fixed unit price or a discount off list. It applies to a
quote line only when it is active, in its dated window, in the quote's currency and at or
below the line's quantity. Of those that apply, the highest quantity break wins, then the
latest start, then the newest row. An agreement that cannot yield a price (a discount on
an item with no list price) is skipped: a price is never invented to fill the gap, and a
currency is never converted.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Any, Dict, List, Optional

from src.services.sell_side._db import NotFound, dict_cursor

PRICE_KINDS = ("fixed", "discount_pct")
_COLS = ("account_price_id, account_id, distributor_id, distributor_sku, price_kind, "
         "unit_price, discount_pct, currency, min_quantity, valid_from, valid_to, "
         "reference, status, created_by, created_date")


@dataclass(frozen=True)
class AgreedPrice:
    account_price_id: int
    unit_price: Decimal
    price_kind: str


def _price_of(row: Dict[str, Any], list_price: Optional[Decimal]) -> Optional[Decimal]:
    if row["price_kind"] == "fixed":
        return Decimal(row["unit_price"])
    if list_price is None:
        return None
    return (Decimal(list_price) * (Decimal(100) - Decimal(row["discount_pct"])) / Decimal(100)
            ).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)


def resolve(cur: Any, *, account_id: str, distributor_id: str, distributor_sku: str,
            currency: str, quantity: Decimal, list_price: Optional[Decimal],
            on_date: Optional[dt.date] = None) -> Optional[AgreedPrice]:
    """``cur`` must be a RealDictCursor. None when no agreement applies."""
    on_date = on_date or dt.date.today()
    cur.execute(
        f"SELECT {_COLS} FROM proc.bp_account_price WHERE account_id = %s AND distributor_id = %s "
        "AND distributor_sku = %s AND status = 'active' AND currency = %s "
        "AND min_quantity <= %s AND valid_from <= %s AND (valid_to IS NULL OR valid_to >= %s) "
        "ORDER BY min_quantity DESC, valid_from DESC, account_price_id DESC",
        (account_id, distributor_id, distributor_sku, currency, quantity, on_date, on_date))
    for row in cur.fetchall():
        price = _price_of(row, list_price)
        if price is not None:
            return AgreedPrice(row["account_price_id"], price, row["price_kind"])
    return None


def register(conn: Any, *, account_id: str, distributor_id: str, distributor_sku: str,
             price_kind: str, currency: str, valid_from: dt.date,
             unit_price: Optional[Decimal] = None, discount_pct: Optional[Decimal] = None,
             min_quantity: Decimal = Decimal(0), valid_to: Optional[dt.date] = None,
             reference: Optional[str] = None, created_by: Optional[str] = None
             ) -> Dict[str, Any]:
    if price_kind not in PRICE_KINDS:
        raise ValueError(f"price_kind must be one of {PRICE_KINDS}")
    if price_kind == "fixed":
        if unit_price is None or discount_pct is not None or Decimal(unit_price) < 0:
            raise ValueError("a fixed agreement needs a unit_price (not negative) and no discount_pct")
    elif discount_pct is None or unit_price is not None or not (0 <= Decimal(discount_pct) < 100):
        raise ValueError("a discount agreement needs a discount_pct from 0 up to (not including) 100 "
                         "and no unit_price")
    if Decimal(min_quantity) < 0:
        raise ValueError("min_quantity cannot be negative")
    if valid_to is not None and valid_to < valid_from:
        raise ValueError("valid_to is before valid_from")
    if valid_to is not None and valid_to < dt.date.today():
        raise ValueError("valid_to is already in the past")
    if len((currency or "").strip()) != 3:
        raise ValueError("currency must be a three-letter code")
    cur = dict_cursor(conn)
    try:
        cur.execute("SELECT 1 FROM proc.bp_account WHERE account_id = %s", (account_id,))
        if cur.fetchone() is None:
            raise NotFound(f"account {account_id!r} does not exist")
        cur.execute(
            "INSERT INTO proc.bp_account_price (account_id, distributor_id, distributor_sku, "
            "price_kind, unit_price, discount_pct, currency, min_quantity, valid_from, valid_to, "
            f"reference, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING {_COLS}",
            (account_id, distributor_id, distributor_sku, price_kind, unit_price, discount_pct,
             currency.strip().upper(), min_quantity, valid_from, valid_to, reference, created_by))
        row = dict(cur.fetchone())
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return row


def withdraw(conn: Any, account_price_id: int, *, actor: Optional[str] = None) -> Dict[str, Any]:
    """Withdrawn, not deleted: a quote that used it still names it."""
    cur = dict_cursor(conn)
    try:
        cur.execute(f"UPDATE proc.bp_account_price SET status = 'withdrawn' "
                    f"WHERE account_price_id = %s RETURNING {_COLS}", (account_price_id,))
        row = cur.fetchone()
        if row is None:
            raise NotFound(f"agreed price {account_price_id} does not exist")
    except Exception:
        conn.rollback()
        raise
    conn.commit()
    return dict(row)


def list_for_account(conn: Any, account_id: str) -> List[Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(f"SELECT {_COLS} FROM proc.bp_account_price WHERE account_id = %s "
                "ORDER BY distributor_id, distributor_sku, min_quantity, account_price_id",
                (account_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.rollback()
    return rows
