"""What one catalog item costs us at one quantity.

A volume break changes the margin exactly at the quantities a large quote turns
on, so the flat cost_price is only the answer when no break applies. The tier
used is returned, so the margin can be re-derived by hand later (spec §4.2).
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime as dt
from decimal import Decimal
from typing import Optional, Sequence, Tuple

from src.services.sell_side import special_bids
from src.services.sell_side._db import NotFound


@dataclass(frozen=True)
class CostAt:
    catalog_item_id: int
    distributor_id: str
    distributor_sku: str
    mpn: Optional[str]
    item_description: str
    unit_of_measure: Optional[str]
    currency: str
    list_price: Optional[Decimal]
    unit_cost: Optional[Decimal]
    cost_tier_applied: Optional[Decimal]
    is_current: bool
    cost_source: str = "catalog"          # catalog | special_bid
    special_bid_id: Optional[int] = None


def pick_tier(tiers: Sequence[Tuple[Decimal, Decimal]],
              quantity: Decimal) -> Optional[Tuple[Decimal, Decimal]]:
    """The (min_quantity, cost) break with the highest floor at or below quantity."""
    eligible = [t for t in tiers if t[0] <= quantity]
    return max(eligible, key=lambda t: t[0]) if eligible else None


def cost_at(cur, catalog_item_id: int, quantity: Decimal, *, account_id: Optional[str] = None,
            on_date: Optional[dt.date] = None) -> CostAt:
    """``cur`` must be a RealDictCursor. With an ``account_id``, a special bid registered
    for that customer and applying at this quantity replaces the catalog cost (and says so)."""
    cur.execute(
        "SELECT catalog_item_id, distributor_id, distributor_sku, mpn, item_description, "
        "unit_of_measure, currency, list_price, cost_price, valid_to IS NULL AS is_current "
        "FROM proc.bp_catalog_item WHERE catalog_item_id = %s",
        (catalog_item_id,))
    item = cur.fetchone()
    if item is None:
        raise NotFound(f"catalog item {catalog_item_id} does not exist")
    cur.execute(
        "SELECT min_quantity, cost_price, currency FROM proc.bp_catalog_cost_tier "
        "WHERE catalog_item_id = %s ORDER BY min_quantity", (catalog_item_id,))
    tiers = cur.fetchall() or []
    for t in tiers:
        if t["currency"] != item["currency"]:
            raise ValueError(
                f"catalog item {catalog_item_id} is priced in {item['currency']} but a "
                f"cost tier is in {t['currency']}; refusing to mix them")
    tier = pick_tier([(t["min_quantity"], t["cost_price"]) for t in tiers], quantity)
    bid = None
    if account_id is not None:
        bid = special_bids.resolve(
            cur, account_id=account_id, distributor_id=item["distributor_id"],
            distributor_sku=item["distributor_sku"], currency=item["currency"].strip(),
            quantity=quantity, on_date=on_date)
    return CostAt(
        catalog_item_id=item["catalog_item_id"], distributor_id=item["distributor_id"],
        distributor_sku=item["distributor_sku"], mpn=item["mpn"],
        item_description=item["item_description"],
        unit_of_measure=item["unit_of_measure"], currency=item["currency"].strip(),
        list_price=item["list_price"],
        unit_cost=(bid.cost_price if bid else tier[1] if tier else item["cost_price"]),
        cost_tier_applied=None if bid else (tier[0] if tier else None),
        is_current=bool(item["is_current"]),
        cost_source="special_bid" if bid else "catalog",
        special_bid_id=bid.special_bid_id if bid else None,
    )
