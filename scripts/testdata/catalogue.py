"""A priced catalogue mapped onto the real L5 taxonomy.

Prices drift upward over the 3.5-year window with per-item noise. Without drift,
benchmark pricing and price-variance detection would have nothing to find, and
test D1 could not distinguish a working engine from a stub.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
from typing import Sequence

from scripts.testdata.reference import TaxonomyLeaf
from scripts.testdata.rng import make_rng, weighted_apportion

EPOCH = date(2023, 1, 1)
ANNUAL_DRIFT = 0.038  # 3.8% a year, roughly consistent with the period's inflation

_QUALIFIERS = (
    "Standard", "Professional", "Enterprise", "Compact", "Heavy-Duty", "Premium",
    "Essential", "Advanced", "Modular", "Certified",
)

# A subscription is not sold by the tonne. Pairing each noun with the units it
# is plausibly bought in keeps the catalogue readable on screen, where a
# "Retainer, 4 tonne" undermines everything around it.
_GOODS = ("Assembly", "Module", "Unit", "Kit", "Component", "Bundle")
_SERVICES = ("Service Package", "Subscription", "Retainer", "Installation")
_NOUNS = (*_GOODS, *_SERVICES)

_UNITS_FOR = {
    **{noun: ("each", "box", "pack", "case", "metre", "tonne") for noun in _GOODS},
    "Service Package": ("day", "month"),
    "Subscription": ("month", "licence"),
    "Retainer": ("month", "day"),
    "Installation": ("day", "hour"),
}

# Every (qualifier, noun) pair, in a fixed order. Items within a leaf take
# distinct pairs, so a description is unique without needing a part code
# stapled to the end of it -- 60 pairs per leaf against roughly 20 items.
_NAME_PAIRS = tuple((q, n) for q in _QUALIFIERS for n in _NOUNS)

UNITS = tuple(sorted({unit for units in _UNITS_FOR.values() for unit in units}))


@dataclass(frozen=True)
class CatalogueItem:
    item_id: str
    description: str
    leaf: TaxonomyLeaf
    unit_of_measure: str
    base_price: Decimal
    currency: str
    preferred_supplier_id: str


def build_catalogue(
    seed: int, leaves: Sequence[TaxonomyLeaf], supplier_ids: Sequence[str]
) -> list[CatalogueItem]:
    """Exactly 5,000 items, spread so every leaf gets at least one."""
    rng = make_rng(seed, "catalogue")
    per_leaf = weighted_apportion([1] * len(leaves), 5000)

    items: list[CatalogueItem] = []
    counter = 0
    for leaf, count in zip(leaves, per_leaf):
        wanted = max(count, 1) if count == 0 else count
        # Distinct name pairs within the leaf, starting at a per-leaf offset so
        # two leaves do not open with the same qualifier.
        offset = rng.randrange(len(_NAME_PAIRS))
        for position in range(wanted):
            counter += 1
            qualifier, noun = _NAME_PAIRS[(offset + position) % len(_NAME_PAIRS)]
            magnitude = rng.choice([1, 1, 1, 10, 10, 100, 1000])
            base = Decimal(str(round(rng.uniform(0.8, 9.9) * magnitude, 2)))
            items.append(
                CatalogueItem(
                    item_id=f"ITM{counter:06d}",
                    description=f"{qualifier} {leaf.l5} {noun}",
                    leaf=leaf,
                    unit_of_measure=rng.choice(_UNITS_FOR[noun]),
                    base_price=base,
                    currency="GBP",
                    preferred_supplier_id=supplier_ids[counter % len(supplier_ids)],
                )
            )
    return items[:5000]


def price_on(item: CatalogueItem, when: date, *, seed: int) -> Decimal:
    """The item's price at `when`: base price, drifted, plus deterministic noise."""
    years = (when - EPOCH).days / 365.25
    drifted = float(item.base_price) * ((1.0 + ANNUAL_DRIFT) ** years)

    digest = hashlib.sha256(
        f"{seed}:{item.item_id}:{when.isoformat()}".encode("utf-8")
    ).digest()
    # Deterministic noise in [-4%, +4%], stable for a given item and date.
    noise = 1.0 + ((int.from_bytes(digest[:4], "big") / 0xFFFFFFFF) - 0.5) * 0.08

    return Decimal(str(drifted * noise)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
