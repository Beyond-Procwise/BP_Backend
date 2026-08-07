"""Deterministic unit-of-measure normalisation.

Maps the units this corpus actually uses onto a canonical name and a dimension,
and refuses everything else. Of the 28 distinct unit_of_measure values across
the three _trgt line tables, fourteen are units and fourteen are payment terms,
scope descriptions or prices that landed in the UoM column:

    '30 days from quote date'  'annual in advance'  'included'
    'transition 7 weeks'       'onboarding 10 weeks'
    'implementation (one-off, fixed) - £72,000.00'

Coercing any of those into a unit would be fabrication, so they yield
UOM_UNMAPPED and the raw string carries forward un-normalised.

Pure functions, no I/O, no database.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Tuple

UOM_UNMAPPED = "UOM_UNMAPPED"

# Months and years have no fixed length in days. We state a convention rather
# than pretending the ambiguity is not there, and we stamp this code onto any
# result that depends on it so a downstream comparison can see the assumption.
CALENDAR_CONVENTION = "CALENDAR_CONVENTION_30D_365D"

_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class UomResult:
    """The outcome of normalising one UoM string.

    ``canonical is None`` if and only if ``UOM_UNMAPPED in reason_codes``.
    ``factor`` is the unit expressed in days, and only time units carry one;
    count, mass and length units have no common basis to convert to here.
    """

    canonical: Optional[str]
    dimension: Optional[str]
    factor: Optional[Decimal]
    reason_codes: Tuple[str, ...]


_HOUR_IN_DAYS = Decimal("1") / Decimal("24")

# key -> (canonical, dimension, factor-in-days)
_CANONICAL: Dict[str, Tuple[str, str, Optional[Decimal]]] = {
    # count
    "each": ("each", "count", None),
    "case": ("case", "count", None),
    "pack": ("pack", "count", None),
    "box": ("box", "count", None),
    "seat": ("seat", "count", None),
    "licence": ("licence", "count", None),
    "shipment": ("shipment", "count", None),
    # count -- observed in the canonical product master (proc.bp_product_master)
    "set": ("set", "count", None),
    "sheet": ("sheet", "count", None),
    "roll": ("roll", "count", None),
    "pen": ("pen", "count", None),
    "module": ("module", "count", None),
    # time
    "hour": ("hour", "time", _HOUR_IN_DAYS),
    "day": ("day", "time", Decimal("1")),
    "week": ("week", "time", Decimal("7")),
    "month": ("month", "time", Decimal("30")),
    "quarter": ("quarter", "time", Decimal("90")),
    "year": ("year", "time", Decimal("365")),
    # mass
    "tonne": ("tonne", "mass", None),
    # length
    "metre": ("metre", "length", None),
}

_ALIASES: Dict[str, str] = {
    # count
    "ea": "each",
    "eaches": "each",
    "unit": "each",
    "units": "each",
    "cases": "case",
    "cs": "case",
    "packs": "pack",
    "pk": "pack",
    "boxes": "box",
    "seats": "seat",
    "licences": "licence",
    "license": "licence",
    "licenses": "licence",
    "lic": "licence",
    "shipments": "shipment",
    # time
    "hr": "hour",
    "hrs": "hour",
    "hours": "hour",
    "days": "day",
    "dy": "day",
    "weeks": "week",
    "wk": "week",
    "wks": "week",
    "mo": "month",
    "mth": "month",
    "mths": "month",
    "months": "month",
    "yr": "year",
    "yrs": "year",
    "years": "year",
    "annum": "year",
    # 'Monthly' lowercases to 'monthly', which is an adverbial spelling of the
    # unit rather than the unit itself -- a casing pass alone does not catch it.
    "monthly": "month",
    "weekly": "week",
    "daily": "day",
    "hourly": "hour",
    "quarterly": "quarter",
    "qtr": "quarter",
    "quarters": "quarter",
    # plurals of the units observed in the canonical product master
    "sets": "set",
    "sheets": "sheet",
    "rolls": "roll",
    "pens": "pen",
    "modules": "module",
    "per annum": "year",
    # mass
    "t": "tonne",
    "mt": "tonne",
    "tonnes": "tonne",
    "tonnes(metric)": "tonne",
    "metric tonne": "tonne",
    "ton": "tonne",
    "tons": "tonne",
    # length
    "m": "metre",
    "mtr": "metre",
    "mtrs": "metre",
    "metres": "metre",
    "meter": "metre",
    "meters": "metre",
}

_TIME_CONVENTION_UNITS = {"month", "quarter", "year"}


def _key(raw: str) -> str:
    """Lowercase, collapse internal whitespace, strip a trailing full stop."""
    k = _WHITESPACE.sub(" ", raw).strip().lower()
    return k[:-1] if k.endswith(".") else k


def normalise_uom(raw: Optional[str]) -> UomResult:
    """Map ``raw`` onto a canonical unit, or refuse it.

    Matching is EXACT on the normalised key. Never substring-match. This is the
    single most important rule in this module: 'transition 7 weeks' contains
    'week', 'implementation (one-off, fixed) - £72,000.00' contains 'on', and a
    substring match would silently turn a scope description or a payment term
    into a time unit. A wrong unit is worse than no unit, because a wrong one
    makes an incomparable pair of numbers look comparable.

    An absent or blank UoM is UNMAPPED, never defaulted to 'each'. Guessing
    here would bake the guess into the fact base, where nothing downstream can
    tell it apart from a unit the document actually stated.
    """
    if raw is None or not isinstance(raw, str):
        return UomResult(None, None, None, (UOM_UNMAPPED,))

    key = _key(raw)
    if not key:
        return UomResult(None, None, None, (UOM_UNMAPPED,))

    key = _ALIASES.get(key, key)

    entry = _CANONICAL.get(key)
    if entry is None:
        return UomResult(None, None, None, (UOM_UNMAPPED,))

    canonical, dimension, factor = entry
    codes: Tuple[str, ...] = ()
    if canonical in _TIME_CONVENTION_UNITS:
        codes = (CALENDAR_CONVENTION,)

    return UomResult(canonical, dimension, factor, codes)
