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

import logging
import re
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

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


# ---------------------------------------------------------------------------
# Runtime vocabulary
#
# proc.bp_uom_canonical is the source of truth; the maps above are a seed that
# a test asserts matches it. Loading at runtime means a unit can be added
# without a deploy.
#
# Three properties this must have, each of which is a way it could go wrong:
#
#   * A failed or EMPTY load must never blank the vocabulary. A normaliser that
#     recognises nothing marks every unit UOM_UNMAPPED, and those absences get
#     written into the fact base as though the documents had stated nothing. A
#     slightly stale map is enormously preferable to a confident wrong silence.
#   * No query in the per-value path. normalise_uom is called once per line;
#     a lookup per call is the N+1 pattern that already cost this codebase an
#     information_schema query per document.
#   * Only status='active' loads, filtered in SQL. A 'proposed' unit is an
#     observation awaiting a human; if it resolved, confirming would be moot.
# ---------------------------------------------------------------------------

_ACTIVE_SQL = """
    SELECT uom_code, dimension, aliases, factor_days, factor_convention
      FROM proc.bp_uom_canonical
     WHERE status = 'active'
"""

_DEFAULT_TTL_SECONDS = 300.0


@dataclass(frozen=True)
class Vocabulary:
    """A resolved unit vocabulary and where it came from."""

    canonical: Mapping[str, Tuple[str, str, Optional[Decimal]]]
    aliases: Mapping[str, str]
    convention_units: FrozenSet[str]
    source: str


SEED_VOCABULARY = Vocabulary(
    canonical=dict(_CANONICAL),
    aliases=dict(_ALIASES),
    convention_units=frozenset(_TIME_CONVENTION_UNITS),
    source="builtin-seed",
)

_lock = threading.Lock()
_active: Vocabulary = SEED_VOCABULARY
_loaded_at: Optional[float] = None


def build_vocabulary(rows: Iterable[Mapping[str, Any]], *, source: str) -> Vocabulary:
    """Assemble a Vocabulary from bp_uom_canonical rows. Pure."""
    canonical: Dict[str, Tuple[str, str, Optional[Decimal]]] = {}
    aliases: Dict[str, str] = {}
    convention: set[str] = set()

    for row in rows:
        code = (row.get("uom_code") or "").strip().lower()
        if not code:
            continue
        dimension = row.get("dimension")
        factor_raw = row.get("factor_days")
        factor = None
        if factor_raw is not None:
            try:
                # str() first: a float from the driver must not carry binary
                # rounding into a Decimal.
                factor = Decimal(str(factor_raw))
            except Exception:
                factor = None
        canonical[code] = (code, dimension, factor)
        if row.get("factor_convention"):
            convention.add(code)
        for alias in row.get("aliases") or ():
            key = (alias or "").strip().lower()
            if key and key != code:
                aliases[key] = code

    return Vocabulary(canonical, aliases, frozenset(convention), source)


def active_vocabulary() -> Vocabulary:
    """The vocabulary currently in force."""
    return _active


def reset_vocabulary() -> None:
    """Drop back to the built-in seed. For tests and for a forced re-read."""
    global _active, _loaded_at
    with _lock:
        _active = SEED_VOCABULARY
        _loaded_at = None


def ensure_vocabulary(cur, *, ttl_seconds: float = _DEFAULT_TTL_SECONDS) -> Vocabulary:
    """Load the vocabulary from the database if the cached copy is stale.

    Takes a cursor rather than opening a connection so it can reuse the one the
    caller already has, and so this module still performs no I/O of its own.
    Cheap to call repeatedly: within the TTL it does nothing at all.

    Never raises. A failure leaves whatever vocabulary is already in force.
    """
    global _active, _loaded_at

    now = time.monotonic()
    if _loaded_at is not None and (now - _loaded_at) < ttl_seconds:
        return _active

    try:
        cur.execute(_ACTIVE_SQL)
        columns = [d[0] for d in (cur.description or [])]
        rows = [dict(zip(columns, r)) for r in (cur.fetchall() or [])]
    except Exception:
        logger.warning(
            "could not read proc.bp_uom_canonical; continuing with the %s "
            "vocabulary (%d units)", _active.source, len(_active.canonical),
            exc_info=True,
        )
        return _active

    vocabulary = build_vocabulary(rows, source=f"bp_uom_canonical@{len(rows)}units")

    # Check the BUILT vocabulary, not the raw row count. Rows that parse to
    # nothing usable — a changed column list, a cursor answering a different
    # query — are non-empty yet yield no units, and accepting that would
    # install an empty vocabulary that marks the entire corpus UOM_UNMAPPED
    # in silence. Treated as a failed load, NOT as "there are no units".
    if not vocabulary.canonical:
        logger.warning(
            "proc.bp_uom_canonical yielded no usable units from %d row(s); "
            "keeping the %s vocabulary rather than recognising nothing",
            len(rows), _active.source,
        )
        return _active
    with _lock:
        _active = vocabulary
        _loaded_at = now
    logger.info("loaded %d active units from proc.bp_uom_canonical", len(rows))
    return vocabulary


def normalise_uom(raw: Optional[str], *, vocabulary: Optional[Vocabulary] = None) -> UomResult:
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
    vocab = vocabulary if vocabulary is not None else _active

    if raw is None or not isinstance(raw, str):
        return UomResult(None, None, None, (UOM_UNMAPPED,))

    key = _key(raw)
    if not key:
        return UomResult(None, None, None, (UOM_UNMAPPED,))

    key = vocab.aliases.get(key, key)

    entry = vocab.canonical.get(key)
    if entry is None:
        return UomResult(None, None, None, (UOM_UNMAPPED,))

    canonical, dimension, factor = entry
    codes: Tuple[str, ...] = ()
    if canonical in vocab.convention_units:
        codes = (CALENDAR_CONVENTION,)

    return UomResult(canonical, dimension, factor, codes)
