"""Back-populate the structured columns on proc.bp_opportunity.

What can honestly be harvested out of calculation_details is narrow, and the
point of this script is to be explicit about that rather than to make the new
columns look full. Measured on bp_testdb (bp_sqldb holds zero opportunity
rows):

    Duplicate Invoice Recovery  300 rows  -> currency, amount_native
    Price Benchmark Variance      2 rows  -> quantity, unit_price
    Invoice Overbilling           6 rows  -> nothing structured

Invoice Overbilling carries po_total, quote_total and invoice_total: three
different documents' totals. None of them is "the" amount of the finding, and
choosing one would be a guess presented as a harvest. Those rows are marked
INDETERMINATE with every new column left NULL.

The harvest is driven by KEY, not by detector name. A detector renamed
tomorrow keeps working; a detector that starts emitting a currency starts
having it harvested. Keying on the detector's display string would silently
stop working the day someone edits it.

Usage:
    set -a && . ./.env && set +a
    PGDATABASE=bp_testdb ./venv/bin/python scripts/backfill_opportunity_structured.py [--apply]

Without --apply it reports what it would write and changes nothing.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.facts.uom import UOM_UNMAPPED, normalise_uom  # noqa: E402

logger = logging.getLogger("backfill_opportunity_structured")

RESOLVED = "RESOLVED"
INDETERMINATE = "INDETERMINATE"

#: A value was present but could not be read as a number. Recorded rather than
#: dropped, so the row is not mistaken for one that simply had no value.
UNPARSEABLE = "VALUE_UNPARSEABLE"

#: calculation_details key -> structured column.
#:
#: amount_gbp is deliberately absent: it is already converted, and writing it
#: into amount_native would restate a converted figure as the document's own.
#: benchmark_price is absent for the same reason -- it is the comparator, not
#: what was actually paid.
_AMOUNT_KEYS = ("amount_native",)
_UNIT_PRICE_KEYS = ("actual_price", "unit_price")
_QUANTITY_KEYS = ("quantity",)
_CURRENCY_KEYS = ("currency",)
_UOM_KEYS = ("uom", "unit_of_measure")


@dataclass
class Harvest:
    """What one row's calculation_details actually yielded."""

    currency: Optional[str] = None
    amount_native: Optional[Decimal] = None
    unit_price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    uom: Optional[str] = None
    uom_normalised: Optional[str] = None
    fx_rate: Optional[Decimal] = None
    fx_rate_date: Optional[Any] = None
    value_basis: Optional[str] = None
    reason_codes: List[str] = field(default_factory=list)
    facts_state: str = INDETERMINATE


def _first(payload: Dict[str, Any], keys) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _dec(value: Any, label: str, reasons: List[str]) -> Optional[Decimal]:
    """Coerce to Decimal, or refuse and say so. Never silently zero."""
    if value is None:
        return None
    if isinstance(value, bool):  # bools are ints in Python; never a money value
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        reasons.append(f"{UNPARSEABLE}:{label}")
        return None


def harvest(calculation_details: Optional[Dict[str, Any]]) -> Harvest:
    """Read the structured values out of one calculation_details payload.

    Returns INDETERMINATE with every column NULL when nothing structured is
    present. INDETERMINATE is not a synonym for zero: a finding whose amount
    could not be resolved must not read as a finding worth nothing.
    """
    out = Harvest()
    if not isinstance(calculation_details, dict) or not calculation_details:
        return out

    reasons: List[str] = []

    currency = _first(calculation_details, _CURRENCY_KEYS)
    if isinstance(currency, str) and currency.strip():
        out.currency = currency.strip()

    out.amount_native = _dec(_first(calculation_details, _AMOUNT_KEYS),
                             "amount_native", reasons)
    out.unit_price = _dec(_first(calculation_details, _UNIT_PRICE_KEYS),
                          "unit_price", reasons)
    out.quantity = _dec(_first(calculation_details, _QUANTITY_KEYS),
                        "quantity", reasons)

    raw_uom = _first(calculation_details, _UOM_KEYS)
    if isinstance(raw_uom, str) and raw_uom.strip():
        out.uom = raw_uom
        result = normalise_uom(raw_uom)
        out.uom_normalised = result.canonical
        reasons.extend(result.reason_codes)

    out.reason_codes = reasons

    resolved = any(v is not None for v in (
        out.currency, out.amount_native, out.unit_price, out.quantity, out.uom))
    out.facts_state = RESOLVED if resolved else INDETERMINATE

    # A row whose only signal failed to parse has resolved nothing.
    if out.facts_state == RESOLVED and out.currency is None and all(
        v is None for v in (out.amount_native, out.unit_price, out.quantity, out.uom)
    ):
        out.facts_state = INDETERMINATE

    return out


_SELECT = """
    SELECT opportunity_id, detector_type, calculation_details
      FROM proc.bp_opportunity
"""

_UPDATE = """
    UPDATE proc.bp_opportunity
       SET currency = %s, amount_native = %s, unit_price = %s, quantity = %s,
           uom = %s, uom_normalised = %s, fx_rate = %s, fx_rate_date = %s,
           value_basis = %s, reason_codes = %s, facts_state = %s
     WHERE opportunity_id = %s
"""


def run(apply: bool = False) -> Dict[str, int]:
    from src.services.db import get_conn

    counts: Dict[str, int] = {
        "rows": 0, RESOLVED: 0, INDETERMINATE: 0,
        "currency": 0, "amount_native": 0, "unit_price": 0, "quantity": 0,
    }

    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        cur.execute("select current_database()")
        logger.info("database: %s (apply=%s)", cur.fetchone()[0], apply)

        cur.execute(_SELECT)
        rows = cur.fetchall()

        for opportunity_id, detector_type, payload in rows:
            counts["rows"] += 1
            h = harvest(payload)
            counts[h.facts_state] += 1
            for key in ("currency", "amount_native", "unit_price", "quantity"):
                if getattr(h, key) is not None:
                    counts[key] += 1

            if apply:
                cur.execute(_UPDATE, (
                    h.currency, h.amount_native, h.unit_price, h.quantity,
                    h.uom, h.uom_normalised, h.fx_rate, h.fx_rate_date,
                    h.value_basis, h.reason_codes or None, h.facts_state,
                    opportunity_id,
                ))

        if apply:
            conn.commit()
        else:
            conn.rollback()

    return counts


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="write the harvest; without it, report only")
    args = parser.parse_args()

    counts = run(apply=args.apply)
    print(f"rows            : {counts['rows']}")
    print(f"RESOLVED        : {counts[RESOLVED]}")
    print(f"INDETERMINATE   : {counts[INDETERMINATE]}")
    print(f"  currency      : {counts['currency']}")
    print(f"  amount_native : {counts['amount_native']}")
    print(f"  unit_price    : {counts['unit_price']}")
    print(f"  quantity      : {counts['quantity']}")
    if not args.apply:
        print("\n(dry run -- nothing written; pass --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
