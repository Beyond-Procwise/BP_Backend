"""Win probability from closed outcomes, or no win probability at all (spec §4.5).

Measured per opportunity_type as won / (won + lost). Withdrawn is excluded: we
withdrew, the customer decided nothing. Below the governed minimum of closed
outcomes a type gets NOTHING -- an uncalibrated 0.5 in the column would be
indistinguishable from a measured one.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, List, Mapping, Optional, Tuple

from src.services.governed_limits import limit as _governed_limit
from src.services.sell_side._db import dict_cursor
from src.services.sell_side.money import q4


def _MIN_CLOSED() -> int:
    return _governed_limit("reseller_catalog", "calibration_min_closed", cast=int)


@dataclass(frozen=True)
class Calibration:
    opportunity_type: str
    won: int
    lost: int
    rate: Optional[Decimal]
    applied: bool


def rates(counts: Mapping[str, Tuple[int, int]], min_closed: int) -> List[Calibration]:
    out = []
    for kind in sorted(counts):
        won, lost = counts[kind]
        closed = won + lost
        if closed >= min_closed and closed > 0:
            out.append(Calibration(kind, won, lost, q4(Decimal(won) / Decimal(closed)), True))
        else:
            out.append(Calibration(kind, won, lost, None, False))
    return out


def calibrate(conn: Any) -> List[Calibration]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT opportunity_type, count(*) FILTER (WHERE outcome = 'won') AS won, "
        "count(*) FILTER (WHERE outcome = 'lost') AS lost "
        "FROM proc.bp_sales_opportunity GROUP BY opportunity_type")
    counts = {r["opportunity_type"]: (r["won"], r["lost"]) for r in cur.fetchall()}
    result = rates(counts, _MIN_CLOSED())
    for c in result:
        if c.applied:
            cur.execute(
                "UPDATE proc.bp_sales_opportunity SET win_probability = %s, "
                "win_probability_basis = 'calibrated', last_modified_date = now() "
                "WHERE opportunity_type = %s AND outcome = 'open' "
                "AND (win_probability_basis IS NULL OR win_probability_basis = 'calibrated')",
                (c.rate, c.opportunity_type))
    conn.commit()
    return result
