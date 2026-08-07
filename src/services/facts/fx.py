"""FX resolution for commercial facts.

A fact stamps its own rate, the date of the rate it used, and the source of
that rate. Nothing downstream looks FX up again: if a report re-renders
tomorrow against a refreshed rate table, yesterday's numbers must not move.

KNOWN LIMITATION (F4). ``proc.bp_fx_rates`` is a snapshot table, not a dated
series. Its columns are ``(base_currency, currency, rate, fetched_at)``, and
``fetched_at`` records when we fetched the rate — not the date the rate was
effective. It therefore cannot answer "what was GBP/USD on 2025-04-01", and
``rate_date`` here is the snapshot timestamp, NOT the transaction date. That is
reproducibility, which the brief requires, and it is not historical accuracy,
which the brief does not get from this table. A dated rate corpus is separate
work and is deliberately out of scope for this phase.

Measured on bp_sqldb 2026-08-07: 1,162 rows, 7 distinct ``fetched_at`` values
(2026-07-16 .. 2026-07-28), and exactly one ``base_currency`` — USD. Two
consequences drive the implementation below:

  * The same pair appears once per snapshot, so a query with no ordering
    returns an arbitrary row. Every lookup here orders by ``fetched_at DESC``
    and takes one row, so resolution is deterministic.
  * Because the only base is USD, a pair like EUR->GBP has no direct row and
    must be crossed through the table's base currency.

Deterministic, no LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional, Tuple

FX_UNAVAILABLE = "FX_UNAVAILABLE"

_TABLE = "proc.bp_fx_rates"

# Ordering is load-bearing, not cosmetic: the table holds one row per pair per
# snapshot, so without it the driver may hand back any of the seven.
_DIRECT_SQL = f"""
    SELECT rate, fetched_at, base_currency
      FROM {_TABLE}
     WHERE upper(base_currency) = %s AND upper(currency) = %s
     ORDER BY fetched_at DESC
     LIMIT 1
"""

_LEG_SQL = f"""
    SELECT rate, fetched_at, base_currency
      FROM {_TABLE}
     WHERE upper(currency) = %s
     ORDER BY fetched_at DESC
     LIMIT 1
"""


@dataclass(frozen=True)
class FxResult:
    """A resolved rate, or an explicit refusal.

    ``rate`` converts an amount in ``from_ccy`` into ``to_ccy`` by
    multiplication. ``rate is None`` if and only if ``FX_UNAVAILABLE`` is in
    ``reason_codes`` — there is no silent 1.0 fallback, because a guessed
    parity is indistinguishable downstream from a real one.
    """

    rate: Optional[Decimal]
    rate_date: Optional[datetime]
    source: Optional[str]
    reason_codes: Tuple[str, ...]


_UNAVAILABLE = FxResult(None, None, None, (FX_UNAVAILABLE,))


def _ccy(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    code = value.strip().upper()
    return code or None


def _dec(value: Any) -> Optional[Decimal]:
    """Coerce to Decimal without going through float."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _source(fetched_at: Optional[datetime]) -> str:
    if fetched_at is None:
        return _TABLE.split(".")[-1]
    stamp = fetched_at.isoformat().replace("+00:00", "Z")
    return f"bp_fx_rates@{stamp}"


def _fetch(cur, sql: str, params) -> Optional[tuple]:
    try:
        cur.execute(sql, params)
        return cur.fetchone()
    except Exception:
        # A resolver that raises would take down fact assembly for an entire
        # document over one missing rate. Fail closed instead.
        return None


def resolve_fx(cur, from_ccy: str, to_ccy: str) -> FxResult:
    """Resolve ``from_ccy`` -> ``to_ccy`` against the rate snapshot.

    Returns rate 1 for a same-currency pair without touching the database.
    Otherwise tries the direct row, then the inverse row, then a cross-rate
    through the table's base currency. Anything unresolved is FX_UNAVAILABLE
    with a NULL rate — never a guessed parity.
    """
    src = _ccy(from_ccy)
    dst = _ccy(to_ccy)
    if src is None or dst is None:
        return _UNAVAILABLE

    if src == dst:
        return FxResult(Decimal("1"), None, "identity", ())

    # 1. Direct row: base = from, currency = to.
    row = _fetch(cur, _DIRECT_SQL, (src, dst))
    if row:
        rate, fetched_at, _base = row[0], row[1], row[2]
        rate = _dec(rate)
        if rate is not None and rate != 0:
            return FxResult(rate, fetched_at, _source(fetched_at), ())

    # 2. Inverse row: base = to, currency = from.
    row = _fetch(cur, _DIRECT_SQL, (dst, src))
    if row:
        rate, fetched_at, _base = row[0], row[1], row[2]
        rate = _dec(rate)
        if rate is not None and rate != 0:
            return FxResult(Decimal("1") / rate, fetched_at, _source(fetched_at), ())

    # 3. Cross-rate through the table's own base currency. Both legs must come
    #    from the same base, or the division is meaningless.
    leg_from = _fetch(cur, _LEG_SQL, (src,))
    leg_to = _fetch(cur, _LEG_SQL, (dst,))
    if leg_from and leg_to:
        r_from, t_from, b_from = _dec(leg_from[0]), leg_from[1], _ccy(leg_from[2])
        r_to, t_to, b_to = _dec(leg_to[0]), leg_to[1], _ccy(leg_to[2])
        if r_from and r_to and b_from and b_from == b_to:
            # base -> src is r_from, base -> dst is r_to, so src -> dst is r_to / r_from.
            stamp = min(t for t in (t_from, t_to) if t is not None) if (t_from or t_to) else None
            return FxResult(r_to / r_from, stamp, _source(stamp), ())

    return _UNAVAILABLE
