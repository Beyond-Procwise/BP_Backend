"""The single place a figure becomes text.

DISPLAY ONLY. Nothing here may be parsed back into a number, summed, sorted or
compared — those operate on the raw value. This mirrors the rule already
carried at the top of ``src/lib/format/currency.js`` in the UI repo, and for
the same reason: the SpendIQ engine once summed rendered strings.

Why a second formatter exists at all
------------------------------------
The client has formatted money since July (``formatCompactCurrency``), and it
keeps doing so for every figure Procurement Home and the SpendIQ dashboard draw
themselves. Rendering the analytic answer on the server adds a second
formatter, and the only tolerable relationship between two formatters in one
product is agreement: a supplier reading £1.1M in the answer must not read
£1,100,000 on the tile beside it. So this is a deliberate port, case for case,
and ``tests/services/analytics/test_formatting.py`` is the parity contract.

The one intentional divergence is the rounding rule. JavaScript's ``toFixed``
rounds off the binary double, so its behaviour on an exact half depends on how
the literal happened to be stored. Here the value goes through ``Decimal`` via
its string form and rounds half up, which is what a reader expects and what the
rest of this codebase does with money (see ``services/benchmark`` and its
``excel_round``). For every mantissa either side actually produces, the two
agree.
"""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Optional

# Rendered in place of an amount we do not have. A value we do not have is not a
# zero, and must never be shown as one.
EMPTY_AMOUNT = "—"

# Ordered largest-first: the first threshold the value clears is its tier.
_TIERS: tuple[tuple[int, str], ...] = (
    (10**12, "T"),
    (10**9, "B"),
    (10**6, "M"),
    (10**3, "K"),
)

_SYMBOLS = {
    "GBP": "£",
    "USD": "$",
    "EUR": "€",
    "JPY": "¥",
    "NZD": "NZ$",
    "AUD": "A$",
}


def _as_decimal(value: Any) -> Optional[Decimal]:
    """The value as an exact decimal, or None if it is not a real number.

    Catches None, NaN and both infinities in one place. Conversion goes through
    ``str`` so that 23.45 is the decimal 23.45 and not the double 23.4499…,
    which is what makes half-up rounding predictable.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def _quantize(value: Decimal, decimals: int) -> Decimal:
    exponent = Decimal(1).scaleb(-decimals)
    return value.quantize(exponent, rounding=ROUND_HALF_UP)


def _group(value: Decimal) -> str:
    """Whole-number grouping, en-GB — which is to say, commas."""
    return f"{int(value):,}"


def currency_symbol(currency: Optional[str]) -> str:
    """The prefix for a currency, spelled out when we have no symbol for it.

    An empty or absent currency means "we do not know what this amount is in",
    and the number is rendered bare. Stamping a £ on a ₹ figure would state a
    different number, not tidy the same one.
    """
    if not currency:
        return ""
    code = str(currency).upper()
    return _SYMBOLS.get(code, f"{code} ")


def format_money(amount: Any, currency: Optional[str] = "GBP", decimals: int = 1) -> str:
    """A monetary amount, compact.

    Below 1,000 the value renders in full with no suffix and no pence (£639).
    At or above 1,000 it scales to K/M/B/T so the integer part stays one to
    three digits (£15.0K, £1.1M).
    """
    value = _as_decimal(amount)
    if value is None:
        return EMPTY_AMOUNT

    symbol = currency_symbol(currency)
    sign = "-" if value < 0 else ""
    magnitude = abs(value)

    if magnitude < 1000:
        return f"{sign}{symbol}{_group(_quantize(magnitude, 0))}"

    tier_index = next(i for i, (threshold, _) in enumerate(_TIERS) if magnitude >= threshold)
    threshold, suffix = _TIERS[tier_index]
    mantissa = _quantize(magnitude / Decimal(threshold), decimals)

    # Rounding can carry the mantissa to 1000 — 999,950 / 1e3 is 999.95, which
    # renders "1000.0K" and four integer digits. Promote it a tier instead.
    if mantissa >= 1000 and tier_index > 0:
        threshold, suffix = _TIERS[tier_index - 1]
        mantissa = _quantize(magnitude / Decimal(threshold), decimals)

    return f"{sign}{symbol}{mantissa:.{decimals}f}{suffix}"


def format_pct(value: Any, decimals: int = 1) -> str:
    """A share, already expressed in percentage points (23.4 -> "23.4%")."""
    parsed = _as_decimal(value)
    if parsed is None:
        return EMPTY_AMOUNT
    return f"{_quantize(parsed, decimals):.{decimals}f}%"


def format_int(value: Any) -> str:
    """A count. Always grouped, never abbreviated — 2,005 suppliers is a fact."""
    parsed = _as_decimal(value)
    if parsed is None:
        return EMPTY_AMOUNT
    return _group(_quantize(parsed, 0))


def format_delta(value: Any, decimals: int = 1) -> str:
    """A period-on-period change in percentage points, always signed.

    The sign is explicit in both directions, including zero: "+0.0%" reads as a
    measured non-move, where a bare "0.0%" reads as a missing measurement.
    """
    parsed = _as_decimal(value)
    if parsed is None:
        return EMPTY_AMOUNT
    rounded = _quantize(parsed, decimals)
    sign = "-" if rounded < 0 else "+"
    return f"{sign}{abs(rounded):.{decimals}f}%"
