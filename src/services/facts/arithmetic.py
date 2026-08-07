"""The role-consistency check.

``quantity x unit_rate = extended_line`` is what TESTS a role assignment rather
than trusting the label on the column. The schemas already name ``unit_price``
apart from ``line_total``, and a real bug in this codebase still booked a line
total as a unit price — correct names, wrong values. Names could not catch it;
arithmetic can.

Measured on bp_sqldb (genuinely extracted data, not seeded):

    quote lines           316 rows, 183 testable, 143 consistent (78.1%)
    invoice lines         152 rows,  99 testable,  92 consistent (92.9%)
    purchase order lines  171 rows, 124 testable, 114 consistent (91.9%)

Two things follow. Between 7% and 22% of real lines fail the check — at least
one of the three numbers is mis-assigned or mis-extracted. And where quantity
is 1, about a fifth of all lines, the unit rate and the total are numerically
identical and no arithmetic can separate them. That is the precise blind spot
that let the historical unit-price-as-total bug ship unnoticed, so this module
abstains there rather than reporting a pass.

Pure function, Decimal throughout, no I/O.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Optional

from src.services.facts.models import ArithmeticState

# Absolute, not proportional. A proportional tolerance scales the blind spot
# with the value: the larger the line, the more error it hides, which is
# exactly backwards. Two pence covers ordinary rounding at any magnitude.
DEFAULT_TOLERANCE = Decimal("0.02")


def _dec(value: Any) -> Optional[Decimal]:
    """Coerce to Decimal without routing through binary float arithmetic."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None


def check_line_arithmetic(
    quantity: Any,
    unit_price: Any,
    extended: Any,
    *,
    tolerance: Decimal = DEFAULT_TOLERANCE,
) -> ArithmeticState:
    """Report whether the three numbers corroborate the roles assigned to them.

    Returns an abstention rather than a verdict whenever the check cannot
    establish anything: a missing input, a zero quantity, or a quantity of
    exactly 1. The abstention is the point — a fact whose role was never
    verified must not look identical to one that was.
    """
    q = _dec(quantity)
    u = _dec(unit_price)
    e = _dec(extended)

    if q is None or u is None or e is None:
        return ArithmeticState.UNTESTABLE_MISSING_INPUT

    # Zero quantity tells us nothing: 0 x anything is 0, so the product matches
    # a zero total regardless of whether the unit rate is a rate at all.
    if q == 0:
        return ArithmeticState.UNTESTABLE_MISSING_INPUT

    # This branch MUST come before the equality comparison below. At quantity 1
    # the product equals the unit rate identically, so a qty-1 line would
    # otherwise fall through as CONSISTENT and the abstention would never
    # happen — manufacturing confidence for roughly a fifth of the corpus.
    if q == 1:
        return ArithmeticState.UNTESTABLE_QUANTITY_ONE

    if abs((q * u) - e) <= tolerance:
        return ArithmeticState.CONSISTENT
    return ArithmeticState.INCONSISTENT
