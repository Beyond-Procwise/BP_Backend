"""Quote arithmetic. Pure, Decimal-only, and NULL-honest: an unknown cost is an
unknown margin, never a zero one."""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional, Sequence

_TWO = Decimal("0.01")
_FOUR = Decimal("0.0001")
_ISO = re.compile(r"[A-Z]{3}")


def q2(x: Decimal) -> Decimal:
    return x.quantize(_TWO, rounding=ROUND_HALF_UP)


def q4(x: Decimal) -> Decimal:
    return x.quantize(_FOUR, rounding=ROUND_HALF_UP)


def iso_currency(value: Optional[str]) -> str:
    code = (value or "").strip().upper()
    if not _ISO.fullmatch(code):
        raise ValueError(f"{value!r} is not a three-letter ISO currency code")
    return code


@dataclass(frozen=True)
class LinePrice:
    line_total: Decimal
    line_cost: Optional[Decimal]
    line_margin: Optional[Decimal]
    line_margin_pct: Optional[Decimal]
    discount_pct: Optional[Decimal]


@dataclass(frozen=True)
class QuoteTotals:
    total_ex_tax: Decimal
    total_cost: Optional[Decimal]
    total_margin: Optional[Decimal]
    margin_pct: Optional[Decimal]


def price_line(quantity: Decimal, unit_price: Decimal,
               unit_cost: Optional[Decimal], list_price: Optional[Decimal]) -> LinePrice:
    line_total = q2(quantity * unit_price)
    line_cost = q2(quantity * unit_cost) if unit_cost is not None else None
    line_margin = q2(line_total - line_cost) if line_cost is not None else None
    line_margin_pct = (q4(line_margin / line_total)
                       if line_margin is not None and line_total != 0 else None)
    discount_pct = (q4(Decimal(1) - unit_price / list_price)
                    if list_price is not None and list_price > 0 else None)
    return LinePrice(line_total, line_cost, line_margin, line_margin_pct, discount_pct)


def total_quote(lines: Sequence[LinePrice]) -> QuoteTotals:
    total_ex_tax = q2(sum((l.line_total for l in lines), Decimal(0)))
    if any(l.line_cost is None for l in lines):
        return QuoteTotals(total_ex_tax, None, None, None)
    total_cost = q2(sum((l.line_cost for l in lines), Decimal(0)))
    total_margin = q2(total_ex_tax - total_cost)
    margin_pct = q4(total_margin / total_ex_tax) if total_ex_tax != 0 else None
    return QuoteTotals(total_ex_tax, total_cost, total_margin, margin_pct)
