"""Price harmonisation for one identical item bought at different prices (SC-016): pure, deterministic.

The reference price is a percentile of the prices paid (linear interpolation between the two nearest
prices, the usual spreadsheet method), not the lowest price. The saving on each purchase is what was
paid above the reference, times the quantity; a purchase at or below the reference saves nothing.
It does not normalise Incoterms, payment terms or order size: the prices must already be like for like.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Sequence

METHOD = "percentile by linear interpolation; saving = max(0, price - reference) x quantity"


class HarmonisationError(ValueError):
    """The inputs cannot be harmonised; the message says what to fix."""


def _positive(value: Any, what: str) -> Decimal:
    if isinstance(value, bool) or value is None:
        raise HarmonisationError(f"{what} is not a number")
    try:
        out = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise HarmonisationError(f"{what} is not a number") from None
    if not out.is_finite() or out <= 0:
        raise HarmonisationError(f"{what} must be greater than zero")
    return out


def _percentile(sorted_prices: List[Decimal], percentile: Decimal) -> Decimal:
    position = (Decimal(len(sorted_prices) - 1)) * percentile / Decimal(100)
    low = int(position)
    high = min(low + 1, len(sorted_prices) - 1)
    return sorted_prices[low] + (sorted_prices[high] - sorted_prices[low]) * (position - low)


def harmonise(lines: Sequence[Mapping[str, Any]], percentile: Any, min_observations: int = 3) -> Dict[str, Any]:
    if not lines:
        raise HarmonisationError("no purchase lines were given")
    try:
        pct = Decimal(str(percentile))
    except (InvalidOperation, ValueError, TypeError):
        raise HarmonisationError("percentile is not a number") from None
    if not pct.is_finite() or not (0 <= pct <= 100):
        raise HarmonisationError("percentile must be between 0 and 100")
    rows = []
    for i, line in enumerate(lines):
        if not isinstance(line, Mapping):
            raise HarmonisationError(f"line {i + 1} is not a record with id, price and quantity")
        name = str(line.get("id") or f"line {i + 1}")
        rows.append((name, _positive(line.get("price"), f"{name} price"),
                     _positive(line.get("quantity"), f"{name} quantity")))
    if len(rows) < int(min_observations):
        raise HarmonisationError(f"{len(rows)} prices given, at least {int(min_observations)} are needed for a "
                                 f"reference price")
    reference = _percentile(sorted(p for _, p, _ in rows), pct)
    out_lines: List[Dict[str, Any]] = []
    total = Decimal(0)
    for name, price, qty in rows:
        saving = max(Decimal(0), price - reference) * qty
        total += saving
        out_lines.append({"id": name, "price": float(price), "quantity": float(qty),
                          "reference_price": float(reference), "saving": float(saving)})
    return {"reference_price": float(reference), "percentile": float(pct), "lines": out_lines,
            "total_saving": float(total), "lowest_price": float(min(p for _, p, _ in rows)),
            "highest_price": float(max(p for _, p, _ in rows)), "method": METHOD,
            "assumptions": ["prices are already on the same Incoterm, payment terms, currency and unit of measure",
                            "the lines are the same item"]}
