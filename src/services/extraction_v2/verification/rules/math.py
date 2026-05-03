"""Math verification rules: cross-field arithmetic checks."""
from __future__ import annotations

from decimal import Decimal
from typing import Any

from src.services.extraction_v2.verification.network import Rule


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except Exception:
        return None


def _close(a: Decimal | None, b: Decimal | None, tol: Decimal = Decimal("0.05")) -> bool:
    """True if |a - b| <= tol. False if either is None."""
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


# Subtotal + Tax = Total
INVOICE_TOTAL_MATH = Rule(
    name="invoice_total_math",
    fields=("invoice_amount", "tax_amount", "invoice_total_incl_tax"),
    check=lambda v: _close(
        _to_decimal(v.get("invoice_amount", 0)) + _to_decimal(v.get("tax_amount", 0)),
        _to_decimal(v.get("invoice_total_incl_tax")),
    ),
    on_fail="demote",
    why="invoice_amount + tax_amount must equal invoice_total_incl_tax (±0.05)",
)

PO_TOTAL_MATH = Rule(
    name="po_total_math",
    fields=("total_amount", "tax_amount", "total_amount_incl_tax"),
    check=lambda v: _close(
        _to_decimal(v.get("total_amount", 0)) + _to_decimal(v.get("tax_amount", 0)),
        _to_decimal(v.get("total_amount_incl_tax")),
    ),
    on_fail="demote",
    why="total_amount + tax_amount must equal total_amount_incl_tax (±0.05)",
)

QUOTE_TOTAL_MATH = Rule(
    name="quote_total_math",
    fields=("total_amount", "tax_amount", "total_amount_incl_tax"),
    check=lambda v: _close(
        _to_decimal(v.get("total_amount", 0)) + _to_decimal(v.get("tax_amount", 0)),
        _to_decimal(v.get("total_amount_incl_tax")),
    ),
    on_fail="demote",
    why="total_amount + tax_amount must equal total_amount_incl_tax (±0.05)",
)

# Tax should never equal subtotal (extractor confused tax with amount)
TAX_NOT_EQUAL_SUBTOTAL_INVOICE = Rule(
    name="tax_not_equal_invoice_amount",
    fields=("invoice_amount", "tax_amount"),
    check=lambda v: _to_decimal(v.get("invoice_amount", 0)) is None
                    or _to_decimal(v.get("tax_amount", 0)) is None
                    or abs(_to_decimal(v["invoice_amount"]) - _to_decimal(v["tax_amount"])) > Decimal("0.5"),
    on_fail="abstain",  # this is the "tax = subtotal" classic bug — eject both
    why="tax_amount must NOT equal invoice_amount (extractor confusion)",
)

# Tax percent should be 0-30%
TAX_PERCENT_PLAUSIBLE = Rule(
    name="tax_percent_plausible",
    fields=("tax_percent",),
    check=lambda v: _to_decimal(v.get("tax_percent")) is None
                    or (Decimal("0") <= _to_decimal(v["tax_percent"]) <= Decimal("30")),
    on_fail="abstain",
    why="tax_percent must be in [0, 30]",
)


ALL_MATH_RULES = [
    INVOICE_TOTAL_MATH, PO_TOTAL_MATH, QUOTE_TOTAL_MATH,
    TAX_NOT_EQUAL_SUBTOTAL_INVOICE,
    TAX_PERCENT_PLAUSIBLE,
]
