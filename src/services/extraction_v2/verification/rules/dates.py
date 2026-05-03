"""Date verification rules: ordering and range checks."""
from __future__ import annotations

from datetime import date, timedelta

from src.services.extraction_v2.verification.network import Rule


INVOICE_DATE_BEFORE_DUE = Rule(
    name="invoice_date_before_due",
    fields=("invoice_date", "due_date"),
    check=lambda v: v["invoice_date"] <= v["due_date"],
    on_fail="demote",
    why="invoice_date must be ≤ due_date",
)

ORDER_DATE_BEFORE_DELIVERY = Rule(
    name="order_date_before_delivery",
    fields=("order_date", "expected_delivery_date"),
    check=lambda v: v["order_date"] <= v["expected_delivery_date"],
    on_fail="demote",
    why="order_date must be ≤ expected_delivery_date",
)

QUOTE_DATE_BEFORE_VALIDITY = Rule(
    name="quote_date_before_validity",
    fields=("quote_date", "validity_date"),
    check=lambda v: v["quote_date"] <= v["validity_date"],
    on_fail="demote",
    why="quote_date must be ≤ validity_date",
)


ALL_DATE_RULES = [
    INVOICE_DATE_BEFORE_DUE,
    ORDER_DATE_BEFORE_DELIVERY,
    QUOTE_DATE_BEFORE_VALIDITY,
]
