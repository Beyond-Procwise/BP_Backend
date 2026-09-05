"""Extraction quality maths, registered.

Completeness, reader accuracy, amount agreement and the line-arithmetic role
check. Delegates to ``src.services.extraction`` and ``src.services.facts``.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Any, Mapping, Optional, Sequence

from src.services.extraction.promotion import (
    _compute_confidence_score as _completeness,
)
from src.services.extraction.three_way_match import _agrees
from src.services.facts.arithmetic import check_line_arithmetic

from ..contract import (
    COUNT, LABEL, MONEY, RATIO, RECORD, ROWS, SCORE_100, Output, Term,
)
from ..registry import GoldenVector, formula

_OWNER = "extraction"
_FROM = date(2026, 9, 5)


@formula(
    "extraction.completeness_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="How much of a document's schema the extraction actually filled (0-100)",
    effective_from=_FROM,
    inputs=[
        Term("doc_type", LABEL, "invoice | quote | purchase_order | contract"),
        Term("row", RECORD, "the staged row"),
        Term("required", ROWS, "field names that are required for this doc type"),
    ],
    output=Output("Decimal", SCORE_100, "0-100; None when the doc type has no field list"),
    notes=(
        "**This measures COMPLETENESS, not correctness, and its name has caused real "
        "harm.** Required fields score 2, optional 1, as a percentage of the schema, so "
        "a row with every required field and no optionals lands at ~50%. Using it as an "
        "evidence-quality proxy in the linking engine made the promotion gate "
        "mathematically unreachable by a perfect match (F = 75.5 against a gate of 80): "
        "no invoice could promote and no deal could form. A document can be entirely "
        "complete and entirely wrong -- `extraction.accuracy_score` is the other question."
    ),
    golden=[
        GoldenVector(
            inputs={"doc_type": "invoice",
                    "row": {"invoice_id": "I1", "supplier_name": "Acme",
                            "total_amount": 100.0},
                    "required": ["invoice_id", "supplier_name", "total_amount"]},
            expected=Decimal("13.33"),
            note=(
                "three of fourteen invoice fields filled. Note how LOW a correctly "
                "extracted three-field row scores -- this is why the figure must not "
                "be read as quality"
            ),
        ),
        GoldenVector(
            inputs={"doc_type": "invoice", "row": {},
                    "required": ["invoice_id", "supplier_name", "total_amount"]},
            expected=Decimal("0.00"),
        ),
        GoldenVector(
            inputs={"doc_type": "nonesuch",
                    "row": {"invoice_id": "I1"}, "required": ["invoice_id"]},
            expected=None,
            note=(
                "an unrecognised doc type yields None, not 0 -- 'we have no schema for "
                "this' is not 'this document is empty'"
            ),
        ),
    ],
)
def completeness_score(doc_type: str, row: Mapping[str, Any], required: Sequence[str]):
    return _completeness(doc_type, dict(row), set(required or ()))


@formula(
    "extraction.amount_agreement",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether two money amounts agree within the three-way-match tolerance",
    effective_from=_FROM,
    inputs=[
        Term("a", MONEY, "first amount"),
        Term("b", MONEY, "second amount"),
    ],
    output=Output("bool", MONEY, "True when within 1p absolute or 0.5% relative"),
    notes=(
        "Tolerances `_ABS_TOL = 0.01` and `_REL_TOL = 0.005`. One of FOUR "
        "amount-tolerance conventions in this codebase (gap report D-7): the linking "
        "engine decays linearly to zero at 10% drift, extraction completeness allows "
        "5% or GBP 1.00, and reconciliation allows 1% or GBP 1.00. The same two "
        "numbers can agree in one module and conflict in another."
    ),
    golden=[
        GoldenVector(inputs={"a": 100.00, "b": 100.00}, expected=True),
        GoldenVector(inputs={"a": 100.00, "b": 100.005}, expected=True,
                     note="half a penny is rounding, not a discrepancy"),
        GoldenVector(inputs={"a": 100.00, "b": 105.00}, expected=False),
    ],
)
def amount_agreement(a: float, b: float) -> bool:
    return bool(_agrees(float(a), float(b)))


@formula(
    "extraction.line_arithmetic_state",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether quantity x unit rate = extended line corroborates the assigned roles",
    effective_from=_FROM,
    inputs=[
        Term("quantity", COUNT, "line quantity", required=False),
        Term("unit_rate", MONEY, "price per unit", required=False),
        Term("extended_line", MONEY, "line total", required=False),
    ],
    output=Output("ArithmeticState", LABEL,
                  "consistent | inconsistent | untestable_quantity_one "
                  "| untestable_missing_input"),
    notes=(
        "Two untestable states exist because abstaining is not the same as passing. "
        "Roughly a fifth of real lines have quantity = 1, where a unit rate and a total "
        "are numerically identical and no arithmetic can separate them -- exactly the "
        "blind spot that let a line total ship booked as a unit price. A fact whose "
        "role could not be verified must not look identical to one that was checked."
    ),
    golden=[
        GoldenVector(inputs={"quantity": 10, "unit_rate": 5.0, "extended_line": 50.0},
                     expected="consistent"),
        GoldenVector(inputs={"quantity": 10, "unit_rate": 5.0, "extended_line": 90.0},
                     expected="inconsistent"),
        GoldenVector(inputs={"quantity": 1, "unit_rate": 5.0, "extended_line": 5.0},
                     expected="untestable_quantity_one",
                     note="the qty=1 blind spot, named rather than passed"),
        GoldenVector(inputs={"quantity": None, "unit_rate": 5.0, "extended_line": 50.0},
                     expected="untestable_missing_input"),
    ],
)
def line_arithmetic_state(quantity=None, unit_rate=None, extended_line=None) -> str:
    return check_line_arithmetic(quantity, unit_rate, extended_line).value
