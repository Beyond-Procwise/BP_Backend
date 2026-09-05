"""Duplicate-invoice detection, registered.

Delegates to ``src.services.duplicate_invoice_detector``, which scores candidate
pairs through the linking engine's ``invoice_duplicate`` profile.
"""
from __future__ import annotations

from datetime import date

from src.services import duplicate_invoice_detector as _did

from ..contract import DATE, RATIO, RECORD, SCORE_100, TEXT, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "assurance"
_FROM = date(2026, 9, 5)

_D1 = {"invoice_id": "I1", "invoice_ref": "INV-2025-0455", "supplier_name": "Acme Ltd",
       "supplier_id": "SUP-1", "total_amount": 4500.00, "currency": "GBP",
       "invoice_date": date(2025, 6, 1), "po_id": "PO-9", "line_items": []}
_D2 = dict(_D1, invoice_id="I2", invoice_ref="INV-2025-0456", invoice_date=date(2025, 6, 3))
_D3 = dict(_D1, invoice_id="I3", invoice_ref="XYZ-777", total_amount=99.0,
           invoice_date=date(2024, 1, 1), supplier_name="Other plc", supplier_id="SUP-2")


@formula(
    "duplicate_invoice.pair_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="How likely two invoices are the same invoice billed twice",
    effective_from=_FROM,
    inputs=[
        Term("earlier", RECORD, "the earlier invoice"),
        Term("later", RECORD, "the later invoice"),
    ],
    output=Output("dict", SCORE_100, "F plus the full signal breakdown and band"),
    notes=(
        "Consecutive reference numbers score CONFLICT on ref_prox, not OK: "
        "INV-0455 and INV-0456 are ordinary sequential invoices, and treating "
        "near-identical references as evidence of duplication would flag every "
        "supplier who invoices twice in a week."
    ),
    golden=[
        GoldenVector(inputs={"earlier": _D1, "later": _D2},
                     expected={"F": 42.2127, "decision": "block_or_exception",
                               "F_cap": 0.6},
                     note="same supplier, same amount, consecutive refs 2 days apart"),
        GoldenVector(inputs={"earlier": _D1, "later": _D3},
                     expected={"F": 0.0393, "decision": "block_or_exception"},
                     note="unrelated invoices"),
    ],
)
def duplicate_pair_score(earlier: dict, later: dict) -> dict:
    return _did.score_pair(earlier, later)


@formula(
    "duplicate_invoice.reference_proximity",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether two invoice references are the same reference",
    effective_from=_FROM,
    inputs=[Term("a", TEXT, required=False), Term("b", TEXT, required=False)],
    output=Output("tuple", RATIO, "(score 0-1, status)"),
    golden=[
        GoldenVector(inputs={"a": "INV-2025-0455", "b": "INV-2025-0455"},
                     expected=(1.0, "OK")),
        GoldenVector(inputs={"a": "INV-2025-0455", "b": "INV-2025-0456"},
                     expected=(0.0, "CONFLICT"),
                     note="one edit apart is a DIFFERENT invoice, deliberately"),
        GoldenVector(inputs={"a": "INV-2025-0455", "b": "ZZZ-1"}, expected=(0.5, "MISSING")),
    ],
)
def reference_proximity(a=None, b=None):
    return _did.cmp_ref_prox(a, b)


@formula(
    "duplicate_invoice.date_proximity",
    version="1.0.0",
    owner=_OWNER,
    purpose="How close two invoice dates are, on a graded ladder",
    effective_from=_FROM,
    inputs=[Term("a", DATE, required=False), Term("b", DATE, required=False)],
    output=Output("tuple", RATIO, "(score 0-1, status)"),
    golden=[
        GoldenVector(inputs={"a": date(2025, 6, 1), "b": date(2025, 6, 1)},
                     expected=(1.0, "OK")),
        GoldenVector(inputs={"a": date(2025, 6, 1), "b": date(2025, 6, 3)},
                     expected=(0.85, "OK")),
        GoldenVector(inputs={"a": date(2025, 6, 1), "b": date(2024, 1, 1)},
                     expected=(0.0, "CONFLICT")),
    ],
)
def date_proximity(a=None, b=None):
    return _did.cmp_date_prox(a, b)
