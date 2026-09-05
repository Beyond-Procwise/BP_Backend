"""Document relationship maths, registered.

Every function here delegates to ``src.services.linking_engine``. Nothing is
reimplemented: a second copy of the maths would be a second thing to keep
right, which is the problem this registry exists to solve. The registry adds
the contract, the version, the audit record and the vectors; the arithmetic
stays where it is.

Golden vectors were snapshotted from the live implementations on 2026-09-05 by
``scripts/formulas/snapshot_goldens.py``. They are today's numbers, not the
numbers anyone believes the code produces.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Optional

from src.services import linking_engine as _le

# Importing these registers the ``quote_rival`` and ``invoice_duplicate``
# profiles on the engine's PROFILES table. Without them, a caller naming those
# profiles would get a KeyError from inside the formula body rather than a
# contract refusal, so the import is load-bearing rather than incidental.
from src.services import requirement_similarity as _rs  # noqa: F401
from src.services import duplicate_invoice_detector as _did  # noqa: F401

from ..contract import (
    COUNT, DATE, FACTOR, LABEL, MONEY, RATIO, RECORD, ROWS, SCORE_100, TEXT,
    Output, Term,
)
from ..registry import GoldenVector, formula

_OWNER = "linkage"
_FROM = date(2026, 9, 5)

_INV = {
    "invoice_id": "INV-9001", "po_id": "PO-4001", "supplier_id": "SUP-100",
    "converted_amount_usd": 12500.00, "currency": "GBP",
    "invoice_date": "2025-04-10", "country": "GB", "region": "London",
}
_PO = {
    "po_id": "PO-4001", "supplier_id": "SUP-100",
    "converted_amount_usd": 12500.00, "currency": "GBP",
    "order_date": "2025-03-01", "ship_to_country": "GB", "delivery_region": "London",
}
_LINES = [
    {"item_description": "Dell Latitude 5540 laptop", "quantity": 10, "unit_price": 780.0},
    {"item_description": "Docking station USB-C", "quantity": 10, "unit_price": 95.0},
]


@formula(
    "linking.relationship_confidence",
    version="1.0.0",
    owner=_OWNER,
    purpose="How confidently two procurement documents are the same relationship (F, 0-100)",
    effective_from=_FROM,
    inputs=[
        Term("source_row", RECORD, "the child document (invoice / quote)"),
        Term("target_row", RECORD, "the parent document (purchase order)"),
        Term("profile_name", LABEL, "which registered signal profile to score under"),
        Term("source_lines", ROWS, "child line items", required=False),
        Term("target_lines", ROWS, "parent line items", required=False),
        Term("set_amount_usd", MONEY,
             "aggregate amount of the whole sibling set, for N:1 scoring",
             minimum=0.0, required=False),
    ],
    output=Output("dict", SCORE_100,
                  "F plus the full per-signal breakdown, band decision and stage values"),
    notes=(
        "S (separation) and Q (evidence quality) are pinned to 1.0 in the engine. "
        "See docs/formula-registry-gap-report.md D-6: S=1.0 is sound for the 1:1 "
        "promotion path and is NOT sound for `deal_clustering.awarded_po`, which "
        "scores one bid against every PO and takes the best."
    ),
    golden=[
        GoldenVector(
            inputs=dict(source_row=_INV, target_row=_PO, profile_name="invoice_po",
                        source_lines=_LINES, target_lines=_LINES, set_amount_usd=None),
            expected={"F": 95.6194, "decision": "auto_link", "P_raw": 0.956194,
                      "C": 1.0, "F_cap": 1.0, "S": 1.0, "Q": 1.0},
            note="invoice matching its own PO on every signal",
        ),
        GoldenVector(
            inputs=dict(
                source_row=dict(_INV, converted_amount_usd=19000.00, supplier_id="SUP-200"),
                target_row=_PO, profile_name="invoice_po",
                source_lines=_LINES, target_lines=_LINES, set_amount_usd=None),
            expected={"F": 19.0491, "decision": "block_or_exception", "F_cap": 0.45},
            note="tier-1 supplier conflict imposes the 0.45 cap",
        ),
    ],
)
def relationship_confidence(
    source_row: dict,
    target_row: dict,
    profile_name: str,
    source_lines: Optional[list] = None,
    target_lines: Optional[list] = None,
    set_amount_usd: Optional[float] = None,
) -> dict:
    return _le.score_link(
        source_row, target_row, profile_name, source_lines, target_lines, set_amount_usd
    )


@formula(
    "linking.cluster_dampening",
    version="1.0.0",
    owner=_OWNER,
    purpose="Correlated-signal dampening factor within one evidence cluster",
    effective_from=_FROM,
    inputs=[Term("n_active", COUNT, "signals in this cluster that were observable",
                 minimum=0, maximum=64)],
    output=Output("float", FACTOR, "multiplier applied to the cluster's summed contribution"),
    golden=[
        GoldenVector(inputs={"n_active": 1}, expected=1.0),
        GoldenVector(inputs={"n_active": 2}, expected=0.85),
        GoldenVector(inputs={"n_active": 5}, expected=0.70,
                     note="flat above 2 -- not a continuous decay"),
    ],
)
def cluster_dampening(n_active: int) -> float:
    return _le._dampen(int(n_active))


@formula(
    "linking.line_pair_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="Similarity of two individual line items (description / qty / unit price)",
    effective_from=_FROM,
    inputs=[
        Term("a", RECORD, "one line item"),
        Term("b", RECORD, "the other line item"),
    ],
    output=Output("float", RATIO, "0-1 weighted over only the sub-signals present on both"),
    golden=[
        GoldenVector(inputs={"a": _LINES[0], "b": _LINES[0]}, expected=1.0),
        GoldenVector(
            inputs={"a": {"item_description": "blue widget large"},
                    "b": {"item_description": "blue widget"}},
            expected=0.6666666666666666,
            note="description only; missing qty/price are neutral, not penalised",
        ),
    ],
)
def line_pair_score(a: dict, b: dict) -> float:
    return _le._line_pair_score(a, b)


@formula(
    "linking.line_set_composite",
    version="1.0.0",
    owner=_OWNER,
    purpose="Best-match line-set agreement with a source-extra coverage penalty",
    effective_from=_FROM,
    inputs=[
        Term("source_lines", ROWS, "child line items", required=False),
        Term("target_lines", ROWS, "parent line items", required=False),
    ],
    output=Output("tuple", RATIO, "(score 0-1, status OK|WEAK|CONFLICT|MISSING)"),
    notes="A source covering a SUBSET of the target is not penalised (split shipment).",
    golden=[
        GoldenVector(inputs={"source_lines": _LINES, "target_lines": _LINES},
                     expected=(1.0, "OK")),
        GoldenVector(
            inputs={"source_lines": _LINES + [{"item_description": "carriage",
                                               "quantity": 1, "unit_price": 40.0}],
                    "target_lines": _LINES},
            expected=(0.4444444444444445, "CONFLICT"),
            note="one extra source line costs a third of the coverage",
        ),
        GoldenVector(inputs={"source_lines": [], "target_lines": _LINES},
                     expected=(0.5, "MISSING"),
                     note="no lines is neutral, not a conflict"),
    ],
)
def line_set_composite(source_lines=None, target_lines=None):
    return _le.cmp_line_composite(source_lines or [], target_lines or [])


@formula(
    "linking.amount_agreement",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether two amounts agree, decaying linearly to zero at 10% drift",
    effective_from=_FROM,
    inputs=[
        Term("a", MONEY, "first amount", required=False),
        Term("b", MONEY, "second amount", required=False),
        Term("tol", RATIO, "fractional drift treated as exact agreement",
             minimum=0.0, maximum=0.09, required=False),
    ],
    output=Output("tuple", RATIO, "(score 0-1, status)"),
    notes=(
        "One of FOUR amount-tolerance conventions in this codebase "
        "(gap report D-7). Consolidating them changes numbers and is therefore "
        "a separate, deliberate decision."
    ),
    golden=[
        GoldenVector(inputs={"a": 12500.0, "b": 12500.0, "tol": None}, expected=(1.0, "OK")),
        GoldenVector(inputs={"a": 12875.0, "b": 12500.0, "tol": None},
                     expected=(0.7874865156418555, "WEAK"), note="3% drift"),
        GoldenVector(inputs={"a": 15000.0, "b": 12500.0, "tol": None},
                     expected=(0.0, "CONFLICT"), note="20% drift is past the decay floor"),
        GoldenVector(inputs={"a": None, "b": 12500.0, "tol": None},
                     expected=(0.5, "MISSING"),
                     note="absent evidence scores neutral and contributes q=0 upstream"),
    ],
)
def amount_agreement(a=None, b=None, tol=None):
    if tol is None:
        return _le.cmp_numeric_tol(a, b)
    return _le.cmp_numeric_tol(a, b, tol)


@formula(
    "linking.temporal_plausibility",
    version="1.0.0",
    owner=_OWNER,
    purpose="Whether a child document's date is plausible against its parent's order date",
    effective_from=_FROM,
    inputs=[
        Term("doc_date", DATE, "the child document's date", required=False),
        Term("parent_order_date", DATE, "the parent PO's order date", required=False),
        Term("parent_due_date", DATE, "parent expiry, when one genuinely exists",
             required=False),
    ],
    output=Output("tuple", RATIO, "(score 0-1, status)"),
    golden=[
        GoldenVector(inputs={"doc_date": "2025-04-10", "parent_order_date": "2025-03-01",
                             "parent_due_date": None}, expected=(1.0, "OK")),
        GoldenVector(inputs={"doc_date": "2025-02-01", "parent_order_date": "2025-03-01",
                             "parent_due_date": None}, expected=(0.0, "CONFLICT"),
                     note="child cannot predate its parent"),
        GoldenVector(inputs={"doc_date": "2027-06-01", "parent_order_date": "2025-03-01",
                             "parent_due_date": None}, expected=(0.0, "CONFLICT"),
                     note="beyond the 730-day plausibility window"),
        GoldenVector(inputs={"doc_date": None, "parent_order_date": "2025-03-01",
                             "parent_due_date": None}, expected=(0.5, "MISSING")),
    ],
)
def temporal_plausibility(doc_date=None, parent_order_date=None, parent_due_date=None):
    return _le.cmp_temporal(doc_date, parent_order_date, parent_due_date)


@formula(
    "linking.location_agreement",
    version="1.0.0",
    owner=_OWNER,
    purpose="Country and region agreement between two documents",
    effective_from=_FROM,
    inputs=[
        Term("a_country", TEXT, required=False),
        Term("a_region", TEXT, required=False),
        Term("b_country", TEXT, required=False),
        Term("b_region", TEXT, required=False),
    ],
    output=Output("tuple", RATIO, "(score 0-1, status)"),
    golden=[
        GoldenVector(inputs={"a_country": "GB", "a_region": "London",
                             "b_country": "GB", "b_region": "London"}, expected=(1.0, "OK")),
        GoldenVector(inputs={"a_country": "GB", "a_region": "London",
                             "b_country": "DE", "b_region": "Berlin"},
                     expected=(0.0, "CONFLICT")),
    ],
)
def location_agreement(a_country=None, a_region=None, b_country=None, b_region=None):
    return _le.cmp_location(a_country, a_region, b_country, b_region)


@formula(
    "linking.decision_band",
    version="1.0.0",
    owner=_OWNER,
    purpose="Which action band a relationship confidence F falls into",
    effective_from=_FROM,
    inputs=[Term("F", SCORE_100, "relationship confidence", minimum=0.0, maximum=100.0)],
    output=Output("str", LABEL,
                  "auto_link | auto_link_with_warning | review | weak_relation "
                  "| block_or_exception"),
    notes="Cuts are 92 / 80 / 65 / 45, env-overridable nowhere -- they are module constants.",
    golden=[
        GoldenVector(inputs={"F": 95.0}, expected="auto_link"),
        GoldenVector(inputs={"F": 85.0}, expected="auto_link_with_warning"),
        GoldenVector(inputs={"F": 70.0}, expected="review"),
        GoldenVector(inputs={"F": 50.0}, expected="weak_relation"),
        GoldenVector(inputs={"F": 10.0}, expected="block_or_exception"),
    ],
)
def decision_band(F: float) -> str:
    return _le._band(float(F))
