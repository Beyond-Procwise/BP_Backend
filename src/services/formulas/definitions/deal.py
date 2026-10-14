"""Deal-level metrics, registered --- and two duplicate pairs consolidated.

``deal.realised_savings`` names one quantity that was previously computed twice
under two names: ``efficiency_score`` in the deal analysis service and
``savings`` in the negotiate dashboard.

``deal.pct_change`` registers the deal-analysis percentage change.
``opportunity_dashboard._pct_change`` was expected to be a second copy of it and
turned out not to be: it divides by ``abs(prev)``, rounds to whole percent and
returns a formatted string. It is a display helper with different semantics, so
it is deliberately left where it is rather than consolidated.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Optional, Sequence

from src.services.deal_analysis_service import (
    _pct_change as _pct,
    _weighted_unit_price as _wup,
)
from src.services.requirement_service import evaluate_completeness as _req_complete

from ..contract import COUNT, MONEY, PERCENT, RATIO, ROWS, Output, Term
from ..registry import GoldenVector, formula

_OWNER = "analytics"
_FROM = date(2026, 9, 5)


@formula(
    "deal.weighted_unit_price",
    version="1.0.0",
    owner=_OWNER,
    purpose="Volume-weighted unit price across a document type's line items",
    effective_from=_FROM,
    inputs=[Term("documents", ROWS, "documents each carrying `line_items`")],
    output=Output("float", MONEY, "total line value / total quantity; None if unknowable"),
    notes=(
        "Returns None, not 0.0, when no line carries both a quantity and a unit price. "
        "Services deals are lump-sum and carry no quantity at all, so None is the "
        "normal outcome for them rather than an error."
    ),
    golden=[
        GoldenVector(
            inputs={"documents": [{"line_items": [{"quantity": 10, "unit_price": 100.0},
                                                  {"quantity": 5, "unit_price": 200.0}]}]},
            expected=133.33333333333334,
            note="(10x100 + 5x200) / 15 -- unrounded by design",
        ),
        GoldenVector(inputs={"documents": []}, expected=None),
    ],
)
def weighted_unit_price(documents: Sequence[Mapping[str, Any]]):
    return _wup(list(documents or []))


@formula(
    "deal.pct_change",
    version="1.0.0",
    owner=_OWNER,
    purpose="Percentage change of a value against a baseline",
    effective_from=_FROM,
    replaces=["deal_analysis_service._pct_change"],
    inputs=[
        Term("new", MONEY, "the current value", required=False),
        Term("base", MONEY, "the baseline", required=False),
    ],
    output=Output("float", PERCENT, "percentage points, 2dp; None against a zero baseline"),
    notes=(
        "None against a zero or absent baseline: there is no percentage change "
        "from nothing, and 0.0 would read as 'measured, and it did not move'.\n\n"
        "`opportunity_dashboard._pct_change` LOOKS like a duplicate of this and is "
        "NOT one: it divides by `abs(prev)` (so it disagrees on sign for a negative "
        "baseline), rounds to whole percent, returns a formatted string, and maps a "
        "zero baseline to '+0%' or '+100%' rather than to no answer. It is a display "
        "helper, not this calculation, and consolidating them would change what the "
        "dashboard shows. Left alone deliberately."
    ),
    golden=[
        GoldenVector(inputs={"new": 110.0, "base": 100.0}, expected=10.0),
        GoldenVector(inputs={"new": 90.0, "base": 100.0}, expected=-10.0),
        GoldenVector(inputs={"new": 90.0, "base": 0.0}, expected=None),
        GoldenVector(inputs={"new": 90.0, "base": None}, expected=None),
    ],
)
def pct_change(new=None, base=None):
    return _pct(new, base)


@formula(
    "deal.realised_savings",
    version="1.0.0",
    owner=_OWNER,
    purpose="Money actually saved: the quote-to-invoice unit price gap times invoiced volume",
    effective_from=_FROM,
    replaces=["deal_analysis_service._compute.efficiency_score",
              "negotiate_dashboard.savings"],
    inputs=[
        Term("quoted_unit_price", MONEY, "weighted unit price on the quote",
             minimum=0.0, required=False),
        Term("invoiced_unit_price", MONEY, "weighted unit price actually invoiced",
             minimum=0.0, required=False),
        Term("invoiced_volume", COUNT, "quantity actually invoiced",
             minimum=0.0, required=False),
    ],
    output=Output("float", MONEY, "positive = saved against the quote; None if unknowable"),
    notes=(
        "One name for a quantity previously computed twice: `efficiency_score` in the "
        "deal analysis service and `savings` on the negotiate dashboard. The dashboard "
        "additionally reported 0.0 when there was no quote to compare against, which "
        "reads as 'we saved nothing' rather than 'there is no baseline'; this returns "
        "None, and callers that want the old display string ask for it explicitly."
    ),
    golden=[
        GoldenVector(
            inputs={"quoted_unit_price": 110.0, "invoiced_unit_price": 100.0,
                    "invoiced_volume": 250.0},
            expected=2500.0,
        ),
        GoldenVector(
            inputs={"quoted_unit_price": 100.0, "invoiced_unit_price": 110.0,
                    "invoiced_volume": 250.0},
            expected=-2500.0, note="an overrun is a negative saving, not a zero",
        ),
        GoldenVector(
            inputs={"quoted_unit_price": None, "invoiced_unit_price": 100.0,
                    "invoiced_volume": 250.0},
            expected=None, note="no quote is no baseline, so no savings claim",
        ),
    ],
)
def realised_savings(quoted_unit_price=None, invoiced_unit_price=None,
                     invoiced_volume=None):
    if quoted_unit_price is None or invoiced_unit_price is None or invoiced_volume is None:
        return None
    return round((float(quoted_unit_price) - float(invoiced_unit_price))
                 * float(invoiced_volume), 2)


@formula(
    "requirement.completeness_score",
    version="1.0.0",
    owner=_OWNER,
    purpose="Proportion of a requirement's required fields that are genuinely filled",
    effective_from=_FROM,
    inputs=[
        Term("requirement", ROWS, "the requirement record", required=False),
        Term("required_fields", ROWS, "field names that must be present", required=False),
    ],
    output=Output("tuple", RATIO, "(0-1 score, list of missing field names)"),
    notes="A blank string counts as missing. No fabrication: only genuinely-filled fields score.",
    golden=[
        GoldenVector(
            inputs={"requirement": {"category": "IT", "quantity": None, "budget": "",
                                    "need_by": "2025-09-01"},
                    "required_fields": ["category", "quantity", "budget", "need_by"]},
            expected=(0.5, ["quantity", "budget"]),
        ),
        GoldenVector(inputs={"requirement": {"a": 1, "b": 2},
                             "required_fields": ["a", "b"]}, expected=(1.0, [])),
    ],
)
def requirement_completeness(requirement=None, required_fields=None):
    return _req_complete(dict(requirement or {}), list(required_fields or []))
