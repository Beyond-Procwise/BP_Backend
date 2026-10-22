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


@formula(
    "deal.share_pct",
    version="1.0.0",
    owner=_OWNER,
    purpose="One part as a percentage of a whole (a supplier's share of spend, a tail's share of invoices)",
    effective_from=date(2026, 10, 22),
    inputs=[
        Term("part", MONEY, "the part, in the same unit as the whole", required=False),
        Term("whole", MONEY, "the whole", required=False),
    ],
    output=Output("float", PERCENT, "percentage points, 2dp; None against a zero or absent whole"),
    notes=(
        "Registered so a share is a calculation the verifier can trace rather than arithmetic the "
        "model does in its head (live, 2026-10-22: a tail review divided two cube figures and then "
        "struck the result out as unverifiable). None against a zero or absent whole: a share of "
        "nothing is not 0 percent."
    ),
    golden=[
        GoldenVector(inputs={"part": 2522861.12, "whole": 17640482.17}, expected=14.3, tolerance=0.01),
        GoldenVector(inputs={"part": 25.0, "whole": 100.0}, expected=25.0),
        GoldenVector(inputs={"part": 0.0, "whole": 100.0}, expected=0.0),
        GoldenVector(inputs={"part": 5.0, "whole": 0.0}, expected=None),
        GoldenVector(inputs={"part": None, "whole": 100.0}, expected=None),
    ],
)
def share_pct(part=None, whole=None):
    try:
        p, w = float(part), float(whole)
    except (TypeError, ValueError):
        return None
    if w == 0:
        return None
    return round(p / w * 100.0, 2)


@formula(
    "deal.scaled_total",
    version="1.0.0",
    owner=_OWNER,
    purpose="A unit amount scaled by a count (a monthly value over twelve months, a unit price over a quantity)",
    effective_from=date(2026, 10, 22),
    inputs=[
        Term("unit_value", MONEY, "the amount per unit or per period", required=False),
        Term("multiplier", COUNT, "how many units or periods", required=False, minimum=0.0),
    ],
    output=Output("float", MONEY, "unit_value times multiplier, 2dp; None if either is absent"),
    notes=(
        "Registered so a projected or annual total is a traceable calculation rather than a figure "
        "the model multiplies in its head (live, 2026-10-22: 4,900 a month became 58,800 a year and "
        "the verifier struck it out). The multiplier is an assumption: it must come from the "
        "request, a result or the skill's settings, never be chosen by the model."
    ),
    golden=[
        GoldenVector(inputs={"unit_value": 4900.0, "multiplier": 12}, expected=58800.0),
        GoldenVector(inputs={"unit_value": 19.99, "multiplier": 3}, expected=59.97),
        GoldenVector(inputs={"unit_value": 100.0, "multiplier": 0}, expected=0.0),
        GoldenVector(inputs={"unit_value": None, "multiplier": 12}, expected=None),
    ],
)
def scaled_total(unit_value=None, multiplier=None):
    try:
        v, m = float(unit_value), float(multiplier)
    except (TypeError, ValueError):
        return None
    return round(v * m, 2)
