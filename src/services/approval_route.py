"""Who must sign a deal, derived from its category and value.

Moved from the SpendIQ Pipeline's Approve stage (engine.js: APPROVAL_ROUTES,
APPROVAL_VALUE_RULES, APPROVAL_NONVALUE_RULES, dealApprovalRoute) so the board paper can state
the route as traced figures (ruled 2026-09-25). The rules are the screen's, word for word; the
screen still holds its own copy, and tests/services/test_approval_route.py pins this one to it.

The route is what MUST happen, not what has: no endpoint records an approval decision against a
deal, so nothing here says anyone has signed.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

#: The currency the thresholds are written in. A value in another currency is not tested
#: against them: silently re-denominating a control is the defect the screen refuses too.
APPROVAL_MATRIX_CURRENCY = "GBP"

APPROVAL_ROUTES: Dict[str, List[Tuple[str, str]]] = {
    "SaaS / IT": [("AI validation", "Automated · level 0"), ("IT approver", "Head of IT"),
                  ("Finance gate", "Budget & policy"), ("CFO sign-off", "Required > £250k")],
    "Operations": [("AI validation", "Automated"), ("Ops director", "Capex owner"),
                   ("Finance", "Budget confirmed"), ("PO desk", "Issue PO")],
    "Logistics": [("AI validation", "Automated"), ("Category buyer", "Rate owner"),
                  ("Finance gate", "Rate validation")],
    "Office & facilities": [("Auto-validate", "Rules engine"),
                            ("Category buyer", "Single approver < £50k")],
    "Platform / Enterprise": [("Qualification", "Auto-qualified"),
                              ("Deal desk", "Discount review"),
                              ("Pricing approval", "VP Sales · > 8% discount")],
    "Prof. services": [("AI validation", "Automated"), ("Legal", "SOW review"),
                       ("Finance", "Rate card")],
    "Default": [("AI validation", "Automated"), ("Approver", "Single approver")],
}

#: The value rules already written into the notes above, restated as data so they can be
#: tested. Keyed "matrix|approver" so one matrix's threshold never leaks into another's.
APPROVAL_VALUE_RULES: Dict[str, Tuple[str, Decimal]] = {
    "SaaS / IT|CFO sign-off": ("gt", Decimal(250000)),
    "Office & facilities|Category buyer": ("lt", Decimal(50000)),
}

#: Rules that are not value rules: they cannot be tested here and say so.
APPROVAL_NONVALUE_RULES: Dict[str, str] = {
    "Platform / Enterprise|Pricing approval":
        "A discount threshold, not a value threshold. The discount on this deal is not on the "
        "payload, so this approver can be neither required nor ruled out.",
}


@dataclass(frozen=True)
class RouteRow:
    who: str
    rule: str
    placement: str          # "required" | "not-required" | "untestable"
    why: str


@dataclass(frozen=True)
class Route:
    matrix: str
    matched: bool           # False when the category had no matrix and Default applied
    rows: List[RouteRow]


def _money(value: Decimal) -> str:
    return f"{APPROVAL_MATRIX_CURRENCY} {int(round(value)):,}"


def route(category: Optional[str], native: Optional[Decimal], currency: Optional[str]) -> Route:
    """The route for a deal of ``category`` worth ``native`` in ``currency``."""
    key = category if category in APPROVAL_ROUTES else "Default"
    rows: List[RouteRow] = []
    for who, rule in APPROVAL_ROUTES[key]:
        rk = f"{key}|{who}"
        if rk in APPROVAL_NONVALUE_RULES:
            rows.append(RouteRow(who, rule, "untestable", APPROVAL_NONVALUE_RULES[rk]))
            continue
        vr = APPROVAL_VALUE_RULES.get(rk)
        if vr is None:
            rows.append(RouteRow(who, rule, "required", "In the route at any value."))
            continue
        if native is None or not currency:
            rows.append(RouteRow(who, rule, "untestable",
                                 f"The threshold is {rule.lower()}, and this deal carries no "
                                 "value that can be tested against it."))
            continue
        if str(currency).upper() != APPROVAL_MATRIX_CURRENCY:
            rows.append(RouteRow(
                who, rule, "untestable",
                f"The threshold is set in {APPROVAL_MATRIX_CURRENCY} and this deal is billed in "
                f"{currency}. No {APPROVAL_MATRIX_CURRENCY} rate is recorded on this screen, so "
                "it cannot be tested — only the any-value approvers above can be placed."))
            continue
        op, amount = vr
        hit = native > amount if op == "gt" else native < amount
        rows.append(RouteRow(who, rule, "required" if hit else "not-required",
                             f"{rule} — this deal is {_money(native)}."))
    return Route(matrix=key, matched=key == category, rows=rows)
