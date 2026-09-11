"""The sales ladder, verbatim from the UI seed.

Source of truth: beyond_procwise_ui/src/lib/processTaxonomy/salesLifecycle.js
(generated from seeds/sales_lifecycle/v1.0.0.csv). These ids are not invented
here; if the seed changes, this changes with it and the test fails first.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

PHASES: Tuple[str, ...] = ("sales.opportunity", "sales.margin", "sales.approval")

SUBPROCESSES: Dict[str, str] = {
    "sales.opportunity.qualified": "sales.opportunity",
    "sales.opportunity.quote-drafted": "sales.opportunity",
    "sales.opportunity.quote-reviewed": "sales.opportunity",
    "sales.margin.cost-to-serve-modelled": "sales.margin",
    "sales.margin.discount-checked": "sales.margin",
    "sales.margin.margin-floor-tested": "sales.margin",
    "sales.approval.deal-desk-review": "sales.approval",
    "sales.approval.pricing-approval": "sales.approval",
    "sales.approval.conditions-attached": "sales.approval",
}

# Where a quote sits on the ladder in each status. Issued stays on the last
# rung reached: the ladder ends at approval and has no "sent" rung to claim.
QUOTE_RUNG: Dict[str, Tuple[str, str]] = {
    "draft": ("sales.opportunity", "sales.opportunity.quote-drafted"),
    "in_review": ("sales.approval", "sales.approval.deal-desk-review"),
    "approved": ("sales.approval", "sales.approval.pricing-approval"),
    "issued": ("sales.approval", "sales.approval.pricing-approval"),
}


def check(phase_id: Optional[str], subprocess_id: Optional[str]) -> None:
    if phase_id is not None and phase_id not in PHASES:
        raise ValueError(f"{phase_id!r} is not a phase on the sales ladder")
    if subprocess_id is None:
        return
    owner = SUBPROCESSES.get(subprocess_id)
    if owner is None:
        raise ValueError(f"{subprocess_id!r} is not a sub-process on the sales ladder")
    if phase_id != owner:
        raise ValueError(f"{subprocess_id!r} belongs to {owner!r}, not {phase_id!r}")
