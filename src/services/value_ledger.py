"""The value ledger: money a finding or opportunity actually produced.

Pure rules first (validated, converted, derived) so they unit-test on plain values; the
SQL layer below them is thin. Spec: docs/superpowers/specs/2026-09-25-value-ledger-design.md
"""
from __future__ import annotations

import logging
import re
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any, Optional

log = logging.getLogger(__name__)

OUTCOME_TYPES = frozenset({"avoided", "claimed", "recovered", "claim_dropped",
                           "realised_saving", "terms_improved", "cycle_time"})
# What a person may say when closing an open money finding. `accepted` closes it and
# records no row: no money moved.
FINDING_OPEN_OUTCOMES = ("avoided", "claimed", "accepted")
SETTLE_OUTCOMES = ("recovered", "claim_dropped")
SETTLED_STATES = frozenset({"avoided", "recovered", "claim_dropped", "realised_saving"})

_CCY = re.compile(r"^[A-Z]{3}$")
_PENNY = Decimal("0.01")


class LedgerError(ValueError):
    """A request the ledger refuses. ``code`` is what the route returns to the UI."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_outcome(outcome_type: str, amount: Any, currency: Optional[str],
                     evidence_ref: Optional[str]) -> tuple[Optional[Decimal], Optional[str]]:
    if outcome_type not in OUTCOME_TYPES:
        raise LedgerError("invalid_outcome", f"unknown outcome {outcome_type!r}")
    if outcome_type == "claim_dropped":
        return None, None
    try:
        value = Decimal(str(amount).strip())
    except (InvalidOperation, AttributeError):
        raise LedgerError("invalid_amount", f"amount {amount!r} is not a number")
    if not value.is_finite() or value <= 0:
        raise LedgerError("invalid_amount", "amount must be greater than zero")
    value = value.quantize(_PENNY, rounding=ROUND_HALF_UP)
    ccy = str(currency or "").strip().upper()
    if outcome_type != "cycle_time" and not _CCY.match(ccy):
        raise LedgerError("invalid_currency", f"currency {currency!r} is not a 3-letter code")
    if outcome_type == "recovered" and not str(evidence_ref or "").strip():
        raise LedgerError("evidence_required",
                          "a recovered amount needs its credit note or document reference")
    return value, (ccy or None)


def convert_to_gbp(amount: Decimal, currency: str, rates: Optional[dict]) -> dict:
    """GBP at record time, with the rate that produced it. No rate -> None, never a guess."""
    if currency == "GBP":
        return {"amount_gbp": amount, "fx_rate": None, "fx_as_of": None}
    if not rates or currency not in rates or "GBP" not in rates:
        return {"amount_gbp": None, "fx_rate": None, "fx_as_of": None}
    rate = Decimal(str(rates["GBP"])) / Decimal(str(rates[currency]))
    return {"amount_gbp": (amount * rate).quantize(_PENNY, rounding=ROUND_HALF_UP),
            "fx_rate": rate, "fx_as_of": rates.get("_fetched_at")}


def current_state(rows: list[dict]) -> Optional[dict]:
    """The latest row for one source that no other row supersedes."""
    replaced = {r.get("supersedes_id") for r in rows if r.get("supersedes_id")}
    live = [r for r in rows if r["outcome_id"] not in replaced]
    if not live:
        return None
    return max(live, key=lambda r: (r["recorded_at"], r["outcome_id"]))
