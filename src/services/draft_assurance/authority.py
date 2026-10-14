"""Does the commitment a counter makes sit within the sender's authority? Fail-closed.

Reads the platform's own ``resolve_authority`` (the governed, fail-closed limit) rather than a
second copy of the rule. Three outcomes, none of them a guess:

* ``within``      the limit is known, in the same currency, and the commitment is under it
* ``exceeds``     the commitment is over the limit
* ``unresolved``  no limit, a different currency (no exchange rate is invented), or no amount

Anything but ``within`` means a human must clear it. At draft time the authority is the drafting
agent's; at send time the approver's authority is enforced by the approval step.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

from . import validator as V


def commitment_amount(data: Dict[str, Any]) -> Optional[Decimal]:
    """The most a counter binds us to: quantity x price over its lines, else the stated counter price."""

    total = Decimal(0)
    lines = data.get("line_items")
    counter = V.to_decimal(data.get("counter_price"))
    if isinstance(lines, list) and lines and counter is not None:
        for line in lines:
            qty = V.to_decimal((line or {}).get("quantity") or (line or {}).get("qty"))
            price = V.to_decimal((line or {}).get("counter_price") or (line or {}).get("unit_price") or (line or {}).get("price"))
            if qty is None or price is None:
                return counter
            total += qty * price
        return total
    return counter


def check_commitment(policy_engine: Any, agent: str, amount: Optional[Decimal],
                     currency: Optional[str]) -> Dict[str, Any]:
    if amount is None:
        return {"verdict": "unresolved", "reason": "the counter states no amount to check"}
    try:
        from src.services.governance_tools.authority import resolve_authority

        auth = (resolve_authority(policy_engine, [agent]) or {}).get(agent) or {}
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "unresolved", "reason": f"authority could not be resolved: {type(exc).__name__}"}
    if not auth.get("governed") or auth.get("limit_gbp") in (None, ""):
        return {"verdict": "unresolved", "reason": auth.get("reason") or "no authority limit is defined"}
    try:
        limit = Decimal(str(auth["limit_gbp"]))
    except InvalidOperation:
        return {"verdict": "unresolved", "reason": "the authority limit is not a number"}
    cur = (auth.get("limit_currency") or "GBP").upper()
    if (currency or "").upper() != cur:
        return {"verdict": "unresolved", "limit": str(limit), "limit_currency": cur,
                "reason": f"commitment is in {currency or 'an unknown currency'} and the limit is in {cur}; no rate is assumed"}
    return {"verdict": "within" if amount <= limit else "exceeds", "amount": str(amount),
            "limit": str(limit), "limit_currency": cur}
