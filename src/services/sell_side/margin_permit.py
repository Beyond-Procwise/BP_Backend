"""Who may read cost and margin: ONE rule, called by the sales router and by the data gateway.

Cost and margin go only to a caller a STATED policy permits. margin.read is a read, and a read no policy speaks to is allowed for
every role -- the anonymous caller included. An allow from that default carries no policy_id, so it is refused here: a missing
SalesMarginReadAuthorityPolicy row withholds margin from everyone rather than handing it to anyone.

This was ``api.routers.sales._require_margin_permit``. It lives here so the gateway asks the same question instead of holding a
copy of the answer. ``gate`` is injectable only so the router can pass its own module-level reference (its tests patch that).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.services import agent_actions

from api.endpoint_gate import NotPermitted, require as _default_gate

MARGIN_NOTE = ("This is front-end margin only: distributor back-end rebates are not "
               "modelled, so it understates what a line may finally earn.")


def check(principal: Any, context: Dict[str, Any], *, agent: str, gate: Optional[Callable[..., Any]] = None) -> None:
    decision = (gate or _default_gate)("margin.read", principal, agent=agent, context=context)
    if decision.policy_id:
        return
    agent_actions.record_action_or_fail(
        phase="authorize", action_type="margin.read", agent=agent, status="denied",
        summary="no stated permit for margin.read; the reversible-read default does "
                "not extend to cost and margin",
        details={**context, "principal": getattr(principal, "subject", None) or None,
                 "allowed_by": decision.policy_name})
    raise NotPermitted("margin.read needs a stated permit and none is active, so "
                       "cost and margin are withheld.")
