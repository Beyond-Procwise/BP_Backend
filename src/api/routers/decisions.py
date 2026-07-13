"""Decisions — make one, and check why it was made.

POST /decisions/finding/{id}   — decide what to do about a finding, grounded in
                                 the facts and the governed policy, and persist it.
GET  /decisions/{decision_id}  — the decision WITH the evidence that produced it.

The GET is the whole point. A decision you cannot re-derive from source is a guess
you have chosen to trust.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/decisions", tags=["Decisions"])


def get_agent_nick(request: Request):
    nick = getattr(request.app.state, "agent_nick", None)
    if not nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return nick


class DecideRequest(BaseModel):
    # What the human clicked, if anything. Recorded alongside the engine's own view
    # so a call that went against the evidence is visible afterwards.
    requested: Optional[str] = None
    user_id: Optional[str] = None
    workflow_id: Optional[str] = None


@router.post("/finding/{finding_id}")
def decide_finding(
    finding_id: str,
    body: DecideRequest,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    from engines.decision_engine import DecisionEngine

    engine = DecisionEngine(agent_nick)
    decision = engine.decide_finding(finding_id, requested=body.requested)
    engine.record(
        decision,
        workflow_id=body.workflow_id,
        agent="decision_engine",
        created_by=body.user_id or "api",
    )

    payload = decision.to_dict()
    # Surface disagreement loudly rather than quietly doing as it is told: if a human
    # asked to approve something the evidence says should escalate, that is exactly
    # the moment worth recording and showing.
    if body.requested and body.requested.lower() != decision.decision.lower():
        payload["conflicts_with_request"] = {
            "requested": body.requested,
            "engine_decision": decision.decision,
            "note": (
                "The engine's view differs from the action requested. Both are "
                "recorded; the request is not overridden."
            ),
        }
    return payload


@router.get("/{decision_id}")
def get_decision(decision_id: int, agent_nick=Depends(get_agent_nick)) -> Dict[str, Any]:
    """The decision, and every fact it was computed from."""
    from engines.decision_engine import DecisionEngine

    row = DecisionEngine(agent_nick).trace(decision_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")
    return row
