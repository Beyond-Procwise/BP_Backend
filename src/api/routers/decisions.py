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


class ActionRequest(BaseModel):
    action: str
    user_id: Optional[str] = None
    # Only for apply_value, when the operator supplies a figure other than the expected.
    value: Optional[str] = None
    # Required when the action contradicts what the evidence supports. Recorded against
    # the actor. Not a flag — a sentence.
    override_reason: Optional[str] = None


@router.post("/finding/{finding_id}/action")
def act_on_finding(
    finding_id: str,
    body: ActionRequest,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Carry out the human's decision on a finding. The engine advises; it does not veto.

    This is what the Action Centre buttons call, and it does the thing:
    `apply_value` writes the corrected figure onto the finding (the UI used to post the
    action and drop the value, so "Apply value" never applied one), `flag` leaves the
    finding OPEN, `dismiss` closes it as accepted risk.

    Human-in-the-loop, precisely:
      * the engine states what the evidence supports before anything happens;
      * if the human's action contradicts that, the call comes back with
        requires_override=true and the reasoning — it will not proceed on a bare click;
      * supply override_reason to go ahead. Who acted, what they were told, and why they
        went the other way are all recorded on proc.bp_decision.

    The human is never blocked. They are asked to mean it.

    The source extraction is never overwritten — the correction is recorded against the
    finding, so what the document actually said stays intact.
    """
    from engines.decision_engine import DecisionEngine

    result = DecisionEngine(agent_nick).execute(
        finding_id,
        body.action,
        user_id=body.user_id or "api",
        value=body.value,
        override_reason=body.override_reason,
    )
    if result.get("error") and not result.get("requires_override"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/{decision_id}")
def get_decision(decision_id: int, agent_nick=Depends(get_agent_nick)) -> Dict[str, Any]:
    """The decision, and every fact it was computed from."""
    from engines.decision_engine import DecisionEngine

    row = DecisionEngine(agent_nick).trace(decision_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")
    return row
