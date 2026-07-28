"""Decisions — make one, and check why it was made.

POST /decisions/finding/{id}            — decide what to do about a finding,
                                           grounded in the facts and the governed
                                           policy, and persist it.
POST /decisions/finding/{id}/action     — carry out a human's approve/reject/etc.
                                           on a finding (Action Centre).
POST /decisions/email-reply/{id}        — decide whether to send a supplier reply
                                           unattended, or escalate it.
POST /decisions/email-reply/{id}/action — carry out a human's send/reject on an
                                           already-decided, escalated email reply.
GET  /decisions                         — the escalation queue (Todo list).
GET  /decisions/{decision_id}           — the decision WITH the evidence that
                                           produced it.

The GET is the whole point. A decision you cannot re-derive from source is a guess
you have chosen to trust.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
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


_EMAIL_AGENT = "email_drafting_agent"


@router.post("/email-reply/{response_id}")
def decide_email_reply(
    response_id: str,
    body: Optional[DecideRequest] = None,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Decide a supplier reply: answer it unattended, or escalate it to a human.

    Authority is resolved HERE, from the governed policy -- never taken from the
    request. A limit supplied by the caller would be a limit chosen by the caller,
    which is exactly the ApprovalsAgent mistake this system does not repeat.
    """
    from engines.decision_engine import DecisionEngine
    from src.services.governance_tools.authority import resolve_authority

    authority = resolve_authority(agent_nick.policy_engine, [_EMAIL_AGENT]).get(_EMAIL_AGENT)
    engine = DecisionEngine(agent_nick)
    decision = engine.decide_email_reply(
        response_id,
        authority=authority,
        requested=(body.requested if body else None),
    )
    decision_id = engine.record(
        decision,
        workflow_id=(body.workflow_id if body else None),
        agent=_EMAIL_AGENT,
        created_by=(body.user_id if body and body.user_id else "system"),
    )
    payload = decision.to_dict()
    payload["decision_id"] = decision_id
    return {"decision": payload, "decision_id": decision_id}


class EmailActionRequest(BaseModel):
    action: str
    user_id: Optional[str] = None
    # Required when the action contradicts what the evidence supports. Recorded
    # against the actor. Not a flag -- a sentence. Same convention as
    # ActionRequest.override_reason on the finding path.
    override_reason: Optional[str] = None


@router.post("/email-reply/{decision_id}/action")
def act_on_email_reply(
    decision_id: int,
    body: EmailActionRequest,
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Carry out the human's send/reject on an already-decided email reply.

    This is the sibling of `act_on_finding` for the email path, and exists because
    `act_on_finding` cannot serve it: it calls `DecisionEngine.execute()`, which
    keys off `finding_id` and reads/writes proc.bp_extraction_discrepancy -- a table
    an email decision has no row in. Nothing in this endpoint's path touches that
    table; it reads and writes proc.bp_decision only, via
    `DecisionEngine.act_on_email_reply`.

    Keyed by `decision_id` (not `response_id`) because that is what the queue at
    `GET /decisions` hands the caller -- the queue row carries no `response_id`.

    Human-in-the-loop, same convention as the finding action route: the stored
    decision already states what the evidence supported; sending against one that
    was escalated contradicts that and comes back with requires_override=true and
    the reasoning; rejecting does not conflict, since "do not send" is what an
    escalation is asking a human to weigh. Supplying override_reason proceeds and
    is recorded against the actor. The human is never blocked -- they are asked to
    mean it.
    """
    from engines.decision_engine import DecisionEngine

    result = DecisionEngine(agent_nick).act_on_email_reply(
        decision_id,
        body.action,
        user_id=body.user_id or "api",
        override_reason=body.override_reason,
    )
    if result.get("error") and not result.get("requires_override"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("")
def list_decisions(
    subject_type: Optional[str] = Query(default=None),
    status: str = Query(default="open"),
    limit: int = Query(default=100, ge=1, le=500),
    agent_nick=Depends(get_agent_nick),
) -> Dict[str, Any]:
    """Escalated decisions awaiting a human — the rows behind the Todo list.

    Only escalations are returned. A decision the agent resolved itself is not a
    task; it is an audit record, and putting it here would fill the list with work
    nobody has to do.

    The filters are in SQL, before the LIMIT, and `total` is the true server-side
    count rather than the page size -- a previous screen in this codebase pinned
    its badge to the page size, and a growing backlog looked like a plateau.
    """
    where = ["d.resolution = 'escalated'"]
    params: List[Any] = []
    if subject_type:
        where.append("d.subject_type = %s")
        params.append(subject_type)
    if status:
        where.append("d.status = %s")
        params.append(status)
    clause = " AND ".join(where)

    sql = f"""
        SELECT d.decision_id, d.subject_type, d.subject_id, d.supplier_id, d.deal_id,
               d.decision, d.resolution, d.rationale, d.policy_name, d.facts, d.created_at
          FROM proc.bp_decision d
         WHERE {clause}
         ORDER BY d.created_at DESC
         LIMIT %s
    """
    count_sql = f"SELECT count(*) FROM proc.bp_decision d WHERE {clause}"
    try:
        with agent_nick.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, tuple(params) + (limit,))
                cols = [c[0] for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
                cur.execute(count_sql, tuple(params))
                total = int((cur.fetchone() or [0])[0] or 0)
    except Exception as exc:  # noqa: BLE001
        logger.exception("failed to list decisions")
        raise HTTPException(status_code=500, detail=str(exc))

    for row in rows:
        created = row.get("created_at")
        if created is not None and not isinstance(created, str):
            row["created_at"] = created.isoformat()
    return {"data": rows, "total": total}


@router.get("/{decision_id}")
def get_decision(decision_id: int, agent_nick=Depends(get_agent_nick)) -> Dict[str, Any]:
    """The decision, and every fact it was computed from."""
    from engines.decision_engine import DecisionEngine

    row = DecisionEngine(agent_nick).trace(decision_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")
    return row
