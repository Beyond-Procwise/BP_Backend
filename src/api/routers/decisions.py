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
GET  /decisions/email-reply/{id}/message — the supplier's inbound message behind an
                                           escalated email decision, so a person can
                                           read it beside the reply they are approving.
GET  /decisions                         — the escalation queue (Todo list).
GET  /decisions/{decision_id}           — the decision WITH the evidence that
                                           produced it.

The GET is the whole point. A decision you cannot re-derive from source is a guess
you have chosen to trust.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from api.auth import require_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/decisions", tags=["Decisions"])


def get_agent_nick(request: Request):
    nick = getattr(request.app.state, "agent_nick", None)
    if not nick:
        raise HTTPException(status_code=503, detail="AgentNick not available")
    return nick


def _actor(principal: Any) -> str:
    """Who acted. From the token, never from the body.

    A body-supplied actor is forgeable, and these endpoints resolve findings
    and release supplier mail -- the name recorded against that has to mean
    something.
    """

    subject = str(getattr(principal, "subject", "") or "").strip()
    if not subject:
        raise HTTPException(
            status_code=401, detail="this action must be attributed to a person"
        )
    return subject


class DecideRequest(BaseModel):
    # What the human clicked, if anything. Recorded alongside the engine's own view
    # so a call that went against the evidence is visible afterwards.
    requested: Optional[str] = None
    workflow_id: Optional[str] = None


@router.post("/finding/{finding_id}")
def decide_finding(
    finding_id: str,
    body: DecideRequest,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
) -> Dict[str, Any]:
    actor = _actor(principal)

    from engines.decision_engine import DecisionEngine

    engine = DecisionEngine(agent_nick)
    decision = engine.decide_finding(finding_id, requested=body.requested)
    engine.record(
        decision,
        workflow_id=body.workflow_id,
        agent="decision_engine",
        created_by=actor,
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
    principal=Depends(require_user),
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
    actor = _actor(principal)

    from engines.decision_engine import DecisionEngine

    result = DecisionEngine(agent_nick).execute(
        finding_id,
        body.action,
        user_id=actor,
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
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Decide a supplier reply: answer it unattended, or escalate it to a human.

    Authority is resolved HERE, from the governed policy -- never taken from the
    request. A limit supplied by the caller would be a limit chosen by the caller,
    which is exactly the ApprovalsAgent mistake this system does not repeat.
    """
    actor = _actor(principal)

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
        created_by=actor,
    )
    payload = decision.to_dict()
    payload["decision_id"] = decision_id
    return {"decision": payload, "decision_id": decision_id}


class EmailActionRequest(BaseModel):
    action: str
    # Required when the action contradicts what the evidence supports. Recorded
    # against the actor. Not a flag -- a sentence. Same convention as
    # ActionRequest.override_reason on the finding path.
    override_reason: Optional[str] = None


@router.post("/email-reply/{decision_id}/action")
def act_on_email_reply(
    decision_id: int,
    body: EmailActionRequest,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
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
    actor = _actor(principal)

    from engines.decision_engine import DecisionEngine

    result = DecisionEngine(agent_nick).act_on_email_reply(
        decision_id,
        body.action,
        user_id=actor,
        override_reason=body.override_reason,
    )
    if result.get("error") and not result.get("requires_override"):
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/email-reply/{decision_id}/message")
def get_email_reply_message(
    decision_id: int,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """The supplier's own message behind an escalated email decision.

    Reviewing a reply without the message it answers is not reviewing, and until this
    route existed there was no way to read one: the text is stored, but nothing served
    it. The review panel shows what comes back here beside the reply a person is about
    to send.

    SCOPED THE SAME WAY AS THE ACTION ROUTE, and for the same reason: the lookup
    requires `decision_id` AND subject_type 'email_reply', in SQL. Email decision ids
    and extraction-finding ids are independent sequences that both start at 1, so an
    unscoped lookup by id is a live path to reading (or acting on) something entirely
    unrelated -- a mistake this plan has already had to fix once. A finding's id
    presented here is a 404, not a different document.

    HONEST ABOUT ABSENCE. A decision whose message cannot be found comes back 200 with
    `available: false` and a plain-English `note`, not an error and never a synthesised
    body: the panel renders the absence. 404 is reserved for "there is no such
    email-reply decision", which is a different statement.

    Nothing in the response -- including every error path -- names a table, a column or
    a driver. The keys are the parts of an email (`from`, `subject`, `received_at`,
    `body`), which is what the reader is looking at.

    Reading a decision is not an approval-class action, so authentication alone
    gates this route -- no capability check on top of it, which would lock out
    roles that legitimately need visibility.
    """
    _actor(principal)  # authenticated caller required; no capability check for a read

    # Taken from the engine rather than repeated as a literal here: the scoping value
    # and the value the decision was WRITTEN with must be the same string, always.
    from engines.decision_engine import DecisionEngine

    subject_type = DecisionEngine._EMAIL_SUBJECT_TYPE

    # The decision first: it is also the authorisation check. Its subject_id is the
    # thread identifier the message is found by.
    decision_sql = """
        SELECT decision_id, subject_id, supplier_id, deal_id
          FROM proc.bp_decision
         WHERE decision_id = %s AND subject_type = %s
    """
    # Preferred match is the thread identifier the decision carries. The id fallback
    # covers the one case where a decision recorded no thread identifier because the
    # reply itself could not be found -- it is the same identifier the decision was
    # made from, not a guess at a different message. ORDER BY prefers a thread match
    # and then the most recent message on it, so a multi-round thread resolves to the
    # message the escalation is about rather than an arbitrary one.
    message_sql = """
        SELECT sr.response_from, sr.response_subject, sr.response_text, sr.response_body,
               sr.received_time, sr.response_date, sr.supplier_id
          FROM proc.supplier_response sr
         WHERE sr.unique_id = %s OR sr.id::text = %s
         ORDER BY (sr.unique_id = %s) DESC, sr.id DESC
         LIMIT 1
    """
    try:
        with agent_nick.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(decision_sql, (decision_id, subject_type))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(
                        status_code=404,
                        detail=(
                            f"There is no supplier-reply decision {decision_id}. "
                            "Nothing was read."
                        ),
                    )
                cols = [c[0] for c in cur.description]
                decision = dict(zip(cols, row))
                subject_id = str(decision.get("subject_id") or "")

                message: Optional[Dict[str, Any]] = None
                if subject_id:
                    cur.execute(message_sql, (subject_id, subject_id, subject_id))
                    found = cur.fetchone()
                    if found:
                        mcols = [c[0] for c in cur.description]
                        raw = dict(zip(mcols, found))
                        received = raw.get("received_time") or raw.get("response_date")
                        message = {
                            "from": raw.get("response_from") or None,
                            "subject": raw.get("response_subject") or None,
                            # `response_text` is the body the classifier read, so it is
                            # the body a reviewer should be shown -- the same words the
                            # decision was made from. Falls back to the alternate stored
                            # body only when it is empty.
                            "body": raw.get("response_text") or raw.get("response_body") or None,
                            "received_at": (
                                received.isoformat()
                                if received is not None and not isinstance(received, str)
                                else received
                            ),
                            "supplier": raw.get("supplier_id") or None,
                        }
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001
        # The detail stays in the log; the reader gets a sentence.
        logger.exception("failed to read the supplier message for decision %s", decision_id)
        raise HTTPException(
            status_code=500,
            detail="The supplier's message could not be read. Nothing was changed.",
        )

    available = bool(message and message.get("body"))
    return {
        "decision_id": decision_id,
        "subject_id": subject_id or None,
        "supplier_id": decision.get("supplier_id"),
        "deal_id": decision.get("deal_id"),
        "available": available,
        "message": message if available else None,
        "note": None if available else (
            "The supplier's message could not be found for this decision, so there is "
            "nothing to show. What the decision was made on is still recorded against it."
        ),
    }


@router.get("")
def list_decisions(
    subject_type: Optional[str] = Query(default=None),
    status: str = Query(default="open"),
    limit: int = Query(default=100, ge=1, le=500),
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """Escalated decisions awaiting a human — the rows behind the Todo list.

    Only escalations are returned. A decision the agent resolved itself is not a
    task; it is an audit record, and putting it here would fill the list with work
    nobody has to do.

    The filters are in SQL, before the LIMIT, and `total` is the true server-side
    count rather than the page size -- a previous screen in this codebase pinned
    its badge to the page size, and a growing backlog looked like a plateau.

    THE CITED SENTENCE. Each escalated email decision records the supplier's own
    quoted sentence as an `evidence` fact called `supporting_sentence` -- and it is
    the single most useful thing on the card, because it is what the supplier
    actually wrote. This query used to select `facts` but not `evidence`, so the
    sentence never reached the client and the review panel had nothing to show.

    It is selected NARROWLY (two derived columns) rather than by returning the whole
    `evidence` array, deliberately: this is a polled list endpoint that can be asked
    for up to 500 rows, the recorded decisions in bp_sqldb already average ~1.5 KB of
    evidence each (measured 2026-07-28), and an email decision records ~20 facts with
    long provenance strings on top of that. Shipping all of it on every poll to render
    one sentence would multiply the queue payload for data no card displays. The full
    array is still available a row at a time from `GET /decisions/{decision_id}`,
    which is what the provenance view is for.

    `supporting_sentence_grounding` is returned WITH the sentence and is not
    decoration. The value is the evidence record's own reference -- `verbatim` when
    the sentence was found in the supplier's message, `NOT FOUND in source` when it
    was not (an ungrounded reading is one of the reasons a reply gets escalated in
    the first place). A caller must not attribute an ungrounded sentence to the
    supplier, so the flag travels with the text rather than being assumed.

    Reading the queue is not an approval-class action: authentication alone
    gates this route, no capability check.
    """
    _actor(principal)  # authenticated caller required; no capability check for a read

    where = ["d.resolution = 'escalated'"]
    params: List[Any] = []
    if subject_type:
        where.append("d.subject_type = %s")
        params.append(subject_type)
    if status:
        where.append("d.status = %s")
        params.append(status)
    clause = " AND ".join(where)

    # The CASE guard is load-bearing: jsonb_array_elements() errors on a value that is
    # not an array, and `evidence` is nullable (a decision that failed before it
    # gathered any). CASE short-circuits, so a null or object payload yields NULL here
    # instead of failing the whole queue query.
    sql = f"""
        SELECT d.decision_id, d.subject_type, d.subject_id, d.supplier_id, d.deal_id,
               d.decision, d.resolution, d.rationale, d.policy_name, d.facts, d.created_at,
               CASE WHEN jsonb_typeof(d.evidence) = 'array' THEN (
                    SELECT e->>'value' FROM jsonb_array_elements(d.evidence) e
                     WHERE e->>'fact' = 'supporting_sentence' LIMIT 1
               ) END AS supporting_sentence,
               CASE WHEN jsonb_typeof(d.evidence) = 'array' THEN (
                    SELECT e->>'reference' FROM jsonb_array_elements(d.evidence) e
                     WHERE e->>'fact' = 'supporting_sentence' LIMIT 1
               ) END AS supporting_sentence_grounding
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
        # `detail=str(exc)` put the driver's own words on the wire -- a jsonb or
        # missing-column failure answered every caller with the failing statement, table
        # and column names included. This UI happens to swallow the body, but the
        # gateway, curl and Swagger all render it. Same treatment as the message route
        # and the fail-closed rationale: the reader gets a sentence, the log gets
        # everything, and the reference in the sentence is what connects them.
        ref = uuid.uuid4().hex[:8]
        logger.exception(
            "failed to list decisions [ref %s]: %s: %s", ref, type(exc).__name__, exc
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "The escalation queue could not be read. Nothing was changed. "
                f"The details were recorded for support under reference {ref}."
            ),
        )

    for row in rows:
        created = row.get("created_at")
        if created is not None and not isinstance(created, str):
            row["created_at"] = created.isoformat()
    return {"data": rows, "total": total}


@router.get("/{decision_id}")
def get_decision(
    decision_id: int,
    agent_nick=Depends(get_agent_nick),
    principal=Depends(require_user),
) -> Dict[str, Any]:
    """The decision, and every fact it was computed from.

    Reading a decision is not an approval-class action: authentication alone
    gates this route, no capability check.
    """
    _actor(principal)  # authenticated caller required; no capability check for a read

    from engines.decision_engine import DecisionEngine

    row = DecisionEngine(agent_nick).trace(decision_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"decision {decision_id} not found")
    return row
