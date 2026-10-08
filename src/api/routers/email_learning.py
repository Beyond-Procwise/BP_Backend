"""The queues the email-learning job leaves for people, and the email agent's metrics.

    GET  /email-learning/queues                        how much is waiting in each queue
    GET  /email-learning/data-quality                  facts a reviewer changed that Postgres backs
    POST /email-learning/data-quality/{id}/decision    resolve | dismiss
    GET  /email-learning/review-items                  patterns across reviewers
    POST /email-learning/review-items/{id}/decision    accept | dismiss
    GET  /email-learning/style-rules                   YOUR OWN proposed rules, only
    POST /email-learning/style-rules/{id}/decision     approve | edit | reject  (the person the rule is about)
    GET  /email-learning/exemplars                     candidates (no text)
    GET  /email-learning/exemplars/{id}                one candidate WITH its text (gated harder)
    POST /email-learning/exemplars/{id}/decision       approve | reject  (never the author)
    GET  /email-learning/eval-candidates               corrections that could become eval cases (no draft text)
    POST /email-learning/eval-candidates/{id}/decision export | reject
    GET  /email-learning/classifier-examples           answers people gave to "which kind of email is this?"
    GET  /email-learning/inbound-flags                 replies a person must look at (suspected payment-detail change)
    POST /email-learning/inbound-flags/{id}/decision   confirm (keeps the block) | clear (lifts it; approver authority)
    POST /email-learning/classifier-examples/{id}/decision export | reject
    GET  /email-learning/metrics                       edit distance and fact-conflict rate by family over time

Who did a thing is ``principal.subject`` and nothing else: no body field names a person, and there is no fallback when the
principal is blank. Responses carry human labels and row ids, never an internal table or column name, and never raw email
text except the one detail call that is gated as a configuration change.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from api.auth import require_user
from src.services import guardrail, rbac
from src.services.draft_assurance import connections, inbound, learning, metrics, queues

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/email-learning", tags=["Email learning"])

READ, DECIDE, APPROVE = ("email.learning.read", "read"), ("email.learning.decide", "write"), ("exemplar.approve", "configure")
CLEAR_FLAG = ("inbound.flag.clear", "approve_email")


class Decision(BaseModel):
    action: str
    text: Optional[str] = None       # style rule edit
    note: Optional[str] = None       # data-quality note


def get_agent_nick(request: Request):
    agent_nick = getattr(request.app.state, "agent_nick", None)
    if not agent_nick:
        raise HTTPException(status_code=503, detail="the email agent is not available")
    return agent_nick


def _subject(principal: Any) -> str:
    who = str(getattr(principal, "subject", "") or "").strip()
    if not who:
        raise HTTPException(status_code=401, detail="this action must name an authenticated person")
    return who


def _gate(principal: Any, gate: tuple, context: Optional[Dict[str, Any]] = None) -> str:
    """Name the person, then ask the guardrail. Returns the person."""

    who = _subject(principal)
    decision = guardrail.authorize(gate[0], gate[1], principal, context or {}, policy_engine=rbac.policy_engine())
    if not decision.allowed:
        raise HTTPException(status_code=403, detail="you are not permitted to do this")
    return who


def _store(agent_nick: Any):
    return connections.writer(agent_nick)


def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def _settle(result: Dict[str, Any]) -> Dict[str, Any]:
    if result.get("ok"):
        return {"ok": True}
    error = result.get("error") or "nothing changed"
    status = 404 if "no such" in error else 422
    raise HTTPException(status_code=status, detail=error)


# --- reading ---------------------------------------------------------------------------------------------------------------

@router.get("/queues")
def get_queues(principal=Depends(require_user), agent_nick=Depends(get_agent_nick)) -> Dict[str, Any]:
    who = _gate(principal, READ)
    with _store(agent_nick) as conn:
        return {"waiting": queues.counts(conn, who)}


def _listing(principal, agent_nick, fn, status: Optional[str], limit: int, key: str) -> Dict[str, Any]:
    _gate(principal, READ)
    try:
        with _store(agent_nick) as conn:
            return {key: fn(conn, status=status, limit=limit)}
    except ValueError as exc:
        raise _bad_request(exc)


@router.get("/data-quality")
def list_data_quality(status: Optional[str] = "open", limit: int = 50, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    return _listing(principal, agent_nick, queues.list_data_quality, status, limit, "items")


@router.get("/review-items")
def list_review_items(status: Optional[str] = "open", limit: int = 50, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    return _listing(principal, agent_nick, queues.list_review_items, status, limit, "items")


@router.get("/eval-candidates")
def list_eval_candidates(status: Optional[str] = "candidate", limit: int = 50, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    return _listing(principal, agent_nick, queues.list_eval_candidates, status, limit, "items")


@router.get("/classifier-examples")
def list_classifier_examples(status: Optional[str] = "candidate", limit: int = 50, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    return _listing(principal, agent_nick, queues.list_classifier_examples, status, limit, "items")


@router.get("/exemplars")
def list_exemplars(status: Optional[str] = "candidate", limit: int = 50, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    return _listing(principal, agent_nick, queues.list_exemplars, status, limit, "items")


@router.get("/style-rules")
def list_style_rules(limit: int = 50, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)) -> Dict[str, Any]:
    who = _gate(principal, READ)
    with _store(agent_nick) as conn:
        return {"items": queues.list_style_rules(conn, who, limit)}


@router.get("/exemplars/{exemplar_id}")
def get_exemplar(exemplar_id: int, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)) -> Dict[str, Any]:
    _gate(principal, APPROVE, {"exemplar_id": exemplar_id, "view": "text"})
    with _store(agent_nick) as conn:
        found = queues.exemplar_detail(conn, exemplar_id)
    if found is None:
        raise HTTPException(status_code=404, detail="no such candidate")
    return found


@router.get("/metrics")
def get_metrics(bucket: str = "week", family: Optional[str] = None, days: int = 90,
                principal=Depends(require_user), agent_nick=Depends(get_agent_nick)) -> Dict[str, Any]:
    _gate(principal, READ)
    days = max(1, min(int(days), 3650))
    try:
        with _store(agent_nick) as conn:
            rows = metrics.by_family(conn, bucket=bucket, since=datetime.now(timezone.utc) - timedelta(days=days), family=family)
    except ValueError as exc:
        raise _bad_request(exc)
    return {"bucket": bucket, "days": days, "rows": rows}


# --- deciding --------------------------------------------------------------------------------------------------------------

@router.post("/data-quality/{item_id}/decision")
def decide_data_quality(item_id: int, body: Decision, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    who = _gate(principal, DECIDE, {"queue": "data_quality", "id": item_id, "action": body.action})
    with _store(agent_nick) as conn:
        return _settle(learning.decide_dq_item(conn, item_id, who, body.action, body.note))


@router.post("/review-items/{item_id}/decision")
def decide_review_item(item_id: int, body: Decision, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    who = _gate(principal, DECIDE, {"queue": "review_items", "id": item_id, "action": body.action})
    with _store(agent_nick) as conn:
        return _settle(learning.decide_review_item(conn, item_id, who, body.action))


@router.post("/style-rules/{rule_id}/decision")
def decide_style_rule(rule_id: int, body: Decision, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    who = _gate(principal, DECIDE, {"queue": "style_rules", "id": rule_id, "action": body.action})
    with _store(agent_nick) as conn:
        return _settle(learning.decide_style_rule(conn, rule_id, who, body.action, body.text))


@router.post("/eval-candidates/{item_id}/decision")
def decide_eval_candidate(item_id: int, body: Decision, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    who = _gate(principal, DECIDE, {"queue": "eval_candidates", "id": item_id, "action": body.action})
    with _store(agent_nick) as conn:
        return _settle(learning.decide_candidate(conn, "eval", item_id, who, body.action))


@router.post("/classifier-examples/{item_id}/decision")
def decide_classifier_example(item_id: int, body: Decision, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    who = _gate(principal, DECIDE, {"queue": "classifier_examples", "id": item_id, "action": body.action})
    with _store(agent_nick) as conn:
        return _settle(learning.decide_candidate(conn, "classifier", item_id, who, body.action))


@router.get("/inbound-flags")
def list_inbound_flags(status: Optional[str] = "open", limit: int = 50, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    return _listing(principal, agent_nick, queues.list_inbound_flags, status, limit, "items")


@router.post("/inbound-flags/{flag_id}/decision")
def decide_inbound_flag(flag_id: int, body: Decision, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    # Keeping the block is an ordinary write. LIFTING it needs approver authority: the gate follows the action asked for.
    gate = CLEAR_FLAG if body.action == "clear" else DECIDE
    who = _gate(principal, gate, {"queue": "inbound_flags", "id": flag_id, "action": body.action})
    with _store(agent_nick) as conn:
        return _settle(inbound.decide_flag(conn, flag_id, who, body.action, body.note))


@router.post("/exemplars/{exemplar_id}/decision")
def decide_exemplar(exemplar_id: int, body: Decision, principal=Depends(require_user), agent_nick=Depends(get_agent_nick)):
    who = _gate(principal, APPROVE, {"queue": "exemplars", "id": exemplar_id, "action": body.action})
    with _store(agent_nick) as conn:
        if body.action == "approve":
            try:
                rules = learning.load_rules(rbac.policy_engine())
            except learning.LearningRulesUnavailable as exc:
                raise HTTPException(status_code=503, detail=f"the review period is not configured: {exc}")
            return _settle(learning.approve_exemplar(conn, exemplar_id, who, rules))
        if body.action == "reject":
            return _settle(learning.reject_exemplar(conn, exemplar_id, who))
    raise HTTPException(status_code=422, detail="action must be approve or reject")
