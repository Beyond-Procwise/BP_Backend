"""The people knowledge graph: questions, proposals, the tier switch, and erasure.

Admin (policy.read / policy.write):
  GET  /admin/graph/who-can-act?action=&amount=&as_of=     who may do this (informs; the auth layer still decides)
  GET  /admin/graph/who-has-acted?key=&as_of=              decisions on an item / supplier / category
  GET  /admin/graph/who-knows?key=&k=&as_of=               owners and people who keep deciding there
  GET  /admin/graph/bottlenecks?window_days=&as_of=        slow approvers, repeated escalations
  GET  /admin/graph/suggest?actor=&entity=&supplier=&category=&amount=&as_of=   tiered recipients, each labelled
  POST /admin/graph/refresh                                run the daily sync + derivation now
  GET/PUT /admin/graph/tiers    GET /admin/graph/acceptance   the learning loop and the (default OFF) switch
  GET  /admin/graph/proposals   POST /admin/graph/proposals/{id}/confirm|reject
  DELETE /admin/graph/users/{subject}?pseudonymise=        erase a person's activity and derived edges
Signed-in person:
  DELETE /people/me/activity                               erase MY activity and derived edges
Runtime authorisation never reads these answers.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_user
from api.endpoint_gate import require as gate
from services.db import get_conn
from services.people import admin as people_admin
from services.people.admin import PeopleError, TENANT
from services.people.graph import activity, proposed, queries, service

router = APIRouter(tags=["people-graph"])


def _who(principal: Any) -> str:
    return str(getattr(principal, "subject", None) or "unknown")


def _require_signed_in(principal: Any) -> str:
    subject = getattr(principal, "subject", None)
    if not subject:
        raise HTTPException(status_code=401, detail="sign in to use this")
    return str(subject)


def _when(v: Optional[str]) -> Optional[datetime]:
    if not v:
        return None
    try:
        return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=422, detail="as_of must be an ISO date-time")


def _problem(exc: PeopleError):
    raise HTTPException(status_code=exc.status, detail=exc.detail)


class TiersBody(BaseModel):
    enabled: bool
    force: bool = False


@router.get("/admin/graph/who-can-act", summary="Who may do this action (from the mirrored RBAC; advisory)")
def who_can_act(action: str, amount: Optional[float] = None, as_of: Optional[str] = None,
                principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_graph_api")
    return {"people": queries.who_can_act(service.get_store(), TENANT, action, amount=amount, as_of=_when(as_of))}


@router.get("/admin/graph/who-has-acted", summary="Who decided on this item, supplier or category")
def who_has_acted(key: str, as_of: Optional[str] = None, limit: int = Query(20, ge=1, le=200),
                  principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_graph_api")
    return {"people": queries.who_has_acted(service.get_store(), TENANT, key, as_of=_when(as_of), limit=limit)}


@router.get("/admin/graph/who-knows", summary="Who knows about a supplier or category")
def who_knows(key: str, k: int = Query(5, ge=1, le=25), as_of: Optional[str] = None,
              principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_graph_api")
    return {"people": queries.who_knows_about(service.get_store(), TENANT, key, k=k, as_of=_when(as_of))}


@router.get("/admin/graph/bottlenecks", summary="Approvers who are slow or often escalate")
def bottlenecks(window_days: int = Query(90, ge=1, le=730), as_of: Optional[str] = None,
                principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_graph_api")
    return {"people": queries.find_bottlenecks(service.get_store(), TENANT, window_days=window_days, as_of=_when(as_of))}


@router.get("/admin/graph/suggest", summary="Tiered recipient suggestions for an actor and an item")
def suggest(actor: str, entity: Optional[str] = None, supplier: Optional[str] = None, category: Optional[str] = None,
            amount: Optional[float] = None, as_of: Optional[str] = None, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_graph_api")
    with get_conn() as conn:
        on = service.tiers_enabled(conn)
    rows = queries.suggest_recipients(service.get_store(), TENANT, actor, entity_key=entity, supplier_key=supplier,
                                      category_key=category, amount=amount, graph_tiers=on, as_of=_when(as_of))
    return {"graph_tiers_enabled": on, "suggestions": rows}


@router.post("/admin/graph/refresh", summary="Run the sync and derivation now")
def refresh(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="people_graph_api", context={"op": "refresh"})
    with get_conn() as conn:
        return service.refresh(conn)


@router.get("/admin/graph/tiers", summary="Is graph-suggested routing on? (default off)")
def get_tiers(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_graph_api")
    with get_conn() as conn:
        return service.acceptance_report(conn)


@router.put("/admin/graph/tiers", summary="Switch graph tiers 2-4 on or off; on needs acceptance evidence unless forced")
def put_tiers(body: TiersBody, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="people_graph_api", context={"op": "tiers"})
    try:
        with get_conn() as conn:
            return service.set_tiers(conn, body.enabled, _who(principal), force=body.force)
    except PeopleError as exc:
        _problem(exc)


@router.get("/admin/graph/acceptance", summary="How often each suggestion tier is accepted")
def acceptance(days: int = Query(90, ge=1, le=730), principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_graph_api")
    with get_conn() as conn:
        return service.acceptance_report(conn, days)


@router.get("/admin/graph/proposals", summary="Inferred edges waiting for a person to confirm")
def proposals(status: str = "proposed", principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_graph_api")
    if status not in ("proposed", "confirmed", "rejected"):
        raise HTTPException(status_code=422, detail="status must be proposed, confirmed or rejected")
    with get_conn() as conn:
        return {"proposals": proposed.list_proposals(conn, status)}


def _require_decide(proposal_id: int, principal, confirm: bool) -> dict[str, Any]:
    gate("policy.write", principal, agent="people_graph_api", context={"proposal": proposal_id})
    try:
        with get_conn() as conn:
            return proposed.decide(conn, service.writer(), proposal_id, _who(principal), confirm)
    except PeopleError as exc:
        _problem(exc)


@router.post("/admin/graph/proposals/{proposal_id}/confirm", summary="Accept an inference; it becomes source=inferred")
def confirm_proposal(proposal_id: int, principal=Depends(require_user)) -> dict[str, Any]:
    return _require_decide(proposal_id, principal, True)


@router.post("/admin/graph/proposals/{proposal_id}/reject", summary="Turn an inference down")
def reject_proposal(proposal_id: int, principal=Depends(require_user)) -> dict[str, Any]:
    return _require_decide(proposal_id, principal, False)


@router.delete("/admin/graph/users/{subject}", summary="Erase a person's activity and derived edges")
def erase_user(subject: str, pseudonymise: bool = False, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="people_graph_api", context={"op": "erase", "person": subject})
    with get_conn() as conn:
        return activity.erase_user(conn, service.get_store(), TENANT, subject, pseudonymise=pseudonymise)


@router.delete("/people/me/activity", summary="Erase my activity and the connections derived from it")
def erase_me(principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    with get_conn() as conn:
        return activity.erase_user(conn, service.get_store(), TENANT, subject)
