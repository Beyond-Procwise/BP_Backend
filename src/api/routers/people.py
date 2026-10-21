"""People, notification routing and the signed-in user's own notifications.

Admin (gated by policy.read / policy.write, i.e. the 'configure' class):
  GET    /admin/people                        list;  GET /admin/people/{subject}
  PUT    /admin/people/{subject}              create or update (only the fields sent change)
  GET    /admin/people/{subject}/approver     who would approve ?amount= for this person (dry run)
  GET    /admin/routing-rules                 effective rule per event type
  PUT    /admin/routing-rules/{event_type}    set targets for this tenant
  DELETE /admin/routing-rules/{event_type}    go back to the built-in default
Signed-in user, own data only:
  GET  /notifications  /notifications/unread-count  /notifications/preferences
  POST /notifications/{id}/read  /notifications/read-all
  PUT  /notifications/preferences
  GET  /people/me      PUT /people/me/delegate   (out of office)
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from api.auth import require_user
from api.endpoint_gate import require as gate
from services.db import get_conn
from services.people import admin as svc
from services.people.events import CATALOGUE
from services.people.hierarchy import HierarchyService
from services.people.stores import PgPeopleStore

router = APIRouter(tags=["people"])


def _who(principal: Any) -> str:
    return str(getattr(principal, "subject", None) or "unknown")


def _require_signed_in(principal: Any) -> str:
    """Own-data endpoints: any signed-in user, acting only on their own subject."""
    subject = getattr(principal, "subject", None)
    if not subject:
        raise HTTPException(status_code=401, detail="sign in to use this")
    return str(subject)


def _problem(exc: svc.PeopleError):
    raise HTTPException(status_code=exc.status, detail=exc.detail)


class PersonBody(BaseModel):
    display_name: Optional[str] = None
    email: Optional[str] = None
    manager_subject: Optional[str] = None
    department_id: Optional[str] = None
    is_department_head: Optional[bool] = None
    roles: Optional[List[str]] = None
    approval_limit: Optional[float] = None
    is_active: Optional[bool] = None
    delegate_subject: Optional[str] = None
    delegate_until: Optional[str] = None


class DelegateBody(BaseModel):
    delegate_subject: Optional[str] = None     # null clears
    delegate_until: Optional[str] = None


class RuleBody(BaseModel):
    targets: List[str]
    notify_actor: bool = False


# --------------------------------------------------------------------------- admin: people
@router.get("/admin/people", summary="People known to the platform")
def list_people(department: Optional[str] = None, include_inactive: bool = False,
                principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_api")
    with get_conn() as conn:
        return {"people": svc.list_people(conn, department, include_inactive)}


@router.get("/admin/people/{subject}", summary="One person")
def read_person(subject: str, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_api")
    with get_conn() as conn:
        p = svc.get_person(conn, subject)
    if p is None:
        raise HTTPException(status_code=404, detail="no such person")
    return p


@router.put("/admin/people/{subject}", summary="Create or update a person; only the fields sent change")
def put_person(subject: str, body: PersonBody, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="people_api", context={"person": subject})
    sent = body.model_dump(exclude_unset=True)
    try:
        with get_conn() as conn:
            clean = svc.validate_person_update(PgPeopleStore(conn), svc.TENANT, subject, sent)
            return svc.upsert_person(conn, subject, clean, _who(principal))
    except svc.PeopleError as exc:
        _problem(exc)


@router.get("/admin/people/{subject}/approver", summary="Dry run: who would approve this amount for this person")
def preview_approver(subject: str, amount: float = Query(0, ge=0), principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_api")
    with get_conn() as conn:
        store = PgPeopleStore(conn)
        person = store.get(svc.TENANT, subject)
        if person is None:
            raise HTTPException(status_code=404, detail="no such person")
        h = HierarchyService(store)
        found = h.get_approver_for(person, "preview", Decimal(str(amount)))
        chain = [p.subject for p in h.get_chain_of_command(person)]
    return {"chain_of_command": chain,
            "approver": None if not found else {"subject": found[0].subject, "covering_for": found[1]}}


# --------------------------------------------------------------------------- admin: routing rules
@router.get("/admin/routing-rules", summary="Who each event notifies (tenant rule, else built-in default)")
def list_rules(principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.read", principal, agent="people_api")
    from services.people.resolver import DEFAULT_TARGETS, FALLBACK_TARGETS
    with get_conn() as conn:
        custom = {r["event_type"]: r for r in svc.list_rules(conn)}
    rules = []
    for et, (priority, mandatory) in sorted(CATALOGUE.items()):
        r = custom.get(et)
        rules.append({"event_type": et, "priority": priority, "mandatory": mandatory,
                      "targets": r["targets"] if r else DEFAULT_TARGETS.get(et, FALLBACK_TARGETS),
                      "notify_actor": bool(r and r["notify_actor"]), "customised": r is not None})
    return {"rules": rules}


@router.put("/admin/routing-rules/{event_type}", summary="Set who an event notifies")
def put_rule(event_type: str, body: RuleBody, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="people_api", context={"event": event_type})
    try:
        targets, notify_actor = svc.validate_rule(event_type, body.targets, body.notify_actor)
    except svc.PeopleError as exc:
        _problem(exc)
    with get_conn() as conn:
        svc.put_rule(conn, event_type, targets, notify_actor, _who(principal))
    return {"event_type": event_type, "targets": targets, "notify_actor": notify_actor}


@router.delete("/admin/routing-rules/{event_type}", summary="Revert an event to its built-in routing")
def delete_rule(event_type: str, principal=Depends(require_user)) -> dict[str, Any]:
    gate("policy.write", principal, agent="people_api", context={"event": event_type})
    if event_type not in CATALOGUE:
        raise HTTPException(status_code=404, detail=f"{event_type} is not a known event")
    with get_conn() as conn:
        removed = svc.delete_rule(conn, event_type)
    return {"event_type": event_type, "reverted": bool(removed)}


# --------------------------------------------------------------------------- own notifications
@router.get("/notifications", summary="My notifications, newest first")
def my_notifications(unread_only: bool = False, limit: int = Query(30, ge=1, le=100),
                     before_id: Optional[int] = None, principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    with get_conn() as conn:
        return svc.list_notifications(conn, subject, unread_only, limit, before_id)


@router.get("/notifications/unread-count", summary="Number on the bell")
def my_unread(principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    with get_conn() as conn:
        return {"unread": svc.unread_count(conn, subject)}


@router.post("/notifications/read-all", summary="Mark all my notifications read")
def read_all(principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    with get_conn() as conn:
        return {"marked": svc.mark_all_read(conn, subject)}


@router.post("/notifications/{notification_id}/read", summary="Mark one of my notifications read")
def read_one(notification_id: int, principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    with get_conn() as conn:
        if not svc.mark_read(conn, subject, notification_id):
            raise HTTPException(status_code=404, detail="no such notification")
    return {"id": notification_id, "read": True}


@router.get("/notifications/preferences", summary="My notification preferences")
def my_prefs(principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    with get_conn() as conn:
        return {"preferences": svc.get_prefs(conn, subject),
                "mandatory_events": sorted(e for e, (_, m) in CATALOGUE.items() if m),
                "events": sorted(CATALOGUE)}


@router.put("/notifications/preferences", summary="Change my notification preferences")
def set_my_prefs(body: dict, principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    try:
        clean = svc.validate_prefs(body)
    except svc.PeopleError as exc:
        _problem(exc)
    with get_conn() as conn:
        return {"preferences": svc.put_prefs(conn, subject, clean)}


# --------------------------------------------------------------------------- own profile / out of office
@router.get("/people/me", summary="My place in the organisation")
def me(principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    with get_conn() as conn:
        p = svc.get_person(conn, subject)
    return {"subject": subject, "person": p, "known": p is not None}


@router.put("/people/me/delegate", summary="Set or clear who covers for me while I am out")
def set_my_delegate(body: DelegateBody, principal=Depends(require_user)) -> dict[str, Any]:
    subject = _require_signed_in(principal)
    try:
        with get_conn() as conn:
            if svc.get_person(conn, subject) is None:
                raise HTTPException(status_code=404, detail="you are not set up in the organisation yet; ask an administrator")
            clean = svc.validate_person_update(
                PgPeopleStore(conn), svc.TENANT, subject,
                {"delegate_subject": body.delegate_subject, "delegate_until": body.delegate_until})
            return svc.upsert_person(conn, subject, clean, subject)
    except svc.PeopleError as exc:
        _problem(exc)
