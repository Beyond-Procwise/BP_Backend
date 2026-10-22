"""Handing one open approval to a named person (D-73 A, 2026-10-22).

Who may decide an approval is the decider map (a Cognito group or an email per level name).
A delegation adds ONE more person to that for ONE case: the person it was handed to may decide
it as if they were linked to the current level. Everything else stands: the requester can
never decide their own request, a case that is already decided cannot be handed on, and the
hand-off is recorded in the case's facts (every hop, in order) and announced to the receiver
as a mandatory notification.

Who may hand it on: someone who could decide it now (linked to the current level, or the
person it was last handed to) or an Admin. Who may receive it: an active person in the org
chart, not the one handing it on, not the requester, with an approval limit that covers the
action's amount when the action names one. "No limit at all" is not "unlimited": a person
with no approval limit cannot be handed an approval.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional, Set

from services.agent_policy import approvals, deciders
from services.people.models import UNLIMITED

EVENT_TYPE = "approval.delegated"
MAX_NOTE = 2000


def identities(principal) -> Set[str]:
    out = set()
    for v in (getattr(principal, "subject", None), getattr(principal, "email", None)):
        v = str(v or "").strip().lower()
        if v:
            out.add(v)
    return out


def current(facts: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The hand-off in force, or None."""
    d = facts.get("delegatedTo")
    return d if isinstance(d, dict) and d.get("subject") else None


def is_delegate(principal, facts: Dict[str, Any]) -> bool:
    d = current(facts)
    if not d:
        return False
    theirs = {str(d.get("subject") or "").lower(), str(d.get("email") or "").lower()} - {""}
    return bool(theirs & identities(principal))


def amount_of(facts: Dict[str, Any]) -> Optional[Decimal]:
    """The action's money amount, when the action names one (facts.amount, else args.amount)."""
    raw = facts.get("amount")
    if raw is None:
        raw = ((facts.get("action") or {}).get("args") or {}).get("amount")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return Decimal(str(raw))
    except (InvalidOperation, ValueError):
        return None


def receiver_problem(person: Optional[Dict[str, Any]], facts: Dict[str, Any], *, by: str) -> Optional[str]:
    """Why this person cannot be handed this approval, as a sentence; None when they can."""
    if person is None or not person.get("is_active"):
        return "That person is not an active, known person."
    name = person.get("display_name") or person["subject"]
    if str(person["subject"]).lower() == str(by or "").lower():
        return "You cannot hand an approval to yourself."
    requester = str(facts.get("requestedBy") or "").strip().lower()
    if requester and requester in {str(person["subject"]).lower(), str(person.get("email") or "").lower()}:
        return f"{name} asked for this action and cannot decide on it."
    limit = person.get("approval_limit")
    if limit is None:
        return f"{name} has no approval limit, so cannot decide approvals."
    amount = amount_of(facts)
    if amount is not None and limit != UNLIMITED and Decimal(str(limit)) < amount:
        return f"{name}'s approval limit does not cover this amount."
    return None


def delegate(conn, decision_id: int, *, principal, to_subject: str, note: Optional[str], now: datetime,
             is_admin: bool, mapping: Optional[deciders.Mapping] = None) -> Dict[str, Any]:
    """Hand an open case to `to_subject`. Raises ApprovalRefused when not allowed."""
    from services.people import admin as people
    to_subject = str(to_subject or "").strip()
    if not to_subject:
        raise approvals.ApprovalRefused("receiver_required", "Say who should decide this.", 422)
    note = (note or "").strip() or None
    if note and len(note) > MAX_NOTE:
        raise approvals.ApprovalRefused("note_too_long", f"A note can be at most {MAX_NOTE} characters.", 422)
    actor = getattr(principal, "subject", None)
    if not actor:
        raise approvals.ApprovalRefused("not_signed_in", "Sign in to hand this on.", 401)
    if mapping is None:
        mapping = deciders.load_map(conn)

    with approvals._tx(conn):
        with conn.cursor() as cur:
            case = approvals._load_locked(cur, decision_id)
            if case is None:
                raise approvals.ApprovalRefused("not_found", "No such approval request.", 404)
            if case["status"] != "open":
                raise approvals.ApprovalRefused("not_open", "This request has already been decided.", 409)
            facts = dict(case["facts"])
            levels, level = case["levels"], case["current_level"]
            level_name = levels[level]["name"] if 0 <= level < len(levels) else None
            may_hand = is_admin or is_delegate(principal, facts) or \
                (bool(level_name) and deciders.eligible(principal, level_name, mapping))
            if not may_hand:
                raise approvals.ApprovalRefused(
                    "not_eligible", "Only the person who can decide this now, or an Admin, can hand it on.", 403)
            person = people.get_person(conn, to_subject)
            problem = receiver_problem(person, facts, by=actor)
            if problem:
                raise approvals.ApprovalRefused("cannot_decide", problem, 422)
            by_person = people.get_person(conn, actor)
            by_name = (by_person or {}).get("display_name") or getattr(principal, "email", None) or actor
            hop = {"subject": person["subject"], "name": person.get("display_name") or person["subject"],
                   "email": person.get("email"), "by": actor, "byName": by_name, "at": now.isoformat(),
                   "note": note, "level": level, "levelName": level_name}
            hops = [h for h in (facts.get("delegations") or []) if isinstance(h, dict)] + [hop]
            facts["delegatedTo"] = hop
            facts["delegations"] = hops
            import json
            cur.execute("UPDATE proc.bp_decision SET facts = %s WHERE decision_id = %s AND subject_type = %s",
                        (json.dumps(facts, default=str), decision_id, approvals.SUBJECT_TYPE))
            approvals._announce(cur, EVENT_TYPE, decision_id, actor,
                                f"{by_name} asked you to decide: {approvals._action_text(facts)}",
                                event_key=f":{len(hops)}", explicit=[person["subject"]],
                                body=(note or f"Handed to you by {by_name}."), requester=facts.get("requestedBy"))
    return {"decisionId": decision_id, "delegatedTo": hop, "level": level, "levelName": level_name,
            "hops": len(hops)}
