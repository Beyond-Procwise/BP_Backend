"""The second reviewer (design §2, plan decision 11, ruling Q3 of 2026-10-10).

In a business area whose "second reviewer" setting is on, a version that loosens the live
condition in the direction a learning suggestion points cannot go Active until a second person
has recorded a review of that exact draft version. Whether a review is needed is worked out HERE,
from the two conditions and the stored suggestions, never from anything the browser sends: typing
the suggested number by hand needs the review exactly as "Draft this change" does.

When a review is needed
  * The policy has a live version (a policy never made live has nothing to loosen).
  * A suggestion row of the policy, in ANY status except `superseded` (open, not_yet, dismissed,
    saved), still STARTS FROM THE LIVE CONDITION (`learning_patterns.targets(proposal, live)`:
    the live limit is the proposal's `from`; the live list still holds the value). A row about an
    earlier limit therefore demands nothing, and neither does a superseded one: both are about a
    condition that no longer exists. Saving the loosening as a draft marks its suggestion `saved`
    in that same save, and the scan may reopen it later; the rule holds in each of those states.
  * The new condition is looser than the live one in that suggestion's direction
    (`learning_patterns.applied`): by less than was suggested, by exactly that, or by MORE.
  * The second-reviewer setting is on for the business area of the new form OR of the live
    version, each resolved exactly as a save resolves it (`repo._area`). Both are read so that
    moving the policy to another area in the very request that loosens it changes nothing.

What a review covers
  One saved draft version, the policy's latest. Activation writes a NEW version (base + 1) from
  the form it is sent, so it needs a review row for (policy, base version) AND the sent form's
  condition must equal that version's: the reviewer cannot be bypassed by changing the condition
  in the activating request. (That the base version is the latest is the save's own stale-version
  check; it is relied on, not repeated.)

Who may review
  Someone with the Approver role (the router checks the role). Not the person who saved the
  draft (subject and email, compared as `approvals._same_person` does). Not the policy's owner:
  the form's `owner` is a role name that the decider map links to people, and anyone linked to
  the owner named by the draft or by the live version is refused. If an owner name is linked to
  nobody the rule cannot be evaluated, so the review is refused until an administrator links it.
  WHO ACTIVATES is not restricted here: anyone with the Approver role may, including the second
  reviewer. The rules are about who reviews, not who presses Activate.

Fail closed
  If the suggestion table, the business area or the versions cannot be read, activation is
  refused with `second_review_unavailable` and the policy page shows the review as required. A
  stored suggestion that cannot be read is such a failure, not a pass. If the decider map cannot
  be read, no review is recorded. Activation itself never reads the decider map: the identity
  rules were applied when the review was recorded, on a version that cannot change.

A policy with no suggestion rows costs one SELECT at activation and nothing else.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.agent_policy import approval_views, approvals, deciders, learning_patterns
from services.agent_policy.learning import Refused

logger = logging.getLogger(__name__)

MESSAGE = ("A second reviewer must review this change before it goes Active. "
           "It loosens the policy as learning suggested.")
REFUSALS = {
    "not_found": "No such policy.",
    "not_latest": "This is no longer the latest draft of the policy. Reload and try again.",
    "not_needed": "This change does not need a second review.",
    "own_change": "You saved this change, so someone else must review it.",
    "owner": "The policy's owner cannot be its second reviewer.",
    "owner_unmapped": ("Nobody is linked to this policy's owner yet, so a second review cannot be recorded. "
                       "An administrator must link them first."),
    "already_reviewed": "This draft has already been reviewed.",
    "second_review_unavailable": "Can't check whether a second review is needed right now. Try again shortly.",
}
REQUIRED = {"field": "secondReview", "code": "second_review_required", "message": MESSAGE}
UNAVAILABLE = {"field": "secondReview", "code": "second_review_unavailable",
               "message": REFUSALS["second_review_unavailable"]}

_SUGGESTIONS_SQL = """
    SELECT suggestion_id, status, proposal FROM proc.bp_agent_policy_suggestion
     WHERE policy_key = %s AND status <> 'superseded' ORDER BY suggestion_id
"""
_FORM_SQL = "SELECT form_state FROM proc.bp_agent_policy_version WHERE policy_key = %s AND version = %s"
_SETTING_SQL = "SELECT second_reviewer FROM proc.bp_business_area WHERE area_name = %s"
# The base version's condition and its review, if it has one. One statement.
_REVIEWED_SQL = """
    SELECT v.form_state -> 'hidden' -> 'condition', r.reviewed_by, r.reviewed_at
      FROM proc.bp_agent_policy_version v
      LEFT JOIN proc.bp_agent_policy_second_review r ON r.policy_key = v.policy_key AND r.version = v.version
     WHERE v.policy_key = %s AND v.version = %s
"""
_POLICY_SQL = "SELECT live_version, latest_version FROM proc.bp_agent_policy WHERE policy_key = %s FOR UPDATE"
_VERSION_SQL = ("SELECT saved_as, saved_by, form_state FROM proc.bp_agent_policy_version "
                "WHERE policy_key = %s AND version = %s")
_INSERT_SQL = """
    INSERT INTO proc.bp_agent_policy_second_review (policy_key, version, suggestion_id, reviewed_by, reviewed_at, note)
    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (policy_key, version) DO NOTHING RETURNING version
"""


def _json(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, (str, bytes)) else raw


def _condition(form: Any) -> Any:
    hidden = form.get("hidden") if isinstance(form, dict) else None
    return hidden.get("condition") if isinstance(hidden, dict) else None


def loosening(suggestions: List[Dict[str, Any]], live_condition: Any, new_condition: Any) -> Optional[int]:
    """The id of the first suggestion (any status but `superseded`) that starts from the live
    condition and whose direction the new condition loosens in, by any amount. None when there is
    none. Pure. A suggestion that cannot be read raises: the caller must not take that as "none"."""
    for s in suggestions:
        if s["status"] == "superseded":
            continue
        proposal = s["proposal"]
        if not isinstance(proposal, dict):
            raise TypeError(f"suggestion {s['id']} holds no proposal")
        if learning_patterns.targets(proposal, live_condition) and learning_patterns.applied(proposal, new_condition):
            return s["id"]
    return None


def _setting_on(cur, area_name: Optional[str]) -> bool:
    """The second-reviewer setting of the area, resolved exactly as a save resolves the area."""
    from repositories import agent_policy_repo as repo   # local: the repository imports this module

    cur.execute(_SETTING_SQL, (repo._area(cur, area_name)["area_name"],))
    return bool(cur.fetchone()[0])


def needed(cur, policy_key: str, live_version: Optional[int], new_form: Dict[str, Any]) -> Optional[int]:
    """The suggestion that makes a second review necessary before `new_form` goes Active, or None.
    Raises when it cannot tell (see "Fail closed" above); it never answers None on a failed read."""
    if live_version is None:
        return None
    cur.execute(_SUGGESTIONS_SQL, (policy_key,))
    suggestions = [{"id": r[0], "status": r[1], "proposal": _json(r[2])} for r in cur.fetchall()]
    if not suggestions:
        return None
    cur.execute(_FORM_SQL, (policy_key, live_version))
    live_form = _json(cur.fetchone()[0])
    found = loosening(suggestions, _condition(live_form), _condition(new_form))
    if found is None:
        return None
    areas = {(f.get("businessArea") or None) if isinstance(f, dict) else None for f in (new_form, live_form)}
    # every area is read (no short cut), so an unreadable one is always an error
    return found if any([_setting_on(cur, a) for a in sorted(areas, key=str)]) else None


def activation_problems(cur, policy_key: str, live_version: Optional[int], base_version: int,
                        new_form: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What stops `new_form` (saved on top of `base_version`) going Active for want of a second
    review: nothing, "a review is required", or "it cannot be checked right now". Never raises.

    On a failed read the caller's transaction may be unusable; the caller refuses the activation
    on the problem returned here, so nothing more is written in it."""
    try:
        if needed(cur, policy_key, live_version, new_form) is None:
            return []
        cur.execute(_REVIEWED_SQL, (policy_key, base_version))
        row = cur.fetchone()
        reviewed = row is not None and row[1] is not None
        # the review was of the base version's condition: the form sent now must hold the same one
        if reviewed and _json(row[0]) == _condition(new_form):
            return []
        return [dict(REQUIRED)]
    except Exception:  # noqa: BLE001 - not knowing is never a pass
        logger.exception("second review: whether %s needs one before going Active could not be checked; "
                         "the activation is refused", policy_key)
        return [dict(UNAVAILABLE)]


def _refuse(code: str) -> Refused:
    return Refused(code, REFUSALS[code])


def _iso(when: Any) -> Optional[str]:
    return when.astimezone(timezone.utc).isoformat() if isinstance(when, datetime) else None


def record(conn, policy_key: str, version: int, *, principal, note: Optional[str], now: datetime) -> Dict[str, Any]:
    """A second person's review of one saved draft version. Raises Refused (`code`, `message`).

    The two identity rules (not the person who saved it, not the owner) are applied here, whatever
    the caller checked. The policy row is locked first, as a save locks it, so the version judged
    is still the latest when the review is written."""
    note = (note or "").strip() or None
    with approvals._tx(conn):
        with conn.cursor() as cur:
            cur.execute(_POLICY_SQL, (policy_key,))
            head = cur.fetchone()
            if head is None:
                raise _refuse("not_found")
            live_version, latest_version = head
            if version != latest_version:
                raise _refuse("not_latest")
            cur.execute(_VERSION_SQL, (policy_key, version))
            saved_as, saved_by, form = cur.fetchone()
            form = _json(form)
            if saved_as != "draft":
                raise _refuse("not_latest")
            if live_version is None:
                raise _refuse("not_needed")
            try:
                suggestion_id = needed(cur, policy_key, live_version, form)
                if suggestion_id is not None:
                    cur.execute(_FORM_SQL, (policy_key, live_version))
                    live_form = _json(cur.fetchone()[0])
                    mapping = deciders.load_map(conn)
            except Exception:  # noqa: BLE001 - not knowing is never a pass
                logger.exception("second review: %s version %s could not be checked; no review is recorded",
                                 policy_key, version)
                raise _refuse("second_review_unavailable") from None
            if suggestion_id is None:
                raise _refuse("not_needed")
            if approvals._same_person(principal, saved_by):
                raise _refuse("own_change")
            # the owner in force and the owner the draft names: a draft cannot rename its way round the rule
            owners = sorted({str(f.get("owner") or "").strip() for f in (form, live_form) if isinstance(f, dict)} - {""})
            if set(owners) & set(approval_views.caller_names(principal, mapping)):
                raise _refuse("owner")
            if not owners or deciders.unmapped(owners, mapping):
                raise _refuse("owner_unmapped")
            cur.execute(_INSERT_SQL, (policy_key, version, suggestion_id, principal.subject, now, note))
            if cur.fetchone() is None:
                raise _refuse("already_reviewed")
    return {"policyKey": policy_key, "version": version, "reviewedBy": principal.subject, "reviewedAt": _iso(now)}


_NONE = {"required": False, "version": None, "reviewedBy": None, "reviewedAt": None}


def status(cur, policy_key: str, head: Dict[str, Any], versions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For the policy page: whether the latest version, when it is a draft on top of a live
    version, needs a second review, and who gave it. `head` holds `liveVersion` and
    `latestVersion`; `versions` are the policy's versions as the page shows them. Never raises:
    when it cannot tell, the review is shown as required (and it is said in the log)."""
    live, latest = head.get("liveVersion"), head.get("latestVersion")
    draft = next((v for v in versions if v.get("version") == latest), None)
    if live is None or latest == live or draft is None or draft.get("savedAs") != "draft":
        return dict(_NONE)
    try:
        if needed(cur, policy_key, live, draft.get("form")) is None:
            return dict(_NONE)
        cur.execute(_REVIEWED_SQL, (policy_key, latest))
        row = cur.fetchone()
        return {"required": True, "version": latest, "reviewedBy": row[1] if row else None,
                "reviewedAt": _iso(row[2]) if row else None}
    except Exception:  # noqa: BLE001 - the page must not fail, and must not say "no review needed" on a guess
        logger.exception("second review: whether %s version %s needs one could not be worked out; "
                         "it is shown as required", policy_key, latest)
        return {"required": True, "version": latest, "reviewedBy": None, "reviewedAt": None}
