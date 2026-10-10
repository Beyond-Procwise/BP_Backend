"""The second reviewer (design §2; ruling Q3 of 2026-10-10; fix round 3 ruling).

Four eyes on any change to what the policy enforces, once learning has raised a suggestion on it.

In a business area whose "second reviewer" setting is on, a version of such a policy cannot go
Active until a second person has recorded a review of that exact draft. Whether a review is
needed is worked out HERE, at activation, never from anything the browser sends. The rule makes
NO judgement about whether a change is tighter or looser: three attempts to prove "not looser"
(by the condition's shape, by replaying past decisions, by synthetic decisions) were each got
round. A tightening needs the second person too.

When a version needs a second review: ALL of
  * protected: the business area of the policy's LIVE version has the setting on. A policy with
    no live version that was live before (retired) is judged against its LAST live version. A
    policy that was never live is not covered.
  * learning has spoken: the policy has at least one row in bp_agent_policy_suggestion, of ANY
    status (superseded and dismissed included). The row's content is not read.
  * and EITHER
      - what it enforces changes at all: the new form differs from the live form in any of
        `ENFORCED` (`outcome`, `deciders`, `notify`, `responseTime`, `limit`, everything
        under `hidden`: condition, checkpoint, actions, inputs, units, timeWindow,
        onMissingData and every other key there; the whole of `source`; and
        `effectiveFrom`). `source.document` is enforced: two policies on the same tool from
        the same document are tiered (each keeps its approval levels), from different
        documents they conflict (last level only). `effectiveFrom` is published to the
        orchestrator as `effective.from`. The missing-data rule is compared as it is
        enforced: for the live form, the one compiled into the live version
        (`compiled.trigger.onMissingData`, what enforcement reads; the form with today's
        settings only when no compiled rule can be read); for the new form, the one a save
        would compile (`compiler.on_missing_for`). So a company default an Admin has switched
        since shows as a change, and saying the rule in force out loud does not;
      - leaving protection: the new form's business area resolves (as a save resolves it,
        `repo._area`) to an area whose setting is off.
  "Differs" is decided on canonical JSON text (`_canon`: keys sorted, no whitespace), so `1` and
  `true` differ, `800` and `800.0` differ, key order is irrelevant, and a missing key differs
  from an explicit null. A false positive only ever adds a review.
  Nothing else is consulted: not the suggestion's proposal or status, not the decision history.

What a review covers
  One saved draft version, the policy's latest. Activation writes a NEW version (base + 1) from
  the form it is sent, so it needs a review row for (policy, base version) AND the sent form must
  equal that version's stored form in every key except `checked` and `changeNote` (the two a
  resubmission legitimately changes: who ticked the confirmation, and the note). Both forms are
  compared as a save stores them (the sent one after `attribute_confirmation`), as canonical
  JSON text. No other normalisation: a key that is absent is not the same as a key that is
  null, so the form to resubmit is the one `GET /{key}` returns for that version.

Who may review
  Someone with the Approver role (the router checks the role). Not anyone who saved ANY version
  after the live (or last live) version up to and including the reviewed one (subject and email,
  compared as `approvals._same_person` does). Not the policy's owner: the form's `owner` is a
  role name the decider map links to people, and anyone linked to the owner named by the draft or
  by the live version is refused. A caller whose sign-in carries no email cannot be told apart
  from an owner linked by email, so is refused (`identity_incomplete`) when an owner's entry
  lists any email. If an owner name is linked to nobody the rule cannot be evaluated, so the
  review is refused until an administrator links it. WHO ACTIVATES is not restricted.

Fail closed
  If the suggestion table, the versions or the business area cannot be read, activation is
  refused with `second_review_unavailable` ("try again shortly") and the policy page shows the
  review as required. If the decider map cannot be read, no review is recorded.
  A stored form that was read but cannot be understood (the live or the draft version's form is
  not JSON, not an object, or its row is not there) is a different, permanent fault: trying
  again cannot help, so activation and the review are refused with `second_review_unreadable`
  ("an administrator needs to look at it"). Which of the two it is, is told from the kind of
  failure (`_fault`), after the rule has run: the rule itself is the same.
  The company settings are different: `settings.load_settings` never raises, it logs a warning
  and hands back the ruled defaults. So this module cannot refuse on unreadable settings by
  itself. What is guaranteed is narrower: inside the save's transaction a settings read the
  database refuses leaves the transaction unusable, so the activation ends in this refusal
  (when a review is needed) or in a server error (when the save's next statement fails), and
  no version goes Active either way. Where there is no transaction (the policy page), and
  when the stored settings read but cannot be understood, the rule is judged with the
  defaults.

Cost for everyone else
  A policy with no suggestion row costs ONE statement at activation. One with a suggestion row in
  an area that is not protected costs that, its live form and the area lookup. The decision
  history (bp_policy_firing) is never read.

Known limits (deliberate)
  * A policy that learning has never raised a suggestion on is not covered.
  * Retiring a policy outright, or creating a brand-new one, needs only the Approver role.
  * An Admin can switch the area's setting off (audited with the before and after values).
  * Wording, owner and the review-by date are not covered (name, situation, the two messages,
    category, sub-area, owner, review-by date, examples, change note): editing them needs no
    review.
  * The company's default response time is applied when a version is compiled and is not
    compared here: a form that names no response time picks up whatever the default is at its
    next save, with no review.
  * One Approver can still weaken a protected policy without editing it: by creating a new
    policy on the same tool and deciding the resulting conflict in the new one's favour, or by
    retiring it. Those are stage 4 and stage 1 powers and are outside this module.
  * A policy that is live in an unprotected area is not covered, even as it moves into a
    protected one; a move between two protected areas needs no review.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.agent_policy import approval_views, approvals, deciders, people
from services.agent_policy.compiler import on_missing_for
from services.agent_policy.learning import Refused
from services.agent_policy.settings import load_settings

logger = logging.getLogger(__name__)

MESSAGE = "A second reviewer must review this change before it goes Active."
REFUSALS = {
    "not_found": "No such policy.",
    "not_latest": "This is no longer the latest draft of the policy. Reload and try again.",
    "not_needed": "This change does not need a second review.",
    "own_change": "You saved this change, so someone else must review it.",
    "owner": "The policy's owner cannot be its second reviewer.",
    "owner_unmapped": ("Nobody is linked to this policy's owner yet, so a second review cannot be recorded. "
                       "An administrator must link them first."),
    "identity_incomplete": ("Your sign-in carries no email address, so we cannot tell whether you are this "
                            "policy's owner."),
    "already_reviewed": "This draft has already been reviewed.",
    "second_review_unavailable": "Can't check whether a second review is needed right now. Try again shortly.",
    "second_review_unreadable": ("This policy's saved form could not be read, so a second review cannot be checked. "
                                 "An administrator needs to look at it."),
}
REQUIRED = {"field": "secondReview", "code": "second_review_required", "message": MESSAGE}
UNAVAILABLE = {"field": "secondReview", "code": "second_review_unavailable",
               "message": REFUSALS["second_review_unavailable"]}
#: A stored form that cannot be understood. No `routeTo`: the message already names who must act.
UNREADABLE = {"field": "secondReview", "code": "second_review_unreadable",
              "message": REFUSALS["second_review_unreadable"]}
#: Who may record a second review, for the policy page (the rule itself is `record`'s).
WHO_MAY_REVIEW = "An approver who is not this policy's owner and has not saved this draft."
#: What reading a stored value that is not what it should be raises: bad JSON (ValueError), a form
#: that is not an object or a row that is not there (TypeError, AttributeError), a key or column
#: that is missing (LookupError). A failed statement raises none of these.
_NOT_READABLE = (ValueError, TypeError, AttributeError, LookupError)
#: The two keys of a form that may differ between the reviewed version and the activating request.
NOT_REVIEWED = ("checked", "changeNote")
#: The keys of a form that say what the policy enforces. A difference in any of them (in
#: anything at all under `hidden` or `source`) is a change that needs the second person.
#: `source.document` decides whether two policies on one tool are tiered or conflicting;
#: `effectiveFrom` is published to the orchestrator as `effective.from`.
ENFORCED = ("outcome", "deciders", "notify", "responseTime", "limit", "hidden", "source", "effectiveFrom")

# "Learning has spoken": any row, any status. Its id is kept on the review row.
_SUGGESTION_SQL = """
    SELECT suggestion_id FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s ORDER BY suggestion_id LIMIT 1
"""
_LAST_LIVE_SQL = "SELECT max(version) FROM proc.bp_agent_policy_version WHERE policy_key = %s AND saved_as = 'live'"
# The live version's form, and the document compiled from it (what enforcement is fed).
_FORM_SQL = "SELECT form_state, compiled FROM proc.bp_agent_policy_version WHERE policy_key = %s AND version = %s"
_SETTING_SQL = "SELECT second_reviewer FROM proc.bp_business_area WHERE area_name = %s"
# The base version's whole form and its review, if it has one. One statement.
_REVIEWED_SQL = """
    SELECT v.form_state, r.reviewed_by, r.reviewed_at, r.reviewed_by_email
      FROM proc.bp_agent_policy_version v
      LEFT JOIN proc.bp_agent_policy_second_review r ON r.policy_key = v.policy_key AND r.version = v.version
     WHERE v.policy_key = %s AND v.version = %s
"""
_POLICY_SQL = "SELECT live_version, latest_version FROM proc.bp_agent_policy WHERE policy_key = %s FOR UPDATE"
_VERSION_SQL = ("SELECT saved_as, saved_by, form_state FROM proc.bp_agent_policy_version "
                "WHERE policy_key = %s AND version = %s")
# Everyone who saved a version after the live one, up to and including the reviewed one.
_SAVERS_SQL = ("SELECT saved_by FROM proc.bp_agent_policy_version "
               "WHERE policy_key = %s AND version > %s AND version <= %s ORDER BY version")
_INSERT_SQL = """
    INSERT INTO proc.bp_agent_policy_second_review
           (policy_key, version, suggestion_id, reviewed_by, reviewed_by_email, reviewed_at, note)
    VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT (policy_key, version) DO NOTHING RETURNING version
"""


def _json(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, (str, bytes)) else raw


def _fault(exc: BaseException) -> str:
    """Which refusal a failed check ends in: `second_review_unreadable` when what was read cannot
    be understood (permanent), `second_review_unavailable` for anything else (a failed database
    read, or a failure nobody foresaw: "try again shortly" is the answer that never passes)."""
    return "second_review_unreadable" if isinstance(exc, _NOT_READABLE) else "second_review_unavailable"


def _canon(value: Any) -> str:
    """Canonical JSON text: what "the same" means here. Keys sorted, no whitespace; `1` is not
    `true`, `800` is not `800.0`, and an absent key is not a null one."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _as_enforced(live_form: Any, compiled: Any) -> Any:
    """The live form, carrying the missing-data rule that was compiled into the live version
    (`compiled.trigger.onMissingData`): that is what enforcement reads, whatever the company
    default has become since. The form is handed back as it is when no compiled rule can be read
    (the form with today's settings is then all there is). Never raises; the form is not changed."""
    try:
        rule = _json(compiled)["trigger"]["onMissingData"]
        if isinstance(rule, str) and rule and isinstance(live_form.get("hidden"), dict):
            return {**live_form, "hidden": {**live_form["hidden"], "onMissingData": rule}}
    except Exception:  # noqa: BLE001 - no compiled document, or not one that can be read
        pass
    return live_form


def _enforced(form: Dict[str, Any], settings: Dict[str, Any]) -> str:
    """What the form enforces, as canonical text: its `ENFORCED` keys (only those it has), with
    `hidden.onMissingData` replaced by the rule in effect (`on_missing_for`: the form's own, else
    the company's for its outcome). Pure. Raises when the form cannot be read."""
    part = {k: form[k] for k in ENFORCED if k in form}
    if isinstance(part.get("hidden"), dict):
        part["hidden"] = {**part["hidden"], "onMissingData": on_missing_for(form, settings)}
    return _canon(part)


def _changed(live_form: Dict[str, Any], new_form: Dict[str, Any], settings: Dict[str, Any]) -> bool:
    """Does `new_form` enforce anything other than `live_form` does? `live_form` carries the
    missing-data rule in force (`_as_enforced`). No judgement of direction. Pure."""
    return _enforced(live_form, settings) != _enforced(new_form, settings)


def _setting_on(cur, area_name: Optional[str]) -> bool:
    """The second-reviewer setting of the area, resolved exactly as a save resolves the area."""
    from repositories import agent_policy_repo as repo   # local: the repository imports this module

    cur.execute(_SETTING_SQL, (repo._area(cur, area_name)["area_name"],))
    return bool(cur.fetchone()[0])


def _judge(cur, policy_key: str, live_version: Optional[int], new_form: Any) -> Optional[Dict[str, Any]]:
    """None when `new_form` needs no second review; else what the rule was judged against:
    `{"suggestion_id", "live_version", "live_form"}`. Raises when it cannot tell. The cheap
    questions come first, so a policy learning never spoke about costs one statement."""
    cur.execute(_SUGGESTION_SQL, (policy_key,))
    row = cur.fetchone()
    if row is None:
        return None
    suggestion_id = row[0]
    if live_version is None:                 # retired: what was last live is what is protected
        cur.execute(_LAST_LIVE_SQL, (policy_key,))
        live_version = cur.fetchone()[0]
        if live_version is None:
            return None                      # never live
    cur.execute(_FORM_SQL, (policy_key, live_version))
    stored = cur.fetchone()
    live_form = _json(stored[0])
    if not _setting_on(cur, live_form.get("businessArea") or None):
        return None
    found = {"suggestion_id": suggestion_id, "live_version": live_version, "live_form": live_form}
    if not isinstance(new_form, dict) or not _setting_on(cur, new_form.get("businessArea") or None):
        return found                         # leaving protection (or a form that cannot be read)
    settings = load_settings(cur.connection)
    return found if _changed(_as_enforced(live_form, stored[1]), new_form, settings) else None


def needed(cur, policy_key: str, live_version: Optional[int], new_form: Dict[str, Any]) -> Optional[int]:
    """The id of a suggestion row of the policy when `new_form` needs a second review before it
    goes Active, else None. `live_version` None means "no live version now" (the last live one is
    looked up). Raises when it cannot tell; it never answers None on a failed read."""
    found = _judge(cur, policy_key, live_version, new_form)
    return None if found is None else found["suggestion_id"]


def _same_form(stored: Any, sent: Any) -> bool:
    """Is the form sent for activation the reviewed version's form, `checked` and `changeNote` aside?"""
    def rest(form):
        return _canon({k: v for k, v in form.items() if k not in NOT_REVIEWED})
    return isinstance(stored, dict) and isinstance(sent, dict) and rest(stored) == rest(sent)


def activation_problems(cur, policy_key: str, live_version: Optional[int], base_version: int,
                        new_form: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What stops `new_form` (saved on top of `base_version`) going Active for want of a second
    review: nothing, "a review is required", "it cannot be checked right now" (a failed read), or
    "this policy's saved form could not be read" (a stored form that cannot be understood, which
    trying again does not mend). Never raises.
    `new_form` is the form as the save will store it (after `attribute_confirmation`).

    On a failed read the caller's transaction may be unusable; the caller refuses the activation
    on the problem returned here, so nothing more is written in it."""
    try:
        if needed(cur, policy_key, live_version, new_form) is None:
            return []
        cur.execute(_REVIEWED_SQL, (policy_key, base_version))
        row = cur.fetchone()
        reviewed = row is not None and row[1] is not None
        # the review was of the base version's whole form: the form sent now must be that form
        if reviewed and _same_form(_json(row[0]), new_form):
            return []
        return [dict(REQUIRED)]
    except Exception as exc:  # noqa: BLE001 - not knowing is never a pass
        logger.exception("second review: whether %s needs one before going Active could not be checked; "
                         "the activation is refused", policy_key)
        return [dict(UNREADABLE if _fault(exc) == "second_review_unreadable" else UNAVAILABLE)]


def _refuse(code: str) -> Refused:
    return Refused(code, REFUSALS[code])


def _iso(when: Any) -> Optional[str]:
    return when.astimezone(timezone.utc).isoformat() if isinstance(when, datetime) else None


def record(conn, policy_key: str, version: int, *, principal, note: Optional[str], now: datetime) -> Dict[str, Any]:
    """A second person's review of one saved draft version. Raises Refused (`code`, `message`).

    The identity rules (nobody who saved a version since the live one, not the owner, nobody
    whose sign-in cannot be compared with the owner's) are applied here, whatever the caller
    checked. The policy row is locked first, as a save locks it, so the version judged is still
    the latest when the review is written."""
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
            saved_as, _saved_by, form = cur.fetchone()
            if saved_as != "draft":
                raise _refuse("not_latest")
            try:
                form = _json(form)
                found = _judge(cur, policy_key, live_version, form)
                if found is not None:
                    cur.execute(_SAVERS_SQL, (policy_key, found["live_version"], version))
                    savers = [r[0] for r in cur.fetchall()]
                    mapping = deciders.load_map(conn)
            except Exception as exc:  # noqa: BLE001 - not knowing is never a pass
                logger.exception("second review: %s version %s could not be checked; no review is recorded",
                                 policy_key, version)
                raise _refuse(_fault(exc)) from None
            if found is None:
                raise _refuse("not_needed")
            # anyone who saved a version since the live one had a hand in this change
            if any(approvals._same_person(principal, who) for who in savers):
                raise _refuse("own_change")
            # the owner in force and the owner the draft names: a draft cannot rename its way round the rule
            owners = sorted({str(f.get("owner") or "").strip() for f in (form, found["live_form"])
                             if isinstance(f, dict)} - {""})
            if set(owners) & set(approval_views.caller_names(principal, mapping)):
                raise _refuse("owner")
            if not str(getattr(principal, "email", None) or "").strip() and \
                    any((mapping.get(o) or {}).get("emails") for o in owners):
                raise _refuse("identity_incomplete")
            if not owners or deciders.unmapped(owners, mapping):
                raise _refuse("owner_unmapped")
            # the email is kept beside the id (NULL when the sign-in has none) so the page can name the reviewer
            email = str(getattr(principal, "email", None) or "").strip() or None
            cur.execute(_INSERT_SQL, (policy_key, version, found["suggestion_id"], principal.subject, email, now,
                                      note))
            if cur.fetchone() is None:
                raise _refuse("already_reviewed")
    return {"policyKey": policy_key, "version": version, "reviewedBy": principal.subject, "reviewedAt": _iso(now)}


_NONE = {"required": False, "version": None, "reviewedBy": None, "reviewedByName": None, "reviewedAt": None}


def status(cur, policy_key: str, head: Dict[str, Any], versions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For the policy page: whether the latest version, when it is a draft of a policy that is
    or was live, needs a second review, and who gave it (`reviewedBy` the sign-in id,
    `reviewedByName` the email recorded with the review, else the one remembered for the id, else that id). The same rule as
    activation (`needed`), so the page and the activation never disagree. While a required review
    is not yet recorded the answer also carries `whoMayReview`, the one sentence that says who may. `head` holds `liveVersion` and
    `latestVersion`; `versions` are the policy's versions as the page shows them. Never raises:
    when it cannot tell, the review is shown as required (and it is said in the log)."""
    live, latest = head.get("liveVersion"), head.get("latestVersion")
    if live is None:                         # retired: judged against what was last live
        live = max((v["version"] for v in versions if v.get("savedAs") == "live"), default=None)
    draft = next((v for v in versions if v.get("version") == latest), None)
    if live is None or latest == live or draft is None or draft.get("savedAs") != "draft":
        return dict(_NONE)
    try:
        if needed(cur, policy_key, live, draft.get("form")) is None:
            return dict(_NONE)
        cur.execute(_REVIEWED_SQL, (policy_key, latest))
        row = cur.fetchone()
        by = row[1] if row else None
        # the reviewer by name: the email recorded with the review, else the one remembered for the
        # id (looked up only then), else the id (never without the id)
        email = (str(row[3] or "").strip() or people.names(cur, [by]).get(by)) if by else None
        return {"required": True, "version": latest, "reviewedBy": by,
                "reviewedByName": approval_views.shown_name(by, email) if row else None,
                "reviewedAt": _iso(row[2]) if row else None,
                **({} if by else {"whoMayReview": WHO_MAY_REVIEW})}
    except Exception:  # noqa: BLE001 - the page must not fail, and must not say "no review needed" on a guess
        logger.exception("second review: whether %s version %s needs one could not be worked out; "
                         "it is shown as required", policy_key, latest)
        return {"required": True, "version": latest, "reviewedBy": None, "reviewedByName": None,
                "reviewedAt": None, "whoMayReview": WHO_MAY_REVIEW}
