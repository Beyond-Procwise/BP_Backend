"""The second reviewer (design §2; ruling Q3 of 2026-10-10; fix round 1 rulings, incl. A and B).

In a business area whose "second reviewer" setting is on, a change that makes a learned-about
policy stop asking people about something cannot go Active until a second person has recorded a
review of that exact draft version. Whether a review is needed is worked out HERE, at activation,
from what the policy's own decision history shows. Never from anything the browser sends, and
never from the SHAPE of the condition: a condition can be rewritten in endless ways that mean
the same loosening, so the question asked is "what does this change stop asking people about?".

Witnesses
  A witness is one decision a person made on an approve firing of this policy, under ANY of its
  versions (`learning.witnesses`: one per approval case, people only, the newest
  `learning.MAX_DECISIONS`). It records the checkpoint, the tool (`action_name`), the values the
  firing version's condition read and found (`matched_values`) and those it read and did not
  find (`missing_inputs`). Nothing else about the action was stored.

`gates(form, witness, settings)` -> True | False | None
  True when, under `form`, the witness's action would still pause for a person, worked out as
  enforcement works it out (`enforcement.check` / `_evaluate`):
    * the form's outcome is `approve`;
    * its checkpoint is the witness's;
    * its `hidden.actions.tools` is empty (a policy that lists no action applies to every tool,
      exactly as enforcement reads it) or lists the witness's `action_name`;
    * its condition matches the witness's values, evaluated by the ONE evaluator
      (`conditions.example_result`). `tool.name` is taken from `action_name` (that is what the
      gate puts in the context; `matched_values` holds it only when the firing version's
      condition read it). A field in `missing_inputs` was really absent from the action: it is
      left absent and the form's missing-data rule decides (`compiler.on_missing_for`; as in
      enforcement, anything but an explicit `fail_open` fails closed, i.e. matches).
  None ("cannot tell") when the form or the row cannot be read, the condition is not one the
  evaluator accepts, a value the condition reads is masked (`•••`), or the condition reads a
  field the witness recorded NEITHER as found NOR as missing: "not recorded" is not "absent".

When a version needs a second review: ALL of
  * protected: the business area of the policy's LIVE version has the setting on. A policy with
    no live version that was live before (retired) is judged against its LAST live version. A
    policy that was never live is not covered.
  * learning has spoken: the policy has at least one row in bp_agent_policy_suggestion, of ANY
    status (superseded and dismissed included: a tightening that gets a card superseded must not
    switch the safeguard off). The row's content is not read.
  * and ANY of
      - looser, by evidence: some witness has `gates(live_form, w) is True` and
        `gates(new_form, w) is not True`. "Not True" on purpose: False and cannot-tell both
        require the review;
      - weaker on missing data (ruling A, not evidence): the new form's effective missing-data
        rule is `fail_open` and the live form's is not;
      - leaving protection: the new form's business area resolves (as a save resolves it,
        `repo._area`) to an area whose setting is off.
  Nothing else is consulted: not the suggestion's proposal, not its status.

What a review covers
  One saved draft version, the policy's latest. Activation writes a NEW version (base + 1) from
  the form it is sent, so it needs a review row for (policy, base version) AND the sent form must
  equal that version's stored form in every key except `checked` and `changeNote` (the two a
  resubmission legitimately changes: who ticked the confirmation, and the note). Both forms are
  compared as a save stores them: the sent one after `attribute_confirmation`, both passed
  through JSON. No other normalisation: a key that is absent is not the same as a key that is
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
  If the suggestion table, the versions, the business area, the settings or the decision history
  cannot be read, activation is refused with `second_review_unavailable` and the policy page
  shows the review as required. One witness that cannot be read never raises: it is "cannot
  tell". If the decider map cannot be read, no review is recorded.

Cost for everyone else
  A policy with no suggestion row costs ONE statement at activation. One with a suggestion row in
  an area that is not protected costs that, its live form and the area lookup. Neither reads the
  decision history.

Known consequences and limits (deliberate)
  * Any change that makes the condition read a field no witness recorded needs the review,
    including one that is really a tightening (ruling B: fail closed).
  * Retiring a policy outright needs only the Approver role and is not covered.
  * Creating a brand-new looser policy is not covered (it has no history).
  * A loosening in a region where nobody ever made a decision has no witness, so it is not
    covered (only the newest `learning.MAX_DECISIONS` decisions are read).
  * A policy that is live in an unprotected area is not covered, even as it moves into a
    protected one.
  * An Admin can switch the area's setting off (audited with the before and after values).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from services.agent_policy import approval_views, approvals, conditions, deciders, learning
from services.agent_policy.compiler import on_missing_for
from services.agent_policy.enforcement import MASK
from services.agent_policy.learning import Refused
from services.agent_policy.settings import load_settings

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
    "identity_incomplete": ("Your sign-in carries no email address, so we cannot tell whether you are this "
                            "policy's owner."),
    "already_reviewed": "This draft has already been reviewed.",
    "second_review_unavailable": "Can't check whether a second review is needed right now. Try again shortly.",
}
REQUIRED = {"field": "secondReview", "code": "second_review_required", "message": MESSAGE}
UNAVAILABLE = {"field": "secondReview", "code": "second_review_unavailable",
               "message": REFUSALS["second_review_unavailable"]}
#: The two keys of a form that may differ between the reviewed version and the activating request.
NOT_REVIEWED = ("checked", "changeNote")

# "Learning has spoken": any row, any status. Its id is kept on the review row.
_SUGGESTION_SQL = """
    SELECT suggestion_id FROM proc.bp_agent_policy_suggestion WHERE policy_key = %s ORDER BY suggestion_id LIMIT 1
"""
_LAST_LIVE_SQL = "SELECT max(version) FROM proc.bp_agent_policy_version WHERE policy_key = %s AND saved_as = 'live'"
_FORM_SQL = "SELECT form_state FROM proc.bp_agent_policy_version WHERE policy_key = %s AND version = %s"
_SETTING_SQL = "SELECT second_reviewer FROM proc.bp_business_area WHERE area_name = %s"
# The base version's whole form and its review, if it has one. One statement.
_REVIEWED_SQL = """
    SELECT v.form_state, r.reviewed_by, r.reviewed_at
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
    INSERT INTO proc.bp_agent_policy_second_review (policy_key, version, suggestion_id, reviewed_by, reviewed_at, note)
    VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (policy_key, version) DO NOTHING RETURNING version
"""


def _json(raw: Any) -> Any:
    return json.loads(raw) if isinstance(raw, (str, bytes)) else raw


def _fail_open(form: Any, settings: Dict[str, Any]) -> bool:
    """The form's effective missing-data rule, read as enforcement reads the compiled one:
    only an explicit `fail_open` lets an action through when a value is missing."""
    return on_missing_for(form, settings) == "fail_open"


def gates(form: Any, witness: Dict[str, Any], settings: Dict[str, Any]) -> Optional[bool]:
    """Would the witness's action still pause for a person under `form`? True, False, or None for
    "cannot tell" (see the module docstring). Pure; never raises."""
    try:
        hidden = form["hidden"]
        if form.get("outcome") != "approve":
            return False
        if hidden.get("checkpoint") != witness["checkpoint"]:
            return False
        tool = witness["action_name"]
        listed = (hidden.get("actions") or {}).get("tools") or []
        if not isinstance(listed, list):
            return None
        if listed and tool not in listed:
            return False
        found, missing = witness["matched_values"], witness["missing_inputs"]
        if not isinstance(found, dict) or not isinstance(missing, list):
            return None
        values = {k: v for k, v in found.items() if k not in missing and v is not None}
        values["tool.name"] = tool
        condition = hidden.get("condition")
        conditions.to_engine(condition)
        for field in conditions.condition_fields(condition):
            if field in values:
                if values[field] == MASK:
                    return None          # a hidden value proves nothing either way
            elif field not in missing and field not in found:
                return None              # not recorded is not the same as absent
        on_missing = "fail_open" if _fail_open(form, settings) else "fail_closed"
        return conditions.example_result(condition, "approve", values, on_missing) == "approve"
    except Exception:  # noqa: BLE001 - ConditionError, an unreadable form or row: cannot tell
        return None


def loosening(live_form: Any, new_form: Any, witnesses: List[Dict[str, Any]], settings: Dict[str, Any]) -> bool:
    """Does `new_form` stop asking people about something the live form asks about? By evidence
    (a witness the live form gates and the new form does not, or cannot be shown to), or because
    missing data would newly be let through (ruling A). Pure. Raises when a form cannot be read."""
    if _fail_open(new_form, settings) and not _fail_open(live_form, settings):
        return True
    return any(gates(live_form, w, settings) is True and gates(new_form, w, settings) is not True
               for w in witnesses)


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
    live_form = _json(cur.fetchone()[0])
    if not _setting_on(cur, live_form.get("businessArea") or None):
        return None
    found = {"suggestion_id": suggestion_id, "live_version": live_version, "live_form": live_form}
    if not isinstance(new_form, dict) or not _setting_on(cur, new_form.get("businessArea") or None):
        return found                         # leaving protection (or a form that cannot be read)
    settings = load_settings(cur.connection)
    return found if loosening(live_form, new_form, learning.witnesses(cur, policy_key), settings) else None


def needed(cur, policy_key: str, live_version: Optional[int], new_form: Dict[str, Any]) -> Optional[int]:
    """The id of a suggestion row of the policy when `new_form` needs a second review before it
    goes Active, else None. `live_version` None means "no live version now" (the last live one is
    looked up). Raises when it cannot tell; it never answers None on a failed read."""
    found = _judge(cur, policy_key, live_version, new_form)
    return None if found is None else found["suggestion_id"]


def _same_form(stored: Any, sent: Any) -> bool:
    """Is the form sent for activation the reviewed version's form, `checked` and `changeNote` aside?"""
    def rest(form):
        return {k: v for k, v in json.loads(json.dumps(form)).items() if k not in NOT_REVIEWED}
    return isinstance(stored, dict) and isinstance(sent, dict) and rest(stored) == rest(sent)


def activation_problems(cur, policy_key: str, live_version: Optional[int], base_version: int,
                        new_form: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What stops `new_form` (saved on top of `base_version`) going Active for want of a second
    review: nothing, "a review is required", or "it cannot be checked right now". Never raises.
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
            form = _json(form)
            if saved_as != "draft":
                raise _refuse("not_latest")
            try:
                found = _judge(cur, policy_key, live_version, form)
                if found is not None:
                    cur.execute(_SAVERS_SQL, (policy_key, found["live_version"], version))
                    savers = [r[0] for r in cur.fetchall()]
                    mapping = deciders.load_map(conn)
            except Exception:  # noqa: BLE001 - not knowing is never a pass
                logger.exception("second review: %s version %s could not be checked; no review is recorded",
                                 policy_key, version)
                raise _refuse("second_review_unavailable") from None
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
            cur.execute(_INSERT_SQL, (policy_key, version, found["suggestion_id"], principal.subject, now, note))
            if cur.fetchone() is None:
                raise _refuse("already_reviewed")
    return {"policyKey": policy_key, "version": version, "reviewedBy": principal.subject, "reviewedAt": _iso(now)}


_NONE = {"required": False, "version": None, "reviewedBy": None, "reviewedAt": None}


def status(cur, policy_key: str, head: Dict[str, Any], versions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """For the policy page: whether the latest version, when it is a draft of a policy that is
    or was live, needs a second review, and who gave it. The same rule as activation (`needed`),
    so the page and the activation never disagree. `head` holds `liveVersion` and
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
        return {"required": True, "version": latest, "reviewedBy": row[1] if row else None,
                "reviewedAt": _iso(row[2]) if row else None}
    except Exception:  # noqa: BLE001 - the page must not fail, and must not say "no review needed" on a guess
        logger.exception("second review: whether %s version %s needs one could not be worked out; "
                         "it is shown as required", policy_key, latest)
        return {"required": True, "version": latest, "reviewedBy": None, "reviewedAt": None}
