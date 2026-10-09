"""Approval cases for agent policies: open a case, let a linked person decide, sweep timeouts.

A case is a proc.bp_decision row (subject_type 'agent_policy_approval') that names its levels in
order. Only a person linked to the CURRENT level's decider name may act, never the person whose
request triggered the action. A reject needs a reason. A timeout escalates to the next level; at
the last level (or with one level) it rejects. A timeout NEVER approves.

What a person did is recorded the way the decision engine records any human action: a new
bp_decision row (status 'actioned', actioned_by/at, override_reason) and the original row's
status flipped to 'actioned' (see DecisionEngine._record_human_action and
_close_original_email_decision; those open their own connection and commit, so the same two
statements are issued here inside the case's lock instead).

Full tool arguments are stored in facts because the replay (Task 5) needs them. Masking applies
when the case is DISPLAYED, never here. Notification text never contains input values.

get_conn() is AUTOCOMMIT: every write here runs in an explicit transaction (_tx).
"""
from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional

from services.agent_policy import deciders, durations

logger = logging.getLogger(__name__)

SUBJECT_TYPE = "agent_policy_approval"
TIMEOUT_ACTOR = "system:timeout"
TIMEOUT_REASON = "No decision in time; a timeout never approves"
VERBS = ("approve", "reject")
_ACTION_AGENT = "agent_policy_approvals"
#: The fixed recipient of "cannot be routed" notices; every caller with the Admin role reads it.
ADMIN_RECIPIENT = "Administrators"
GROUP_ACTOR = "system:group"
GROUP_REFUSED_REASON = "Another required approval was refused"


class ApprovalRefused(Exception):
    """A refused act. `status` is the HTTP status a router should answer with."""

    def __init__(self, code: str, message: str, status: int = 403):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


#: Test seam: called inside the sweep's per-row transaction right after the row is locked.
_after_lock: Optional[Callable[[int], None]] = None


# ---------------------------------------------------------------------------- durations
def _delta(value: Any) -> timedelta:
    """The timer length for an ISO duration; unreadable means the company default (durations.resolve)."""
    return timedelta(seconds=durations.parse(durations.resolve(value)))


# ---------------------------------------------------------------------------- db helpers
@contextmanager
def _tx(conn):
    """One explicit transaction on an AUTOCOMMIT connection (get_conn()); commit or roll back.

    Callers must pass an autocommit connection with no open work: on a connection already in a
    transaction this commits (or rolls back) the caller's pending statements too.
    """
    prev = getattr(conn, "autocommit", False)
    if prev:
        conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        if prev:
            conn.autocommit = True


def _facts(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)) and raw:
        return json.loads(raw)
    return {}


def _jsonable(raw: Any, default):
    if raw is None:
        return default
    if isinstance(raw, (str, bytes)):
        return json.loads(raw)
    return raw


def _notify(cur, firing_id: Optional[int], recipients: Iterable[str], message: str, decision_id: int) -> int:
    if firing_id is None:
        return 0
    n = 0
    for r in recipients:
        r = str(r or "").strip()
        if not r:
            continue
        cur.execute(
            "INSERT INTO proc.bp_policy_notification (firing_id, recipient, message, link) "
            "VALUES (%s, %s, %s, %s)",
            (firing_id, r, message, f"decision:{decision_id}"),
        )
        n += 1
    return n


def _action_text(facts: Dict[str, Any]) -> str:
    """A value-free sentence naming the action, for notifications."""
    action = facts.get("action") or {}
    plain = (facts.get("actionPlain") or "").strip()
    what = plain or f"Running {action.get('tool') or 'an action'}"
    key = (facts.get("policy") or {}).get("id")
    return f"{what[:1].upper()}{what[1:]}" + (f" (policy {key})" if key else "")


# ---------------------------------------------------------------------------- open
def open_case(conn, *, policy_doc: Dict[str, Any], firing_id: int, action: Dict[str, Any],
              requested_by: Optional[str], now: datetime,
              extra_facts: Optional[Dict[str, Any]] = None,
              default_response_time: str = durations.COMPANY_DEFAULT,
              mapping: Optional[deciders.Mapping] = None) -> int:
    """Open one approval case for one matched approve policy. Returns its decision_id.

    `action` = {tool, args, agent, workflowId, userId, reason}; args are stored in full.
    `extra_facts` (e.g. firingGroup, ctx) is merged into facts for the gate/replay.
    """
    if mapping is None:
        mapping = deciders.load_map(conn)
    with _tx(conn):
        with conn.cursor() as cur:
            return _insert_case(cur, policy_doc=policy_doc, firing_id=firing_id, action=action,
                                requested_by=requested_by, now=now, extra_facts=extra_facts,
                                default_response_time=default_response_time, mapping=mapping)


def _insert_case(cur, *, policy_doc: Dict[str, Any], firing_id: int, action: Dict[str, Any],
                 requested_by: Optional[str], now: datetime, mapping: deciders.Mapping,
                 extra_facts: Optional[Dict[str, Any]] = None,
                 default_response_time: str = durations.COMPANY_DEFAULT) -> int:
    """open_case's writes on the caller's cursor, inside the CALLER's transaction (no commit).

    For callers that must open cases atomically with their own writes (replay.py)."""
    key = str(policy_doc.get("id") or "")
    intervention = (policy_doc.get("enforcement") or {}).get("intervention") or {}
    sla = intervention.get("sla") or {}
    within = durations.resolve(sla.get("respondWithin"), default_response_time)
    names = [str(e.get("name")).strip() for e in intervention.get("escalateTo") or []
             if isinstance(e, dict) and str(e.get("name") or "").strip()]
    levels = [{"name": n, "respondWithin": within} for n in names]
    # A timeout escalates while a next level exists; only the last level rejects.
    on_timeout = "escalate_next" if len(levels) > 1 else "reject"
    source = policy_doc.get("source") or {}
    to_approver = (policy_doc.get("outputs") or {}).get("toApprover") or {}
    facts: Dict[str, Any] = {
        "action": {k: action.get(k) for k in ("tool", "args", "agent", "workflowId", "userId", "reason")},
        "actionPlain": ((policy_doc.get("context") or {}).get("actions") or {}).get("plain"),
        "policy": {"id": key, "version": policy_doc.get("version"),
                   "situation": (policy_doc.get("trigger") or {}).get("plain"),
                   "excerpt": source.get("excerpt"), "reference": source.get("reference"),
                   "document": source.get("document")},
        "approvalInputs": list(to_approver.get("show") or []),
        "requestedBy": requested_by,
        "firingId": firing_id,
    }
    if extra_facts:
        facts.update({k: v for k, v in extra_facts.items() if k not in facts})
    missing = deciders.unmapped(names, mapping) if names else []
    if not names:
        missing = ["(no approver named)"]
    if missing:
        facts["unroutable"] = missing
    created_by = requested_by or f"agent:{action.get('agent') or 'unknown'}"
    respond_by = now + _delta(within)

    cur.execute(
        """
        INSERT INTO proc.bp_decision (
            subject_type, subject_id, decision, resolution, rationale, status,
            policy_id, policy_name, facts, evidence, workflow_id, agent, created_by,
            options, respond_by, on_timeout, decision_scope, levels, current_level
        ) VALUES (%s,%s,'approve_or_reject','escalated',%s,'open',
                  NULL,%s,%s,'[]',%s,%s,%s,%s,%s,%s,%s,%s,0)
        RETURNING decision_id
        """,
        (SUBJECT_TYPE, f"{key}:{firing_id}",
         ((policy_doc.get("outputs") or {}).get("toAgent") or {}).get("reason"),
         key, json.dumps(facts, default=str), action.get("workflowId"), action.get("agent"),
         created_by, json.dumps(list(VERBS)), respond_by, on_timeout, "action",
         json.dumps(levels)),
    )
    decision_id = int(cur.fetchone()[0])
    cur.execute(
        "UPDATE proc.bp_policy_firing SET decision_id = %s "
        "WHERE firing_id = %s AND result = 'paused_for_approval'",
        (decision_id, firing_id),
    )
    if missing:
        # Nobody can release it until an administrator links the names: tell the administrators.
        _notify(cur, firing_id, [ADMIN_RECIPIENT],
                f"{_action_text(facts)} cannot be routed: link {', '.join(missing)}", decision_id)
    return decision_id


# ---------------------------------------------------------------------------- act
_CASE_SQL = """
    SELECT decision_id, subject_id, decision, resolution, rationale, policy_name, facts, status,
           levels, current_level, workflow_id, on_timeout
      FROM proc.bp_decision
     WHERE decision_id = %s AND subject_type = %s
"""


def _record(cur, case: Dict[str, Any], *, verb: str, actor: str, reason: Optional[str],
            now: datetime, level: int, level_name: Optional[str]) -> int:
    """The human-action row and the original's close, mirroring the decision engine."""
    facts = dict(case["facts"])
    facts["decidedLevel"] = level
    facts["decidedLevelName"] = level_name
    cur.execute(
        """
        INSERT INTO proc.bp_decision (
            subject_type, subject_id, decision, resolution, rationale, policy_id, policy_name,
            facts, evidence, status, actioned_by, actioned_at, override_reason,
            workflow_id, agent, created_by, current_level
        ) VALUES (%s,%s,%s,%s,%s,NULL,%s,%s,'[]','actioned',%s,%s,%s,%s,%s,%s,%s)
        RETURNING decision_id
        """,
        (SUBJECT_TYPE, case["subject_id"], verb, case["resolution"], case["rationale"],
         case["policy_name"], json.dumps(facts, default=str), actor, now, reason,
         case["workflow_id"], _ACTION_AGENT, actor, level),
    )
    action_id = int(cur.fetchone()[0])
    cur.execute(
        "UPDATE proc.bp_decision SET status = 'actioned' WHERE decision_id = %s AND subject_type = %s",
        (case["decision_id"], SUBJECT_TYPE),
    )
    return action_id


def _update_firing(cur, firing_id: Optional[int], *, result: str, level: int, actor: str,
                   now: datetime, reason: Optional[str], decision_id: Optional[int] = None) -> None:
    """Settle the case's own firing row AND the repeat calls the gate linked to it (decision_id).

    Only approve rows: a notify row records the outcome of the WHOLE group (every approval the
    call needed), so it is settled by _close_group once the group is decided, never by one case.
    """
    if firing_id is None and decision_id is None:
        return
    cur.execute(
        """
        UPDATE proc.bp_policy_firing
           SET result = %s, decided_level = %s, decided_by = %s, decided_at = %s, reason = %s
         WHERE (firing_id = %s OR decision_id = %s) AND result = 'paused_for_approval'
           AND outcome = 'approve'
        """,
        (result, level, actor, now, reason, firing_id, decision_id),
    )


# ---------------------------------------------------------------------------- groups
# One tool call that needs several approvals opens one case per approve policy, all sharing
# facts.firing_group (the action runs only when every one approves). A lone case that a replay
# re-check later joined with a new case is the group 'case:<its id>' (replay._effective_group).
_MEMBERS_SQL = """
    SELECT decision_id FROM proc.bp_decision
     WHERE subject_type = %s AND decision = 'approve_or_reject'
       AND (facts->>'firing_group' = %s OR facts->>'firingGroup' = %s OR decision_id = ANY(%s))
     ORDER BY decision_id
"""


def _group_key(decision_id: int, facts: Dict[str, Any]) -> str:
    g = facts.get("firing_group") or facts.get("firingGroup")
    return str(g) if g else f"case:{decision_id}"


def _member_ids(cur, decision_id: int, facts: Dict[str, Any], *, lock: bool = False) -> List[int]:
    """Every case of the group, in id order. `lock` takes FOR UPDATE on all of them in that
    order, so decisions within one group are serialised and never deadlock each other."""
    group = _group_key(decision_id, facts)
    ids = [int(decision_id)]
    if group.startswith("case:") and group[5:].isdigit():
        ids.append(int(group[5:]))
    cur.execute(_MEMBERS_SQL + (" FOR UPDATE" if lock else ""), (SUBJECT_TYPE, group, group, ids))
    return [int(r[0]) for r in cur.fetchall()]


def group_state(cur, decision_id: int, facts: Dict[str, Any]) -> Dict[str, int]:
    """{open, approved, rejected, total} over the case's group. A closed case counts as approved
    only when its latest recorded action is an approval (as the replay counts it)."""
    ids = _member_ids(cur, decision_id, facts)
    cur.execute(
        """
        SELECT c.status,
               (SELECT a.decision FROM proc.bp_decision a
                 WHERE a.subject_type = c.subject_type AND a.subject_id = c.subject_id
                   AND a.decision IN ('approve','reject') AND a.actioned_by IS NOT NULL
                 ORDER BY a.decision_id DESC LIMIT 1)
          FROM proc.bp_decision c WHERE c.subject_type = %s AND c.decision_id = ANY(%s)
        """,
        (SUBJECT_TYPE, ids),
    )
    out = {"open": 0, "approved": 0, "rejected": 0, "total": 0}
    for status, verb in cur.fetchall():
        out["total"] += 1
        if status == "open":
            out["open"] += 1
        elif verb == "approve":
            out["approved"] += 1
        else:
            out["rejected"] += 1
    return out


def _close_group(cur, case: Dict[str, Any], *, refused: Optional[str], actor: str, now: datetime,
                 reason: Optional[str], level: int) -> Dict[str, int]:
    """After `case` was decided: when it was refused (`refused` = 'rejected' | 'timed_out'), close
    every still-open sibling (the action can no longer run) and tell its level; then settle the
    group's notify rows once nothing in the group is open. Returns the group's state.

    Siblings are taken FOR UPDATE SKIP LOCKED: rows this transaction already holds are taken, and
    a sibling another transaction is deciding right now is left to it (never a wait, so the
    sweep cannot deadlock with a decision)."""
    did = int(case["decision_id"])
    if refused:
        for sid in _member_ids(cur, did, case["facts"]):
            if sid == did:
                continue
            sib = _load_locked(cur, sid, skip_locked=True)
            if sib is None or sib["status"] != "open":
                continue
            lv = sib["current_level"]
            lv_name = sib["levels"][lv]["name"] if 0 <= lv < len(sib["levels"]) else None
            _record(cur, sib, verb="reject", actor=GROUP_ACTOR, reason=GROUP_REFUSED_REASON, now=now,
                    level=lv, level_name=lv_name)
            _update_firing(cur, sib["facts"].get("firingId"), result="rejected", level=lv,
                           actor=GROUP_ACTOR, now=now, reason=GROUP_REFUSED_REASON, decision_id=sid)
            if lv_name:
                _notify(cur, sib["facts"].get("firingId"), [lv_name],
                        f"{_action_text(sib['facts'])} no longer needs your decision: "
                        f"{GROUP_REFUSED_REASON[:1].lower()}{GROUP_REFUSED_REASON[1:]}.", sid)
    state = group_state(cur, did, case["facts"])
    if state["open"] == 0:
        result = refused or ("rejected" if state["rejected"] else "approved")
        cur.execute(
            """
            UPDATE proc.bp_policy_firing
               SET result = %s, decided_level = %s, decided_by = %s, decided_at = %s, reason = %s
             WHERE decision_id = ANY(%s) AND outcome <> 'approve' AND result = 'paused_for_approval'
            """,
            (result, level, actor, now, reason, _member_ids(cur, did, case["facts"])),
        )
    return state


def _load_locked(cur, decision_id: int, *, skip_locked: bool = False) -> Optional[Dict[str, Any]]:
    cur.execute(_CASE_SQL + (" FOR UPDATE SKIP LOCKED" if skip_locked else " FOR UPDATE"),
                (decision_id, SUBJECT_TYPE))
    row = cur.fetchone()
    if not row:
        return None
    return _case_from_row(cur, row)


def _case_from_row(cur, row) -> Dict[str, Any]:
    case = dict(zip([d[0] for d in cur.description], row))
    case["facts"] = _facts(case["facts"])
    case["levels"] = _jsonable(case["levels"], [])
    case["current_level"] = int(case["current_level"] or 0)
    return case


def _same_person(principal, requested_by: Optional[str]) -> bool:
    if not requested_by:
        return False
    who = str(requested_by).strip().lower()
    mine = {str(getattr(principal, "subject", "") or "").strip().lower(),
            str(getattr(principal, "email", "") or "").strip().lower()} - {""}
    return who in mine


def act(conn, decision_id: int, *, principal, verb: str, reason: Optional[str], now: datetime,
        replay: Optional[Callable[[int], Any]] = None,
        mapping: Optional[deciders.Mapping] = None) -> Dict[str, Any]:
    """Approve or reject an open case as `principal`. Raises ApprovalRefused when not allowed."""
    if verb not in VERBS:
        raise ApprovalRefused("unknown_verb", "Only approve or reject is possible.", 422)
    reason = (reason or "").strip() or None
    if verb == "reject" and not reason:
        raise ApprovalRefused("reason_required", "A rejection needs a reason.", 422)
    actor = getattr(principal, "subject", None)
    if not actor:
        raise ApprovalRefused("not_signed_in", "Sign in to decide.", 401)

    if mapping is None:
        mapping = deciders.load_map(conn)

    def _permitted(case):
        """The refusals that need the case's own data. Returns (levels, level, level_name, facts)."""
        if case["status"] != "open":
            raise ApprovalRefused("not_open", "This request has already been decided.", 409)
        levels, level = case["levels"], case["current_level"]
        level_name = levels[level]["name"] if 0 <= level < len(levels) else None
        facts = case["facts"]
        if _same_person(principal, facts.get("requestedBy")):
            raise ApprovalRefused("self_approval",
                                  "You cannot decide on an action your own request triggered.", 403)
        if not level_name or not deciders.eligible(principal, level_name, mapping):
            raise ApprovalRefused("not_eligible",
                                  f"Only someone linked to {level_name or 'this level'} can decide this now.",
                                  403)
        return levels, level, level_name, facts

    with _tx(conn):
        with conn.cursor() as cur:
            cur.execute(_CASE_SQL, (decision_id, SUBJECT_TYPE))
            peek = cur.fetchone()
            if peek is None:
                raise ApprovalRefused("not_found", "No such approval request.", 404)
            peeked = _case_from_row(cur, peek)
            # Permission first, on an unlocked read: a refused caller never takes a group lock.
            _permitted(peeked)
            # every case of the group, locked in id order, before this one is read for real
            # (the case row is locked only here, after the group, so the order stays id order)
            _member_ids(cur, decision_id, peeked["facts"], lock=True)
            case = _load_locked(cur, decision_id)
            if case is None:
                raise ApprovalRefused("not_found", "No such approval request.", 404)
            # the case may have moved (decided, escalated) between the read and the lock: judge again
            levels, level, level_name, facts = _permitted(case)
            action_id = _record(cur, case, verb=verb, actor=actor, reason=reason, now=now,
                                level=level, level_name=level_name)
            _update_firing(cur, facts.get("firingId"),
                           result="approved" if verb == "approve" else "rejected",
                           level=level, actor=actor, now=now, reason=reason,
                           decision_id=decision_id)
            group = _close_group(cur, case, refused=None if verb == "approve" else "rejected",
                                 actor=actor, now=now, reason=reason, level=level)

    out = {"decisionId": decision_id, "actionId": action_id, "verb": verb,
           "result": "approved" if verb == "approve" else "rejected",
           "decidedBy": actor, "decidedAt": now.isoformat(), "level": level, "levelName": level_name,
           "reason": reason, "group": group}
    if verb == "approve":
        # After commit, never inside the lock. A replay failure never undoes the decision.
        try:
            _replay_after_commit(decision_id, replay)
        except Exception:  # noqa: BLE001
            logger.exception("replay after approval %s failed", decision_id)
    return out


def _replay_after_commit(decision_id: int, replay: Optional[Callable[[int], Any]]) -> None:
    """Run the approved action: the injected callable (tests), else replay.run (Task 5).

    Imported lazily so this module never depends on the replay at import time. A missing replay
    module is an ERROR, never a silent no-op: an approved action would otherwise vanish.
    """
    if replay is not None:
        replay(decision_id)
        return
    try:
        from services.agent_policy import replay as _replay
    except ModuleNotFoundError as exc:
        if exc.name == _REPLAY_MODULE:
            logger.error("approved action has nowhere to run: %s is missing (decision %s)",
                         _REPLAY_MODULE, decision_id)
            return
        # replay.py exists but something IT imports does not: a broken module, not a missing one
        logger.exception("approved action could not load %s (decision %s): %s",
                         _REPLAY_MODULE, decision_id, type(exc).__name__)
        return
    except ImportError as exc:
        logger.exception("approved action could not load %s (decision %s): %s",
                         _REPLAY_MODULE, decision_id, type(exc).__name__)
        return
    _replay.run(decision_id)


_REPLAY_MODULE = "services.agent_policy.replay"


# ---------------------------------------------------------------------------- sweep
def sweep(conn, now: datetime, *, decision_ids: Optional[List[int]] = None) -> Dict[str, int]:
    """Escalate or reject every open case past its respond_by. Each case in its own transaction.

    `decision_ids` narrows the sweep (tests on the shared database).
    """
    counts = {"escalated": 0, "rejected": 0, "skipped": 0, "errors": 0}
    sql = ("SELECT decision_id FROM proc.bp_decision WHERE subject_type = %s AND status = 'open' "
           "AND respond_by IS NOT NULL AND respond_by <= %s")
    params: List[Any] = [SUBJECT_TYPE, now]
    if decision_ids is not None:
        sql += " AND decision_id = ANY(%s)"
        params.append(list(decision_ids))
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY respond_by", params)
        ids = [int(r[0]) for r in cur.fetchall()]

    for did in ids:
        try:
            counts[_sweep_one(conn, did, now)] += 1
        except Exception:  # noqa: BLE001
            logger.exception("approval sweep failed for decision %s", did)
            counts["errors"] += 1
    return counts


def _sweep_one(conn, decision_id: int, now: datetime) -> str:
    with _tx(conn):
        with conn.cursor() as cur:
            cur.execute(
                _CASE_SQL + " AND status = 'open' AND respond_by <= %s FOR UPDATE SKIP LOCKED",
                (decision_id, SUBJECT_TYPE, now),
            )
            row = cur.fetchone()
            if not row:
                return "skipped"   # decided, escalated, or being handled by another sweep
            if _after_lock:
                _after_lock(decision_id)
            case = dict(zip([d[0] for d in cur.description], row))
            facts = _facts(case["facts"])
            case["facts"] = facts
            levels = _jsonable(case["levels"], [])
            level = int(case["current_level"] or 0)
            firing_id = facts.get("firingId")
            what = _action_text(facts)

            # Escalate while a next level exists, whatever on_timeout says (it only describes the
            # case for the screen); only the last level rejects.
            if level < len(levels) - 1:
                nxt = level + 1
                cur.execute(
                    "UPDATE proc.bp_decision SET current_level = %s, respond_by = %s WHERE decision_id = %s",
                    (nxt, now + _delta(levels[nxt].get("respondWithin")), decision_id),
                )
                _notify(cur, firing_id, [levels[nxt]["name"]],
                        f"{what} needs your decision; the previous approver did not answer in time.",
                        decision_id)
                return "escalated"

            # Last level, or a single level: reject. A timeout never approves.
            level_name = levels[level]["name"] if 0 <= level < len(levels) else None
            _record(cur, case, verb="reject", actor=TIMEOUT_ACTOR, reason=TIMEOUT_REASON, now=now,
                    level=level, level_name=level_name)
            _update_firing(cur, firing_id, result="timed_out", level=level, actor=TIMEOUT_ACTOR,
                           now=now, reason=TIMEOUT_REASON, decision_id=decision_id)
            case["levels"] = levels
            case["current_level"] = level
            _close_group(cur, case, refused="timed_out", actor=TIMEOUT_ACTOR, now=now,
                         reason=TIMEOUT_REASON, level=level)
            first = levels[0]["name"] if levels else None
            _notify(cur, firing_id, [x for x in (first, facts.get("requestedBy")) if x],
                    f"{what} was rejected: nobody decided in time, and a timeout never approves.",
                    decision_id)
            return "rejected"
