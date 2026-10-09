"""What the approval, notification, decider and firing screens read (stage 3, Task 7).

Read-only shaping of proc.bp_decision approval cases, proc.bp_policy_firing,
proc.bp_policy_notification and proc.bp_policy_decider_map for the gateway-only endpoints.

Masking (global constraint): a sensitive input shows enforcement.MASK in every answer, except
to a person who may decide the case at its CURRENT level -- linked to that level's decider name
and not the person whose request triggered the action (the self-approval bar). Admins read
every case but see what anyone else sees unless they are linked themselves. Which fields are
sensitive is the union over the case's own policy version and every live policy, the same
union enforcement.check masks with, so a field one policy marks sensitive is never shown
because another policy forgot to.

Full tool arguments are never returned: only the policy's approval-details list (toApprover.show).
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from services.agent_policy import approvals, conditions, conflict_payload, deciders
from services.agent_policy.enforcement import MASK
from services.agent_policy.replay import mask_text

KEY_RE = re.compile(r"^[A-Z]{3}-[0-9]{4,}$")
DECIDER_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 &,'.-]{0,63}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_ABSENT = object()
CASE_LIMIT = 200


# ---------------------------------------------------------------------------- helpers
def _j(v: Any, default=None):
    if v is None:
        return default
    if isinstance(v, (str, bytes)):
        return json.loads(v) if v else default
    return v


def _iso(v: Any) -> Optional[str]:
    return v.isoformat() if v is not None and hasattr(v, "isoformat") else v


def _rows(cur) -> List[Dict[str, Any]]:
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def _sensitive_of(doc: Any) -> Set[str]:
    if not isinstance(doc, dict):
        return set()
    return {str(i.get("field")) for i in doc.get("inputs") or []
            if isinstance(i, dict) and i.get("sensitive") and i.get("field")}


def _compiled(cur, pairs: Iterable[Tuple[str, int]]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """{(policy_key, version): compiled doc} for the given pairs."""
    pairs = sorted({(str(k), int(v)) for k, v in pairs if k and v is not None})
    if not pairs:
        return {}
    cur.execute("SELECT policy_key, version, compiled FROM proc.bp_agent_policy_version "
                "WHERE (policy_key, version) IN (SELECT * FROM unnest(%s::text[], %s::int[]))",
                ([k for k, _ in pairs], [v for _, v in pairs]))
    return {(r[0], int(r[1])): _j(r[2], {}) for r in cur.fetchall()}


def _live_sensitive(cur) -> Set[str]:
    cur.execute("SELECT v.compiled FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v "
                "ON v.policy_key = p.policy_key AND v.version = p.live_version WHERE p.status = 'live'")
    out: Set[str] = set()
    for (doc,) in cur.fetchall():
        out |= _sensitive_of(_j(doc, {}))
    return out


def _mask_values(values: Dict[str, Any], sensitive: Set[str]) -> Dict[str, Any]:
    return {k: (MASK if k in sensitive else v) for k, v in (values or {}).items()}


def mask_witness(example: Dict[str, Any], sensitive: Set[str]) -> Dict[str, Any]:
    """An overlap example (flat field names, "args.amount") with its sensitive values masked."""
    return _mask_values(example if isinstance(example, dict) else {}, sensitive)


def mask_args(args: Dict[str, Any], sensitive: Set[str]) -> Dict[str, Any]:
    """Tool arguments (bare names, "amount") with the sensitive ones ("args.amount") masked."""
    return {k: (MASK if f"args.{k}" in sensitive else v) for k, v in (args if isinstance(args, dict) else {}).items()}


def overlap_example(evidence: Any) -> Dict[str, Any]:
    """The witness a conflict case stores in its evidence column."""
    for e in _j(evidence, []) or []:
        if isinstance(e, dict) and e.get("kind") == "overlap" and isinstance(e.get("example"), dict):
            return dict(e["example"])
    return {}


def _lookup(ctx: Dict[str, Any], path: str) -> Any:
    cur: Any = conditions.nest({k: v for k, v in (ctx or {}).items() if k != "checkpoint"})
    for part in str(path).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _ABSENT
        cur = cur[part]
    return cur


def _ctx(facts: Dict[str, Any]) -> Dict[str, Any]:
    stored = facts.get("ctx")
    if isinstance(stored, dict) and stored:
        return stored
    action = facts.get("action") or {}
    return {"tool.name": action.get("tool"), "agent.name": action.get("agent"),
            "agent.reason": action.get("reason"), "args": dict(action.get("args") or {})}


# ---------------------------------------------------------------------------- who
def caller_names(principal, mapping: deciders.Mapping) -> List[str]:
    """The decider names the caller is linked to."""
    return sorted(n for n in mapping or {} if deciders.eligible(principal, n, mapping))


def _identities(principal) -> List[str]:
    out = []
    for v in (getattr(principal, "subject", None), getattr(principal, "email", None)):
        v = str(v or "").strip()
        if v:
            out.extend({v, v.lower()})
    return sorted(set(out))


def _level_name(case: Dict[str, Any]) -> Optional[str]:
    levels, level = case["levels"], case["current_level"]
    return levels[level].get("name") if 0 <= level < len(levels) and isinstance(levels[level], dict) else None


def can_decide(principal, case: Dict[str, Any], mapping: deciders.Mapping) -> bool:
    """Linked to the current level and not the requester (status aside: masking uses this too)."""
    name = _level_name(case)
    if not name or approvals._same_person(principal, case["facts"].get("requestedBy")):
        return False
    return deciders.eligible(principal, name, mapping)


def _linked_to_any_level(principal, case: Dict[str, Any], mapping: deciders.Mapping) -> bool:
    return any(deciders.eligible(principal, str((lv or {}).get("name") or ""), mapping)
               for lv in case["levels"] if isinstance(lv, dict))


def may_read(principal, case: Dict[str, Any], mapping: deciders.Mapping, *, is_admin: bool) -> bool:
    return (is_admin or _linked_to_any_level(principal, case, mapping)
            or approvals._same_person(principal, case["facts"].get("requestedBy")))


# ---------------------------------------------------------------------------- cases
_CASE_COLS = ("decision_id, subject_id, policy_name, facts, status, levels, current_level, "
              "respond_by, on_timeout, options, created_at")


def _case(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    row["facts"] = _j(row.get("facts"), {}) or {}
    row["levels"] = _j(row.get("levels"), []) or []
    row["current_level"] = int(row.get("current_level") or 0)
    row["options"] = _j(row.get("options"), list(approvals.VERBS)) or list(approvals.VERBS)
    return row


def load_case(cur, decision_id: int) -> Optional[Dict[str, Any]]:
    cur.execute(f"SELECT {_CASE_COLS} FROM proc.bp_decision WHERE decision_id = %s AND subject_type = %s "
                "AND decision = 'approve_or_reject'", (decision_id, approvals.SUBJECT_TYPE))
    rows = _rows(cur)
    return _case(rows[0]) if rows else None


_ORDER = {"open": "ORDER BY respond_by ASC NULLS LAST, decision_id ASC",
          "closed": "ORDER BY decision_id DESC", "all": "ORDER BY decision_id DESC"}
_PAGE = 500


def load_cases(cur, status: str, *, keep: Optional[Callable[[Dict[str, Any]], bool]] = None,
               limit: int = CASE_LIMIT) -> List[Dict[str, Any]]:
    """Up to `limit` cases that `keep` accepts, filtered by status IN SQL and by the caller's
    visibility BEFORE the limit (paged), so other people's cases never crowd out the caller's.
    Open: soonest deadline first. Closed / all: newest first."""
    where = {"open": "AND status = 'open'", "closed": "AND status <> 'open'", "all": ""}[status]
    out: List[Dict[str, Any]] = []
    offset = 0
    while len(out) < limit:
        cur.execute(f"SELECT {_CASE_COLS} FROM proc.bp_decision WHERE subject_type = %s "
                    f"AND decision = 'approve_or_reject' {where} {_ORDER[status]} LIMIT %s OFFSET %s",
                    (approvals.SUBJECT_TYPE, _PAGE, offset))
        page = [_case(r) for r in _rows(cur)]
        out.extend(c for c in page if keep is None or keep(c))
        if len(page) < _PAGE:
            break
        offset += _PAGE
    return out[:limit]


def sensitive_for(cur, pairs: Iterable[Tuple[str, int]]) -> Set[str]:
    """Union of the named policy versions' sensitive fields and every live policy's."""
    out = _live_sensitive(cur)
    for doc in _compiled(cur, pairs).values():
        out |= _sensitive_of(doc)
    return out


def _pair(case: Dict[str, Any]) -> Tuple[str, Optional[int]]:
    pol = case["facts"].get("policy") or {}
    v = pol.get("version")
    return str(pol.get("id") or case.get("policy_name") or ""), (int(v) if str(v or "").isdigit() else None)


def outcomes(cur, cases: Iterable[Dict[str, Any]]) -> Dict[str, str]:
    """{subject_id: 'approved' | 'rejected' | 'timed_out'} from each closed case's latest
    recorded action (a timeout is recorded as a reject by approvals.TIMEOUT_ACTOR)."""
    ids = sorted({c["subject_id"] for c in cases if c.get("status") != "open" and c.get("subject_id")})
    if not ids:
        return {}
    cur.execute("SELECT DISTINCT ON (subject_id) subject_id, decision, actioned_by FROM proc.bp_decision "
                "WHERE subject_type = %s AND subject_id = ANY(%s) AND decision IN ('approve','reject') "
                "AND actioned_by IS NOT NULL ORDER BY subject_id, decision_id DESC",
                (approvals.SUBJECT_TYPE, ids))
    return {r[0]: ("approved" if r[1] == "approve" else
                   "timed_out" if r[2] == approvals.TIMEOUT_ACTOR else "rejected")
            for r in cur.fetchall()}


# ---------------------------------------------------------------------------- live conflict block
LIVE_SUBJECT_TYPE = "live_conflict"   # conflict_cases.SUBJECT_LIVE; not imported (it pulls in more)


def _live_id(case: Dict[str, Any]) -> Optional[int]:
    v = case["facts"].get("liveConflict")
    return int(v) if isinstance(v, int) and not isinstance(v, bool) else None


def live_cases(cur, cases: Iterable[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """{live decision_id: its row} for the member cases among `cases` (facts.liveConflict)."""
    ids = sorted({i for i in (_live_id(c) for c in cases) if i is not None})
    if not ids:
        return {}
    cur.execute("SELECT decision_id, rationale, facts, evidence, options FROM proc.bp_decision "
                "WHERE decision_id = ANY(%s) AND subject_type = %s", (ids, LIVE_SUBJECT_TYPE))
    out = {}
    for r in _rows(cur):
        r["facts"] = _j(r.get("facts"), {}) or {}
        r["options"] = _j(r.get("options"), []) or []
        out[int(r["decision_id"])] = r
    return out


def live_pairs(lives: Iterable[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """(policy key, version) of every policy a live case involves: their sensitive inputs mask
    the block too, so a field one involved policy marks sensitive is never shown."""
    out = []
    for lv in lives:
        for p in (lv.get("facts") or {}).get("policies") or []:
            v = (p or {}).get("version")
            if (p or {}).get("id") and str(v or "").isdigit():
                out.append((str(p["id"]), int(v)))
    return out


def conflict_block(live_id: int, live: Optional[Dict[str, Any]], *, unmasked: bool,
                   sensitive: Set[str]) -> Dict[str, Any]:
    """The live conflict summary on a member case's card. The action's condition values and the
    overlap example are masked like the case's own inputs unless the caller may decide."""
    live = live or {}
    facts = live.get("facts") or {}
    summary = facts.get("summary") or {}
    action = facts.get("action") or {}
    args = dict(action.get("args") or {})
    example = overlap_example(live.get("evidence"))
    if not unmasked:
        args, example = mask_args(args, sensitive), mask_witness(example, sensitive)
    from services.agent_policy import conflict_history   # lazily: conflict_history reads this module
    history = conflict_history.shown(facts.get("history") or [], sensitive=sensitive,
                                     unmasked_for=lambda _e: unmasked)
    return {"caseId": conflict_payload.case_id(live_id), "why": live.get("rationale") or summary.get("why"),
            "policies": list(facts.get("policies") or []), "prior": facts.get("priorDecisions"),
            "options": list(live.get("options") or []), "respondWithin": summary.get("respondWithin"),
            "actionPlain": action.get("plain"), "args": args, "example": example,
            "history": history, "precedentNote": (facts.get("precedent") or {}).get("why")}


def case_view(case: Dict[str, Any], *, unmasked: bool, sensitive: Set[str],
              doc: Optional[Dict[str, Any]], decidable: bool,
              outcome: Optional[str] = None, live: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`live`: the live conflict row this case is a member of (facts.liveConflict), if any."""
    facts = case["facts"]
    pol = facts.get("policy") or {}
    action = facts.get("action") or {}
    ctx = _ctx(facts)
    named = {str(i.get("field")): i for i in (doc or {}).get("inputs") or [] if isinstance(i, dict)}
    inputs = []
    for f in facts.get("approvalInputs") or []:
        f = str(f)
        val = _lookup(ctx, f)
        masked = f in sensitive and not unmasked
        row = {"field": f, "name": (named.get(f) or {}).get("name") or f,
               "value": MASK if masked else (None if val is _ABSENT else val),
               "missing": val is _ABSENT, "masked": masked}
        if (named.get(f) or {}).get("unit"):
            row["unit"] = named[f]["unit"]
        inputs.append(row)
    reason = action.get("reason")
    if reason and not unmasked:
        if "agent.reason" in sensitive:
            reason = MASK
        elif isinstance(reason, str):
            # the reason is free text: any sensitive argument value it quotes is masked in place
            args = {f[len("args."):] for f in sensitive if f.startswith("args.")}
            reason = mask_text(reason, dict(action.get("args") or {}), args)
    levels = case["levels"]
    level = case["current_level"]
    view = {
        "id": int(case["decision_id"]),
        "policyKey": pol.get("id") or case.get("policy_name"),
        "policyVersion": pol.get("version"),
        "status": case["status"],
        # what was decided (None while open); status 'actioned' alone does not say
        "outcome": outcome,
        "actionPlain": facts.get("actionPlain"),
        "inputs": inputs,
        "agentReason": reason,
        "situation": pol.get("situation"),
        "excerpt": pol.get("excerpt"),
        "reference": pol.get("reference"),
        "document": pol.get("document"),
        "level": level,
        "levelName": _level_name(case),
        "levels": [str((lv or {}).get("name") or "") for lv in levels],
        "respondBy": _iso(case.get("respond_by")),
        "onTimeout": case.get("on_timeout"),
        "options": case["options"],
        "reasonRequiredOn": ["reject"],
        "unroutable": list(facts.get("unroutable") or []),
        "requestedBy": facts.get("requestedBy"),
        "createdAt": _iso(case.get("created_at")),
        "canDecide": bool(decidable and case["status"] == "open"),
    }
    live_id = _live_id(case)
    if live_id is not None:
        view["conflict"] = conflict_block(live_id, live, unmasked=unmasked, sensitive=sensitive)
    return view


def list_cases(conn, principal, *, is_admin: bool, status: str = "open") -> List[Dict[str, Any]]:
    mapping = deciders.load_map(conn)
    with conn.cursor() as cur:
        if status == "open":
            shown = load_cases(cur, status, keep=lambda c: is_admin or can_decide(principal, c, mapping))
        else:
            shown = load_cases(cur, status, keep=lambda c: may_read(principal, c, mapping, is_admin=is_admin))
        pairs = [p for p in (_pair(c) for c in shown) if p[1] is not None]
        lives = live_cases(cur, shown)
        sensitive = sensitive_for(cur, pairs + live_pairs(lives.values()))
        docs = _compiled(cur, pairs)
        decided = outcomes(cur, shown)
    out = []
    for c in shown:
        ok = can_decide(principal, c, mapping)
        out.append(case_view(c, unmasked=ok, sensitive=sensitive, doc=docs.get(_pair(c)), decidable=ok,
                             outcome=decided.get(c["subject_id"]), live=lives.get(_live_id(c))))
    return out


def _firing_view(row: Dict[str, Any], sensitive: Set[str],
                 reveal: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """`reveal`: the case's context, for a caller who may decide it. The log stores sensitive
    values already masked; the eligible approver sees them from the case, as in `inputs`."""
    values = _mask_values(_j(row["matched_values"], {}) or {}, sensitive)
    if reveal is not None:
        for k in list(values):
            if values[k] == MASK and (v := _lookup(reveal, k)) is not _ABSENT:
                values[k] = v
    return {
        "id": int(row["firing_id"]),
        "policyKey": row["policy_key"],
        "policyVersion": row["policy_version"],
        "checkpoint": row["checkpoint"],
        "actionName": row["action_name"],
        "agent": row["agent"],
        "workflowId": row["workflow_id"],
        "requestedBy": row["requested_by"],
        "outcome": row["outcome"],
        "result": row["result"],
        "matchedValues": values,
        "missingInputs": list(row["missing_inputs"] or []),
        "decisionId": row["decision_id"],
        "decidedLevel": row["decided_level"],
        "decidedBy": row["decided_by"],
        "decidedAt": _iso(row["decided_at"]),
        "reason": row["reason"],
        "durationMs": row["duration_ms"],
        "reversalOf": row["reversal_of"],
        "createdAt": _iso(row["created_at"]),
    }


_FIRING_COLS = ("firing_id, policy_key, policy_version, checkpoint, action_name, agent, workflow_id, "
                "requested_by, outcome, result, matched_values, missing_inputs, decision_id, decided_level, "
                "decided_by, decided_at, reason, duration_ms, reversal_of, created_at")


def readable(conn, decision_id: int, principal, *, is_admin: bool) -> bool:
    """Whether the caller may read this case (the GET's rule); False when it does not exist."""
    with conn.cursor() as cur:
        case = load_case(cur, decision_id)
    return case is not None and may_read(principal, case, deciders.load_map(conn), is_admin=is_admin)


def get_case(conn, decision_id: int, principal, *, is_admin: bool) -> Optional[Dict[str, Any]]:
    """One case with its history, or None when it does not exist OR the caller may not read it."""
    mapping = deciders.load_map(conn)
    with conn.cursor() as cur:
        case = load_case(cur, decision_id)
        if case is None or not may_read(principal, case, mapping, is_admin=is_admin):
            return None
        facts = case["facts"]
        cur.execute("SELECT decision_id, decision, actioned_by, actioned_at, override_reason, current_level, facts "
                    "FROM proc.bp_decision WHERE subject_type = %s AND subject_id = %s "
                    "AND decision IN ('approve','reject') AND actioned_by IS NOT NULL ORDER BY decision_id",
                    (approvals.SUBJECT_TYPE, case["subject_id"]))
        decisions = [{"id": int(r["decision_id"]), "verb": r["decision"], "by": r["actioned_by"],
                      "at": _iso(r["actioned_at"]), "reason": r["override_reason"],
                      "level": r["current_level"],
                      "levelName": (_j(r["facts"], {}) or {}).get("decidedLevelName"),
                      "timeout": r["actioned_by"] == approvals.TIMEOUT_ACTOR}
                     for r in _rows(cur)]
        cur.execute("SELECT notification_id, recipient, message, created_at FROM proc.bp_policy_notification "
                    "WHERE link = %s ORDER BY created_at, notification_id", (f"decision:{decision_id}",))
        notes = [{"id": int(r["notification_id"]), "recipient": r["recipient"], "message": r["message"],
                  "createdAt": _iso(r["created_at"])} for r in _rows(cur)]
        cur.execute(f"SELECT {_FIRING_COLS} FROM proc.bp_policy_firing WHERE decision_id = %s "
                    "OR firing_id = %s ORDER BY created_at, firing_id",
                    (decision_id, facts.get("firingId") if str(facts.get("firingId") or "").isdigit() else None))
        firing_rows = _rows(cur)
        replay = latest_replay(cur, case)
        pairs = [_pair(case)] if _pair(case)[1] is not None else []
        pairs += [(r["policy_key"], r["policy_version"]) for r in firing_rows]
        lives = live_cases(cur, [case])
        sensitive = sensitive_for(cur, pairs + live_pairs(lives.values()))
        docs = _compiled(cur, pairs[:1])
        decided = outcomes(cur, [case])
        group = approvals.group_state(cur, int(case["decision_id"]), facts)
    ok = can_decide(principal, case, mapping)
    view = case_view(case, unmasked=ok, sensitive=sensitive, doc=docs.get(_pair(case)), decidable=ok,
                     outcome=decided.get(case["subject_id"]), live=lives.get(_live_id(case)))
    view["history"] = {"decisions": decisions, "notes": notes,
                       "firings": [_firing_view(r, sensitive, reveal=_ctx(case["facts"]) if ok else None)
                                   for r in firing_rows],
                       "replay": replay,
                       # every approval the action needs: {open, approved, rejected, total}
                       "group": group}
    return view


REPLAY_SUBJECT_TYPE = "agent_policy_replay"   # replay.SUBJECT_TYPE; not imported (pulls in the tools)


def latest_replay(cur, case: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The latest replay of this case's group: a replay row names every case it ran for in
    facts.caseIds (and the group in facts.firingGroup). resultSummary and error were masked when
    the replay wrote them. None when the action has not been replayed."""
    group = case["facts"].get("firing_group") or case["facts"].get("firingGroup")
    cur.execute("SELECT facts, actioned_at FROM proc.bp_decision WHERE subject_type = %s "
                "AND (facts->'caseIds' @> %s::jsonb OR (%s::text IS NOT NULL AND facts->>'firingGroup' = %s)) "
                "ORDER BY decision_id DESC LIMIT 1",
                (REPLAY_SUBJECT_TYPE, json.dumps([int(case["decision_id"])]), group, group))
    row = cur.fetchone()
    if not row:
        return None
    facts = _j(row[0], {}) or {}
    return {"outcome": facts.get("outcome"), "resultSummary": facts.get("resultSummary"),
            "error": facts.get("error"), "at": _iso(row[1])}


# ---------------------------------------------------------------------------- firings
def policy_firings(conn, policy_key: str, limit: int) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {_FIRING_COLS} FROM proc.bp_policy_firing WHERE policy_key = %s "
                    "ORDER BY created_at DESC, firing_id DESC LIMIT %s", (policy_key, limit))
        rows = _rows(cur)
        sensitive = sensitive_for(cur, [(r["policy_key"], r["policy_version"]) for r in rows])
    return [_firing_view(r, sensitive) for r in rows]


# ---------------------------------------------------------------------------- notifications
def _link_ref(link: str) -> Dict[str, Any]:
    """The record a notification points at, as ids (never a route)."""
    kind, _, ref = str(link or "").partition(":")
    if kind == "decision" and ref.isdigit():
        return {"decisionId": int(ref)}
    if kind == "agent-policy" and KEY_RE.match(ref):
        return {"policyKey": ref}
    if kind == "conflict" and ref.isascii() and ref.isdigit():
        return {"conflictId": int(ref)}
    return {}


def _recipients(conn, principal, *, is_admin: bool = False) -> List[str]:
    names = set(caller_names(principal, deciders.load_map(conn))) | set(_identities(principal))
    if is_admin:
        names.add(approvals.ADMIN_RECIPIENT)   # "cannot be routed" notices go to every Admin
    return sorted(names)


def my_notifications(conn, principal, limit: int, *, is_admin: bool = False) -> List[Dict[str, Any]]:
    names = _recipients(conn, principal, is_admin=is_admin)
    if not names:
        return []
    with conn.cursor() as cur:
        cur.execute("SELECT n.notification_id, n.recipient, n.message, n.link, n.read_by, n.created_at, "
                    "f.policy_key FROM proc.bp_policy_notification n "
                    "LEFT JOIN proc.bp_policy_firing f ON f.firing_id = n.firing_id "
                    "WHERE n.recipient = ANY(%s) ORDER BY n.created_at DESC, n.notification_id DESC LIMIT %s",
                    (names, limit))
        rows = _rows(cur)
    me = str(getattr(principal, "subject", "") or "")
    out = []
    for r in rows:
        ref = _link_ref(r["link"])
        out.append({"id": int(r["notification_id"]), "recipient": r["recipient"], "message": r["message"],
                    "policyKey": ref.get("policyKey") or r["policy_key"], "decisionId": ref.get("decisionId"),
                    "conflictId": ref.get("conflictId"),
                    "read": me in (r["read_by"] or []), "createdAt": _iso(r["created_at"])})
    return out


def mark_read(conn, notification_id: int, principal, *, is_admin: bool = False) -> Optional[Dict[str, Any]]:
    """Mark read for the caller; None when it is not one of theirs. Idempotent."""
    names = _recipients(conn, principal, is_admin=is_admin)
    me = str(getattr(principal, "subject", "") or "")
    with conn.cursor() as cur:
        cur.execute("SELECT notification_id FROM proc.bp_policy_notification "
                    "WHERE notification_id = %s AND recipient = ANY(%s)", (notification_id, names))
        if cur.fetchone() is None:
            return None
        cur.execute("UPDATE proc.bp_policy_notification SET read_by = array_append(read_by, %s) "
                    "WHERE notification_id = %s AND NOT (%s = ANY(read_by))", (me, notification_id, me))
    return {"id": notification_id, "read": True}


# ---------------------------------------------------------------------------- deciders
def list_deciders(conn) -> List[Dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute("SELECT decider_name, groups, emails, notes, last_modified_by, last_modified_at "
                    "FROM proc.bp_policy_decider_map ORDER BY decider_name")
        return [{"name": r[0], "groups": list(r[1] or []), "emails": list(r[2] or []), "notes": r[3],
                 "lastModifiedBy": r[4], "lastModifiedAt": _iso(r[5])} for r in cur.fetchall()]


def decider_problems(name: str, groups: Any, emails: Any, notes: Any) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Validate a decider-map row. Returns (problems, clean_groups, clean_emails)."""
    problems: List[Dict[str, Any]] = []
    if (name or "").strip().casefold() == approvals.ADMIN_RECIPIENT.casefold():
        # the fixed recipient of "cannot be routed" notices, read by every Admin: a mapping under
        # this name would let its non-Admin members read them
        problems.append({"field": "name", "code": "reserved_name",
                         "message": f"{approvals.ADMIN_RECIPIENT} is reserved for Admins; choose another name."})
    elif not DECIDER_NAME_RE.match(name or ""):
        problems.append({"field": "name", "code": "invalid",
                         "message": "A decider name starts with a letter and uses letters, numbers, spaces "
                                    "and & , ' . - only (64 characters at most)."})
    clean_groups: List[str] = []
    if not isinstance(groups, list) or any(not isinstance(g, str) or not g.strip() for g in groups):
        problems.append({"field": "groups", "code": "invalid", "message": "Every group must be a non-empty name."})
    else:
        for g in groups:
            g = g.strip()
            if len(g) > 128:
                problems.append({"field": "groups", "code": "invalid",
                                 "message": "A group name is 128 characters at most."})
                break
            if g not in clean_groups:
                clean_groups.append(g)
    clean_emails: List[str] = []
    if not isinstance(emails, list) or any(not isinstance(e, str) for e in emails):
        problems.append({"field": "emails", "code": "invalid", "message": "Every email must be text."})
    else:
        bad = [e for e in emails if not EMAIL_RE.match(e.strip()) or len(e.strip()) > 254]
        if bad:
            problems.append({"field": "emails", "code": "invalid",
                             "message": f"{len(bad)} email address(es) are not in the form name@example.com."})
        else:
            for e in emails:
                e = e.strip().lower()
                if e not in clean_emails:
                    clean_emails.append(e)
    if not problems and not clean_groups and not clean_emails:
        problems.append({"field": "groups", "code": "someone_required",
                         "message": "Link at least one group or one email address."})
    if notes is not None and (not isinstance(notes, str) or len(notes) > 1000):
        problems.append({"field": "notes", "code": "invalid", "message": "Notes are text, 1000 characters at most."})
    return problems, clean_groups, clean_emails


def upsert_decider(conn, name: str, groups: List[str], emails: List[str], notes: Optional[str],
                   actor: str) -> Dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO proc.bp_policy_decider_map (decider_name, groups, emails, notes, last_modified_by, "
            "last_modified_at) VALUES (%s, %s, %s, %s, %s, now()) "
            "ON CONFLICT (decider_name) DO UPDATE SET groups = EXCLUDED.groups, emails = EXCLUDED.emails, "
            "notes = EXCLUDED.notes, last_modified_by = EXCLUDED.last_modified_by, last_modified_at = now() "
            "RETURNING decider_name, groups, emails, notes, last_modified_by, last_modified_at",
            (name, groups, emails, notes, actor))
        r = cur.fetchone()
    return {"name": r[0], "groups": list(r[1] or []), "emails": list(r[2] or []), "notes": r[3],
            "lastModifiedBy": r[4], "lastModifiedAt": _iso(r[5])}
