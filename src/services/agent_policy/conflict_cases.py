"""Policy conflict cases: raise one when two policies contradict, at most one open per pair.

A case is a proc.bp_decision row (subject_type 'policy_conflict', subject_id = the pair key)
plus its index row in proc.bp_agent_policy_conflict. Detection runs when a policy is saved,
when one is extracted from a document, and in an hourly scan (detect_all). Nothing here ever
changes a policy: the owners decide, and their own save applies the decision (Task 5).

A pair is raised only when code finds an input both policies match (conflict_detect.witness).
Dedup, under a per-pair advisory lock taken before any check:
- the pair already has an open policy case;
- an in-force standing rule covers the pair;
- a decided case already covered these versions (nothing changed since it was decided).
The partial unique index on open policy pairs is the backstop: tripping it is a bug, and it
raises and rolls the whole transaction back.

Deciding (decide_policy): anyone linked to EITHER owner name decides (Admin is not automatic);
one decision closes the case. keep_both writes only a standing-rule row (the pair's previous rule
is superseded, superseded_at and superseded_by always set together); change and limit write no
version (the owner's own save applies them); retire waits for the normal two-step retire. A
retired policy's open cases close as moot (close_moot). Standing rules reach each live policy's
conflicts[] at read time (overlay), because version rows are immutable.

get_conn() is AUTOCOMMIT: every write runs in approvals._tx.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from services.agent_policy import approvals, conflict_detect, conflict_payload, deciders
from services.agent_policy import settings as _settings
from services.policy_condition import ConditionError

logger = logging.getLogger(__name__)

SUBJECT_POLICY = "policy_conflict"
SUBJECT_LIVE = "live_conflict"
DETECTOR = "system:conflict_detector"
_AGENT = "agent_policy_conflicts"
_NO_OWNER = "(no owner named)"

#: Test seam: called inside raise_policy_case right after the pair's advisory lock is taken.
_after_lock: Optional[Callable[[str], None]] = None


def _j(v: Any, default=None):
    if v is None:
        return default
    if isinstance(v, (str, bytes)):
        return json.loads(v)
    return v


def _version(doc: Dict[str, Any]) -> int:
    try:
        return int(doc.get("version") or 0)
    except (TypeError, ValueError):
        return 0


def _owners(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for d in (a, b):
        name = str(d.get("owner") or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _unroutable(a: Dict[str, Any], b: Dict[str, Any], mapping) -> List[str]:
    missing = deciders.unmapped(_owners(a, b), mapping)
    if any(not str(d.get("owner") or "").strip() for d in (a, b)):
        missing.append(_NO_OWNER)
    return missing


def _notify(cur, recipients: Iterable[str], message: str, decision_id: int) -> None:
    for r in recipients:
        r = str(r or "").strip()
        if r:
            cur.execute(
                "INSERT INTO proc.bp_policy_notification (firing_id, recipient, message, link) "
                "VALUES (NULL, %s, %s, %s)",
                (r, message, f"conflict:{decision_id}"),
            )


def _covered(cur, key: str, versions: Dict[str, int]) -> bool:
    """The three dedup checks, run after the pair's lock."""
    cur.execute("SELECT 1 FROM proc.bp_agent_policy_conflict WHERE kind = 'policy' AND pair_key = %s AND is_open",
                (key,))
    if cur.fetchone():
        return True
    cur.execute("SELECT 1 FROM proc.bp_agent_policy_conflict_rule WHERE pair_key = %s AND superseded_at IS NULL",
                (key,))
    if cur.fetchone():
        return True
    (ka, va), (kb, vb) = sorted(versions.items())
    cur.execute(
        """
        SELECT 1 FROM proc.bp_agent_policy_conflict
         WHERE kind = 'policy' AND pair_key = %s AND NOT is_open
           AND (policy_versions->>%s)::int >= %s AND (policy_versions->>%s)::int >= %s
         LIMIT 1
        """,
        (key, ka, va, kb, vb),
    )
    return cur.fetchone() is not None


def _prior(cur, key: str) -> Dict[str, Any]:
    cur.execute(
        "SELECT count(*), (array_agg(outcome ORDER BY decided_at DESC))[1] FROM proc.bp_agent_policy_conflict "
        "WHERE kind = 'policy' AND pair_key = %s AND NOT is_open",
        (key,),
    )
    row = cur.fetchone() or (0, None)
    return {"sameConflict": int(row[0] or 0), "lastOutcome": row[1]}


def _standing_rules(cur, keys: List[str]) -> List[Dict[str, Any]]:
    """In-force rules touching either policy (never this pair: a rule for the pair stops the raise)."""
    cur.execute(
        "SELECT pair_key, prevails, yields, rule_text, decision_id, decided_at "
        "FROM proc.bp_agent_policy_conflict_rule WHERE superseded_at IS NULL "
        "AND (prevails = ANY(%s) OR yields = ANY(%s)) ORDER BY rule_id",
        (keys, keys),
    )
    return [{"pairKey": r[0], "prevails": r[1], "yields": r[2], "rule": r[3],
             "caseId": conflict_payload.case_id(r[4]),
             "decidedAt": r[5].isoformat() if hasattr(r[5], "isoformat") else r[5]}
            for r in cur.fetchall()]


def raise_policy_case(cur, a: Dict[str, Any], b: Dict[str, Any], example: Dict[str, Any], *, raised_by: str,
                      now: datetime, mapping, proposal: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Raise one policy case for the pair inside the CALLER's transaction. None when deduplicated."""
    ka, kb = str(a.get("id")), str(b.get("id"))
    key = conflict_detect.pair_key(ka, kb)
    keys = key.split("|")
    versions = {ka: _version(a), kb: _version(b)}
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"agent_policy_conflict:{key}",))
    if _after_lock:
        _after_lock(key)
    if _covered(cur, key, versions):
        return None

    first, second = (a, b) if ka == keys[0] else (b, a)
    payload = conflict_payload.build(
        "policy", raised_at=now.isoformat(),
        policies=[conflict_payload.policy_entry(first), conflict_payload.policy_entry(second)],
        overlap_example=dict(example), standing_rules=_standing_rules(cur, keys), prior=_prior(cur, key),
        options=conflict_payload.policy_options(first, second), respond_within=None, on_timeout="none")
    cols = conflict_payload.to_columns(payload)
    facts = dict(cols["facts"])
    missing = _unroutable(first, second, mapping)
    if missing:
        facts["unroutable"] = missing
    if proposal is not None:
        facts["proposal"] = dict(proposal)

    cur.execute(
        """
        INSERT INTO proc.bp_decision (
            subject_type, subject_id, decision, resolution, rationale, status,
            policy_id, policy_name, facts, evidence, workflow_id, agent, created_by,
            options, respond_by, on_timeout, decision_scope
        ) VALUES (%s,%s,'resolve_conflict','escalated',%s,'open',
                  NULL,%s,%s,%s,NULL,%s,%s,%s,NULL,%s,NULL)
        RETURNING decision_id
        """,
        (cols["subject_type"], key, conflict_payload.why_line([first, second]), key,
         json.dumps(facts, default=str), json.dumps(cols["evidence"], default=str), _AGENT, DETECTOR,
         json.dumps(cols["options"]), cols["on_timeout"]),
    )
    decision_id = int(cur.fetchone()[0])
    cur.execute(
        "INSERT INTO proc.bp_agent_policy_conflict (decision_id, kind, pair_key, policy_keys, policy_versions, "
        "raised_by) VALUES (%s, 'policy', %s, %s, %s, %s)",
        (decision_id, key, keys, json.dumps(versions), raised_by),
    )
    message = f"Policies {keys[0]} and {keys[1]} conflict; a decision is needed."
    _notify(cur, _owners(first, second), message, decision_id)
    if missing:
        _notify(cur, [approvals.ADMIN_RECIPIENT],
                f"Policies {keys[0]} and {keys[1]} conflict and the case cannot be routed: link {', '.join(missing)}",
                decision_id)
    return decision_id


# ---------------------------------------------------------------------------- detection
def _examples(form: Any) -> List[Dict[str, Any]]:
    form = _j(form, {}) or {}
    out = []
    for ex in form.get("examples") or []:
        inp = ex.get("input") if isinstance(ex, dict) else None
        if isinstance(inp, dict):
            out.append(inp)
    return out


def _load(cur, policy_key: str, among: Optional[List[str]]):
    """(saved latest doc, its example inputs, [other docs]) or None when there is nothing to check."""
    cur.execute(
        "SELECT p.status, v.compiled, v.form_state FROM proc.bp_agent_policy p "
        "JOIN proc.bp_agent_policy_version v ON v.policy_key = p.policy_key AND v.version = p.latest_version "
        "WHERE p.policy_key = %s",
        (policy_key,),
    )
    row = cur.fetchone()
    if not row or row[0] == "retired":
        return None
    saved = _j(row[1], {}) or {}
    sql = ("SELECT v.compiled FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v "
           "ON v.policy_key = p.policy_key AND v.version IN (p.live_version, p.latest_version) "
           "WHERE p.status <> 'retired' AND p.policy_key <> %s")
    params: List[Any] = [policy_key]
    if among is not None:
        sql += " AND p.policy_key = ANY(%s)"
        params.append(list(among))
    cur.execute(sql + " ORDER BY p.policy_key, v.version DESC", params)
    others = [d for d in (_j(r[0], {}) for r in cur.fetchall()) if isinstance(d, dict)]
    return saved, _examples(row[2]), others


def _pairs(saved, examples, others, stats) -> List[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Every (pair_key, other, witness) whose policies contradict, latest version first per pair."""
    found = []
    for other in others:
        if not conflict_detect.design_time_pair(saved, other):
            continue
        stats["pairs"] += 1
        try:
            w = conflict_detect.witness(saved, other, examples)
        except ConditionError as exc:
            stats["errors"] += 1
            logger.warning("conflict check skipped %s vs %s: %s", saved.get("id"), other.get("id"),
                           type(exc).__name__)
            continue
        if w is not None:
            found.append((conflict_detect.pair_key(str(saved.get("id")), str(other.get("id"))), other, w))
    # one global lock order (pair key) so concurrent detections never deadlock; within a pair the
    # latest version goes first, so its versions are the ones recorded
    found.sort(key=lambda t: (t[0], -_version(t[1])))
    return found


def cap_of(settings: Dict[str, Any]) -> int:
    """The company's cap on NEW cases per detection call (and per whole scan)."""
    try:
        cap = int((settings or {}).get("conflict_cases_per_run"))
    except (TypeError, ValueError):
        cap = 0
    return cap if cap > 0 else int(_settings.DEFAULTS["conflict_cases_per_run"])


def _cap_reached(cap: int) -> None:
    logger.warning("conflict case cap %d reached; remaining pairs wait for the next scan", cap)


def _raise_pairs(cur, saved, examples, others, *, now, mapping, raised_by, stats, budget) -> List[int]:
    """Raise the saved policy's pairs while budget["left"] lasts. Sets stats["capped"] when a pair
    was left unraised because the budget ran out (dedup leaves it to a later scan)."""
    raised: List[int] = []
    for _key, other, w in _pairs(saved, examples, others, stats):
        if budget["left"] <= 0:
            stats["capped"] = True
            break
        did = raise_policy_case(cur, saved, other, w, raised_by=raised_by, now=now, mapping=mapping)
        if did is not None:
            raised.append(did)
            budget["left"] -= 1
    stats["raised"] += len(raised)
    return raised


def detect_for(conn, policy_key: str, *, now: Optional[datetime] = None, among: Optional[List[str]] = None,
               raised_by: str = "save") -> List[int]:
    """Raise a case for every policy the saved one contradicts, at most the company cap of NEW cases.
    One transaction. Returns the new case ids."""
    now = now or datetime.now(timezone.utc)
    stats = {"pairs": 0, "raised": 0, "errors": 0, "capped": False}
    cap = cap_of(_settings.load_settings(conn))
    mapping = deciders.load_map(conn)
    with approvals._tx(conn):
        with conn.cursor() as cur:
            loaded = _load(cur, policy_key, among)
            if loaded is None:
                return []
            saved, examples, others = loaded
            raised = _raise_pairs(cur, saved, examples, others, now=now, mapping=mapping, raised_by=raised_by,
                                  stats=stats, budget={"left": cap})
    if stats["capped"]:
        _cap_reached(cap)
    return raised


def _snapshot(conn, among: Optional[List[str]] = None):
    """Every non-retired policy, read ONCE per scan: ({key: (latest doc, example inputs)}, [(key, doc)])
    where the list holds each policy's live and latest versions, latest first."""
    sql = ("SELECT p.policy_key, v.compiled, v.form_state, v.version = p.latest_version "
           "FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v "
           "ON v.policy_key = p.policy_key AND v.version IN (p.live_version, p.latest_version) "
           "WHERE p.status <> 'retired'")
    params: List[Any] = []
    if among is not None:
        sql += " AND p.policy_key = ANY(%s)"
        params.append(list(among))
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY p.policy_key, v.version DESC", params)
        rows = cur.fetchall()
    latest: Dict[str, Tuple[Dict[str, Any], List[Dict[str, Any]]]] = {}
    docs: List[Tuple[str, Dict[str, Any]]] = []
    for key, compiled, form, is_latest in rows:
        doc = _j(compiled, {})
        if not isinstance(doc, dict):
            continue
        docs.append((key, doc))
        if is_latest:
            latest[key] = (doc, _examples(form))
    return latest, docs


def detect_all(conn, *, now: Optional[datetime] = None, among: Optional[List[str]] = None) -> Dict[str, Any]:
    """The hourly safety net: every non-retired policy against all others, at most the company cap
    of NEW cases per whole scan. Policies, the decider map and settings are read once; each policy's
    raises run in their own transaction. Never raises. `among` narrows the scan (tests)."""
    now = now or datetime.now(timezone.utc)
    stats: Dict[str, Any] = {"pairs": 0, "raised": 0, "errors": 0, "capped": False, "mooted": 0}
    try:
        cap = cap_of(_settings.load_settings(conn))
        mapping = deciders.load_map(conn)
        latest, docs = _snapshot(conn, among)
    except Exception as exc:  # noqa: BLE001
        logger.error("conflict scan could not load policies: %s", type(exc).__name__)
        stats["errors"] += 1
        return stats
    _close_moot_backstop(conn, now=now, among=among, stats=stats)
    budget = {"left": cap}
    for key in sorted(latest):
        saved, examples = latest[key]
        others = [d for k, d in docs if k != key]
        try:
            with approvals._tx(conn):
                with conn.cursor() as cur:
                    _raise_pairs(cur, saved, examples, others, now=now, mapping=mapping, raised_by="scan",
                                 stats=stats, budget=budget)
        except Exception as exc:  # noqa: BLE001 - one policy must not stop the scan
            stats["errors"] += 1
            logger.error("conflict scan failed for %s: %s", key, type(exc).__name__)
        if stats["capped"]:
            _cap_reached(cap)
            break
    return stats


def after_save(conn, policy_key: str) -> None:
    """Best effort after a save: the save is never undone, and the hourly scan catches what this missed."""
    key = policy_key
    try:
        detect_for(conn, key)
    except Exception as exc:  # noqa: BLE001
        logger.error("conflict detection failed for %s: %s", key, type(exc).__name__)


# ---------------------------------------------------------------------------- deciding
#: Who closes a case whose policy was retired, and the action scope of every non-rule outcome.
RETIRED_ACTOR = "system:retired"
MOOT = "moot"
_APPLIED = {"keep_both": "standing_rule", "change": "draft_pending", "limit": "draft_pending",
            "retire": "retire_pending"}
_PENDING_VERBS = ("change", "limit", "retire")
_LABELS = {"keep_both": "Keep both: {} takes priority", "change": "Change {}", "limit": "Limit {}",
           "retire": "Retire {}"}


class ConflictRefused(Exception):
    """A refused decision. `status` is the HTTP status a router should answer with
    (the same shape as approvals.ApprovalRefused)."""

    def __init__(self, code: str, message: str, status: int = 403):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


def option_label(option: str) -> str:
    """The words a person reads for an option (the screens use the same ones)."""
    verb, _, key = str(option).partition(":")
    return _LABELS[verb].format(key) if verb in _LABELS else str(option)


def _iso(v: Any) -> Any:
    if isinstance(v, datetime):
        return (v if v.tzinfo else v.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat()
    return v


_POLICY_CASE_SQL = """
    SELECT decision_id, subject_id, resolution, rationale, policy_name, facts, status, options
      FROM proc.bp_decision
     WHERE decision_id = %s AND subject_type = %s
"""


def _case(cur, decision_id: int, *, lock: bool) -> Optional[Dict[str, Any]]:
    cur.execute(_POLICY_CASE_SQL + (" FOR UPDATE" if lock else ""), (decision_id, SUBJECT_POLICY))
    row = cur.fetchone()
    if not row:
        return None
    case = dict(zip([d[0] for d in cur.description], row))
    case["facts"] = _j(case["facts"], {}) or {}
    case["options"] = _j(case["options"], []) or []
    return case


def _case_owners(facts: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for p in facts.get("policies") or []:
        name = str((p or {}).get("owner") or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _record_action(cur, case: Dict[str, Any], *, decision: str, actor: str, now: datetime, reason: Optional[str],
                   scope: str, facts: Dict[str, Any]) -> int:
    """The action row and the original's close, the way approvals._record does it."""
    cur.execute(
        """
        INSERT INTO proc.bp_decision (
            subject_type, subject_id, decision, resolution, rationale, policy_id, policy_name,
            facts, evidence, status, actioned_by, actioned_at, override_reason,
            workflow_id, agent, created_by, decision_scope
        ) VALUES (%s,%s,%s,%s,%s,NULL,%s,%s,'[]','actioned',%s,%s,%s,NULL,%s,%s,%s)
        RETURNING decision_id
        """,
        (SUBJECT_POLICY, case["subject_id"], decision, case["resolution"], case["rationale"], case["policy_name"],
         json.dumps(facts, default=str), actor, now, reason, _AGENT, actor, scope),
    )
    action_id = int(cur.fetchone()[0])
    cur.execute("UPDATE proc.bp_decision SET status = 'actioned' WHERE decision_id = %s AND subject_type = %s",
                (case["decision_id"], SUBJECT_POLICY))
    return action_id


def _close_conflict(cur, decision_id: int, *, outcome: str, actor: str, now: datetime, by_person: bool) -> None:
    cur.execute(
        "UPDATE proc.bp_agent_policy_conflict SET is_open = false, outcome = %s, decided_by = %s, decided_at = %s, "
        "by_person = %s WHERE decision_id = %s AND is_open",
        (outcome, actor, now, by_person, decision_id),
    )


def _latest_versions(cur, keys: List[str]) -> Dict[str, int]:
    cur.execute("SELECT policy_key, latest_version FROM proc.bp_agent_policy WHERE policy_key = ANY(%s)", (keys,))
    return {k: int(v) for k, v in cur.fetchall()}


def _write_rule(cur, *, pair: str, prevails: str, yields: str, case_decision_id: int, action_id: int,
                actor: str, now: datetime) -> None:
    """Supersede the pair's in-force rule (both columns, always together) and insert the new one."""
    cur.execute(
        "UPDATE proc.bp_agent_policy_conflict_rule SET superseded_at = %s, superseded_by = %s "
        "WHERE pair_key = %s AND superseded_at IS NULL",
        (now, action_id, pair),
    )
    cur.execute(
        "INSERT INTO proc.bp_agent_policy_conflict_rule (pair_key, prevails, yields, rule_text, decision_id, "
        "decided_by, decided_at) VALUES (%s, %s, %s, %s, %s, %s, %s)",
        (pair, prevails, yields, f"{prevails} takes priority over {yields}", case_decision_id, actor, now),
    )


def _retired_of(cur, keys: List[str]) -> Optional[str]:
    """The first of the keys whose policy is retired, if any."""
    cur.execute("SELECT policy_key FROM proc.bp_agent_policy WHERE policy_key = ANY(%s) AND status = 'retired' "
                "ORDER BY policy_key LIMIT 1", (list(keys),))
    row = cur.fetchone()
    return row[0] if row else None


def _invalidate_enforcement() -> None:
    from services.agent_policy import live_policies   # lazily: live_policies imports the repo, which imports us
    live_policies.invalidate()


def decide_policy(conn, decision_id: int, *, principal, option: str, reason: Optional[str],
                  limit_text: Optional[str], now: datetime, mapping=None) -> Dict[str, Any]:
    """An owner decides a policy case. Writes the action row, closes the case, and for keep_both the
    standing rule; change, limit and retire write NO policy version (the owner's own save or the
    two-step retire applies them). Raises ConflictRefused when not allowed."""
    reason = (reason or "").strip() or None
    limit_text = (limit_text or "").strip() or None
    option = str(option or "")
    actor = getattr(principal, "subject", None)
    if not actor:
        raise ConflictRefused("not_signed_in", "Sign in to decide.", 401)
    if mapping is None:
        mapping = deciders.load_map(conn)

    def _permitted(case: Optional[Dict[str, Any]]) -> None:
        if case is None:
            raise ConflictRefused("not_found", "No such conflict case.", 404)
        if option not in case["options"]:
            raise ConflictRefused("unknown_option", "That is not one of this case's options.", 422)
        if not reason:
            raise ConflictRefused("reason_required", "A decision needs a reason.", 422)
        if option.startswith("limit:") and not limit_text:
            raise ConflictRefused("limit_required", "Say how the policy should be limited.", 422)
        if case["status"] != "open":
            raise ConflictRefused("not_open", "This conflict has already been decided.", 409)
        owners = _case_owners(case["facts"])
        if not any(deciders.eligible(principal, o, mapping) for o in owners):
            raise ConflictRefused("not_eligible",
                                  "Only someone linked to one of the policies' owners can decide this.", 403)

    verb, _, chosen = option.partition(":")
    retired: Optional[str] = None
    with approvals._tx(conn):
        with conn.cursor() as cur:
            _permitted(_case(cur, decision_id, lock=False))      # refused callers never take the lock
            case = _case(cur, decision_id, lock=True)
            _permitted(case)                                     # it may have been decided meanwhile
            pair = str(case["subject_id"])
            keys = pair.split("|")
            retired = _retired_of(cur, keys)
            if retired:
                # retired behind the case's back (its moot close failed): close it as moot in this
                # transaction, commit, then refuse below; no decision and no rule is written
                _moot(cur, case, retired, now=now)
            else:
                facts = dict(case["facts"])
                facts["caseId"] = conflict_payload.case_id(decision_id)
                facts["limitText"] = limit_text if verb == "limit" else None
                facts["versionsAtDecision"] = _latest_versions(cur, keys)
                scope = conflict_payload.scope_of(option)
                action_id = _record_action(cur, case, decision=option, actor=actor, now=now, reason=reason,
                                           scope=scope, facts=facts)
                _close_conflict(cur, decision_id, outcome=option, actor=actor, now=now, by_person=True)
                if verb == "keep_both":
                    other = next(k for k in keys if k != chosen)
                    _write_rule(cur, pair=pair, prevails=chosen, yields=other, case_decision_id=decision_id,
                                action_id=action_id, actor=actor, now=now)
                _notify(cur, _case_owners(case["facts"]),
                        f"The conflict between {keys[0]} and {keys[1]} was decided: {option_label(option)}.",
                        decision_id)
    if retired:
        _invalidate_enforcement()
        raise ConflictRefused("not_open", f"{retired} was retired; this conflict is closed.", 409)
    _invalidate_enforcement()
    row = {"decision_id": decision_id, "decision": option, "decision_scope": scope, "actioned_by": actor,
           "actioned_at": now.isoformat(), "override_reason": reason}
    out = conflict_payload.returned_decision(row)
    out["actionId"] = action_id
    out["applied"] = _APPLIED[verb]
    return out


def _moot(cur, case: Dict[str, Any], policy_key: str, *, now: datetime) -> None:
    """Close one locked open case as moot because policy_key was retired (caller's transaction)."""
    reason = f"{policy_key} was retired"
    did = int(case["decision_id"])
    facts = dict(case["facts"])
    facts["caseId"] = conflict_payload.case_id(did)
    _record_action(cur, case, decision=MOOT, actor=RETIRED_ACTOR, now=now, reason=reason,
                   scope="this_action", facts=facts)
    _close_conflict(cur, did, outcome=MOOT, actor=RETIRED_ACTOR, now=now, by_person=False)
    keys = str(case["subject_id"]).split("|")
    _notify(cur, _case_owners(case["facts"]),
            f"The conflict between {keys[0]} and {keys[1]} was closed: {reason}.", did)


def close_moot(conn, policy_key: str, *, now: datetime) -> int:
    """Close every open policy case naming a retired policy as moot. Returns how many were closed."""
    closed = 0
    with approvals._tx(conn):
        with conn.cursor() as cur:
            cur.execute("SELECT decision_id FROM proc.bp_agent_policy_conflict WHERE kind = 'policy' AND is_open "
                        "AND policy_keys @> ARRAY[%s]::text[] ORDER BY decision_id", (policy_key,))
            for (did,) in cur.fetchall():
                case = _case(cur, int(did), lock=True)
                if case is None or case["status"] != "open":
                    continue                                     # decided between the read and the lock
                _moot(cur, case, policy_key, now=now)
                closed += 1
    if closed:
        _invalidate_enforcement()
    return closed


def _retired_with_open_cases(conn, among: Optional[List[str]]) -> List[str]:
    """Retired policies that still have an open policy case (their moot close after the retire failed)."""
    sql = ("SELECT DISTINCT p.policy_key FROM proc.bp_agent_policy_conflict c "
           "JOIN proc.bp_agent_policy p ON p.policy_key = ANY(c.policy_keys) "
           "WHERE c.kind = 'policy' AND c.is_open AND p.status = 'retired'")
    params: List[Any] = []
    if among is not None:
        sql += " AND p.policy_key = ANY(%s)"
        params.append(list(among))
    with conn.cursor() as cur:
        cur.execute(sql + " ORDER BY p.policy_key", params)
        return [r[0] for r in cur.fetchall()]


def _close_moot_backstop(conn, *, now: datetime, among: Optional[List[str]], stats: Dict[str, Any]) -> None:
    """The scan's backstop for a retire whose moot close failed. Never counts against the case cap."""
    try:
        keys = _retired_with_open_cases(conn, among)
    except Exception as exc:  # noqa: BLE001
        stats["errors"] += 1
        logger.error("conflict scan could not list retired policies with open cases: %s", type(exc).__name__)
        return
    for key in keys:
        try:
            stats["mooted"] += close_moot(conn, key, now=now)
        except Exception as exc:  # noqa: BLE001 - one policy must not stop the scan
            stats["errors"] += 1
            logger.error("conflict scan could not close cases of retired %s: %s", key, type(exc).__name__)


# ---------------------------------------------------------------------------- standing rules + overlay
def rules_for(cur, keys: List[str]) -> List[Dict[str, Any]]:
    """The in-force standing rules touching any of the keys."""
    cur.execute(
        "SELECT pair_key, prevails, yields, rule_text, decision_id, decided_by, decided_at "
        "FROM proc.bp_agent_policy_conflict_rule WHERE superseded_at IS NULL "
        "AND (prevails = ANY(%s) OR yields = ANY(%s)) ORDER BY rule_id",
        (list(keys), list(keys)),
    )
    cols = ("pair_key", "prevails", "yields", "rule_text", "decision_id", "decided_by", "decided_at")
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def overlay(cur, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Each doc with its conflicts[] set from the in-force standing rules. Returns NEW dicts and
    never changes its input: version rows are immutable and cached copies are shared."""
    keys = sorted({str(d.get("id")) for d in docs if isinstance(d, dict) and d.get("id")})
    if not keys:
        return list(docs)
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    for r in rules_for(cur, keys):
        for me, other in ((r["prevails"], r["yields"]), (r["yields"], r["prevails"])):
            by_key.setdefault(me, []).append({
                "with": other, "rule": r["rule_text"], "caseId": conflict_payload.case_id(r["decision_id"]),
                "decidedAt": _iso(r["decided_at"]), "prevails": r["prevails"]})
    out = []
    for d in docs:
        if isinstance(d, dict):
            d = {**d, "conflicts": [dict(e) for e in by_key.get(str(d.get("id")), [])]}
        out.append(d)
    return out


# ---------------------------------------------------------------------------- history
def open_cases_by_policy(cur) -> Dict[str, List[str]]:
    """policy key -> its open policy caseIds."""
    cur.execute("SELECT decision_id, policy_keys FROM proc.bp_agent_policy_conflict "
                "WHERE kind = 'policy' AND is_open ORDER BY decision_id")
    out: Dict[str, List[str]] = {}
    for did, keys in cur.fetchall():
        for k in keys or []:
            out.setdefault(k, []).append(conflict_payload.case_id(did))
    return out


def _action_dict(row) -> Dict[str, Any]:
    decision_id, decision, scope, by, at, reason = row
    return {"decision_id": decision_id, "decision": decision, "decision_scope": scope, "actioned_by": by,
            "actioned_at": _iso(at), "override_reason": reason}


def history_for(cur, policy_key: str, latest_version: int, status: str) -> Dict[str, Any]:
    """A policy's conflicts (newest first, each with its returned decision) and the change, limit or
    retire decision still waiting for the owner, if any."""
    cur.execute(
        "SELECT decision_id, kind, is_open, policy_keys, created_at, outcome, decided_by, decided_at "
        "FROM proc.bp_agent_policy_conflict WHERE policy_keys @> ARRAY[%s]::text[] "
        "ORDER BY created_at DESC, decision_id DESC",
        (policy_key,),
    )
    rows = cur.fetchall()
    ids = [conflict_payload.case_id(r[0]) for r in rows]
    actions: Dict[str, Dict[str, Any]] = {}
    if ids:
        cur.execute(
            "SELECT facts->>'caseId', decision, decision_scope, actioned_by, actioned_at, override_reason "
            "FROM proc.bp_decision WHERE subject_type IN (%s, %s) AND status = 'actioned' "
            "AND facts->>'caseId' = ANY(%s) ORDER BY actioned_at, decision_id",
            (SUBJECT_POLICY, SUBJECT_LIVE, ids),
        )
        for cid, *rest in cur.fetchall():
            actions[cid] = _action_dict((conflict_payload.parse_case_id(cid), *rest))   # the latest wins
    conflicts = []
    for (did, kind, is_open, keys, created, outcome, by, at), cid in zip(rows, ids):
        decision = None
        if cid in actions:
            decision = conflict_payload.returned_decision(actions[cid])
        elif not is_open and outcome is not None:
            # closed without an action row of ours (a live case settled elsewhere): what the index says
            decision = conflict_payload.returned_decision(
                {"decision_id": did, "decision": outcome, "decision_scope": None, "actioned_by": by,
                 "actioned_at": _iso(at), "override_reason": None})
        conflicts.append({"caseId": cid, "kind": kind, "isOpen": bool(is_open),
                          "otherPolicies": [k for k in keys or [] if k != policy_key],
                          "raisedAt": _iso(created), "decision": decision})
    return {"conflicts": conflicts, "pendingAction": _pending(cur, policy_key, latest_version, status)}


def _pending(cur, policy_key: str, latest_version: int, status: str) -> Optional[Dict[str, Any]]:
    cur.execute(
        "SELECT decision, facts, override_reason, actioned_at FROM proc.bp_decision "
        "WHERE subject_type = %s AND status = 'actioned' AND decision = ANY(%s) "
        "ORDER BY actioned_at DESC, decision_id DESC LIMIT 1",
        (SUBJECT_POLICY, [f"{v}:{policy_key}" for v in _PENDING_VERBS]),
    )
    row = cur.fetchone()
    if not row:
        return None
    option, facts, reason, at = row
    facts = _j(facts, {}) or {}
    verb = option.partition(":")[0]
    if verb == "retire":
        if status == "retired":
            return None
    else:
        at_decision = (facts.get("versionsAtDecision") or {}).get(policy_key)
        if at_decision is None or int(latest_version) != int(at_decision):
            return None
    cid = facts.get("caseId")
    return {"caseId": cid, "action": verb,
            "changeNote": f"Conflict decision {cid}: {option_label(option)} — {reason}",
            "limitText": facts.get("limitText"), "decidedAt": _iso(at)}
