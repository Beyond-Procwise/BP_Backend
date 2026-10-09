"""Live conflict records: one agent action matched several policies that disagree.

A live case is a proc.bp_decision row (subject_type 'live_conflict', subject_id = the pair key of
every involved policy) plus its index row in proc.bp_agent_policy_conflict (kind 'live',
raised_by 'live'). The gate writes it inside its own transaction (insert_live):
- a block among the involved policies: recorded and closed at once (the block still blocks);
- a standing rule that decides it: recorded and closed at once;
- otherwise it stays open while one member approval case per conflicting approve policy, at that
  policy's LAST escalation level, waits; every member must approve.

When the member group closes (approvals._close_group), settle_for_group writes the live case's
action row ('approve' only when every member approved, else 'reject') and closes it. A person's
decision counts towards repeat-N (maybe_propose); N is the governed
agent_policy_conflicts.precedent_count (settings.precedent_count). A timeout, a block record or a standing-rule
auto decision never does (user ruling Q4).

Facts store only the action's CONDITION fields (facts.action.args), never the full arguments:
the member cases hold those for the replay. Every write runs on the caller's cursor, inside the
caller's transaction; nothing here commits.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from services.agent_policy import conflict_cases, conflict_detect, conflict_payload, deciders, durations
from services.agent_policy import settings as _settings

logger = logging.getLogger(__name__)

SUBJECT_LIVE = conflict_cases.SUBJECT_LIVE
SUBJECT_POLICY = conflict_cases.SUBJECT_POLICY
OPTIONS = ["approve", "reject"]            # Task 3 review: live options are exactly these
NOT_ALLOWED = "system:not_allowed"
STANDING_RULE = "system:standing_rule"
PRECEDENT = "system:precedent"
_AGENT = "agent_policy_conflicts"
#: decision -> (actor, scope) for a live case recorded closed; approve/reject only on precedent
_AUTO = {"block": (NOT_ALLOWED, "this_action"), "standing_rule": (STANDING_RULE, "standing_rule"),
         "approve": (PRECEDENT, "this_action"), "reject": (PRECEDENT, "this_action")}
_KIND = {"block": "block", "standing_rule": "standing_rule", "approve": "precedent", "reject": "precedent"}


def _j(v: Any, default=None):
    if v is None:
        return default
    if isinstance(v, (str, bytes)):
        return json.loads(v)
    return v


def _flat(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The gate's ctx as flat field names ("tool.name", "args.amount"), the witness shape."""
    out: Dict[str, Any] = {}
    for k, v in (ctx or {}).items():
        if k == "checkpoint":
            continue
        if k == "args" and isinstance(v, dict):
            out.update({f"args.{ak}": av for ak, av in v.items()})
        else:
            out[k] = v
    return out


def _within(doc: Dict[str, Any], default: str) -> str:
    sla = (((doc.get("enforcement") or {}).get("intervention") or {}).get("sla") or {})
    return durations.resolve(sla.get("respondWithin"), default)


def _plain(docs: List[Dict[str, Any]], tool: Optional[str]) -> str:
    for d in docs:
        plain = (((d.get("context") or {}).get("actions") or {}).get("plain") or "").strip()
        if plain:
            return plain[:1].upper() + plain[1:]
    return f"Running {tool or 'an action'}"


def _prior(cur, key: str) -> Dict[str, Any]:
    """How often this same set of policies was settled live before, and the last outcome."""
    cur.execute(
        "SELECT count(*), (array_agg(outcome ORDER BY decided_at DESC, decision_id DESC))[1] "
        "FROM proc.bp_agent_policy_conflict WHERE kind = 'live' AND pair_key = %s AND NOT is_open",
        (key,),
    )
    row = cur.fetchone() or (0, None)
    return {"sameConflict": int(row[0] or 0), "lastOutcome": row[1]}


def insert_live(cur, lc, *, ctx: Dict[str, Any], action: Dict[str, Any], now: datetime,
                default_response_time: str, status: str, decision: Optional[str] = None,
                actor: Optional[str] = None, reason: Optional[str] = None,
                extra_facts: Optional[Dict[str, Any]] = None,
                extra_evidence: Optional[List[Dict[str, Any]]] = None) -> int:
    """Record one live conflict on the caller's cursor. Returns its decision_id.

    status 'open' -> decision 'approve_or_reject', waiting for its member cases;
    status 'actioned' -> closed now by the system: 'block' | 'standing_rule', or 'approve' |
    'reject' decided on precedent (actor=PRECEDENT, given explicitly). A closed record carries
    facts.decidedBy and facts.versionsAtDecision. created_at is `now` on both rows, the clock the
    decision time is written from, so a closed record is never decided before it was raised. extra_facts are merged into the facts (an
    escalated clash's history); extra_evidence follows the overlap entry (cited precedents)."""
    if status not in ("open", "actioned"):
        raise ValueError(f"status must be 'open' or 'actioned', got {status!r}")
    if status == "actioned" and decision not in _AUTO:
        raise ValueError("an actioned live case needs decision 'block', 'standing_rule', 'approve' or "
                         f"'reject', got {decision!r}")
    if status == "actioned" and decision in ("approve", "reject") and actor != PRECEDENT:
        # explicitly: a missing actor is an error, never a default
        raise ValueError("only precedent records a live case as decided approve or reject: pass actor=PRECEDENT")
    docs = sorted((h["policy"] for h in lc.involved), key=lambda d: str(d.get("id")))
    keys = [str(d.get("id")) for d in docs]
    key = conflict_detect.pair_key(*keys)
    versions = {str(d.get("id")): _version(d) for d in docs}
    args = dict((action or {}).get("args") or {})
    approve_docs = [d for d in docs if (d.get("enforcement") or {}).get("outcome") == "approve"]
    within = (max((_within(d, default_response_time) for d in approve_docs), key=durations.parse)
              if approve_docs else None)
    live_action = {"tool": action.get("tool"), "args": conflict_payload.condition_args(docs, args),
                   "agent": action.get("agent"), "workflowId": action.get("workflowId"),
                   "plain": _plain(docs, action.get("tool"))}
    standing = [{"policy": str(d.get("id")), **r} for d in docs for r in (d.get("conflicts") or [])
                if isinstance(r, dict)]
    payload = conflict_payload.build(
        "live", raised_at=now.isoformat(), policies=[conflict_payload.policy_entry(d) for d in docs],
        overlap_example=conflict_payload.condition_values(docs, _flat(ctx)), standing_rules=standing,
        prior=_prior(cur, key), options=list(OPTIONS), respond_within=within, on_timeout="reject",
        action=live_action)
    cols = conflict_payload.to_columns(payload)
    facts = dict(cols["facts"])
    facts["pairs"] = [list(p) for p in lc.pairs]
    facts["requestedBy"] = action.get("userId")
    facts.update(dict(extra_facts or {}))
    evidence = list(cols["evidence"]) + [dict(e) for e in extra_evidence or []]

    open_ = status == "open"
    if open_:
        decision, actor, scope = "approve_or_reject", None, None
        respond_by = now + timedelta(seconds=durations.parse(durations.resolve(within, default_response_time)))
    else:
        actor = actor or _AUTO[decision][0]
        scope = _AUTO[decision][1]
        respond_by = None
        facts["decidedBy"] = conflict_cases.decided_by(_KIND[decision], actor)
        facts["versionsAtDecision"] = dict(versions)
    cur.execute(
        """
        INSERT INTO proc.bp_decision (
            subject_type, subject_id, decision, resolution, rationale, status,
            policy_id, policy_name, facts, evidence, workflow_id, agent, created_by,
            options, respond_by, on_timeout, decision_scope, actioned_by, actioned_at, override_reason,
            created_at
        ) VALUES (%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING decision_id
        """,
        (cols["subject_type"], key, decision, "escalated" if open_ else "resolved",
         conflict_payload.why_line(docs), "open" if open_ else "actioned", key,
         json.dumps(facts, default=str), json.dumps(evidence, default=str), action.get("workflowId"),
         action.get("agent"), action.get("userId") or f"agent:{action.get('agent') or 'unknown'}",
         json.dumps(cols["options"]), respond_by, cols["on_timeout"], scope, actor,
         None if open_ else now, None if open_ else reason, now),
    )
    decision_id = int(cur.fetchone()[0])
    cur.execute("UPDATE proc.bp_decision SET facts = facts || %s::jsonb WHERE decision_id = %s",
                (json.dumps({"caseId": conflict_payload.case_id(decision_id)}), decision_id))
    cur.execute(
        "INSERT INTO proc.bp_agent_policy_conflict (decision_id, kind, pair_key, policy_keys, policy_versions, "
        "raised_by, is_open, outcome, decided_by, decided_at, by_person, created_at) "
        "VALUES (%s, 'live', %s, %s, %s, 'live', %s, %s, %s, %s, %s, %s)",
        (decision_id, key, keys, json.dumps(versions), open_, None if open_ else decision,
         None if open_ else actor, None if open_ else now, None if open_ else False, now),
    )
    return decision_id


def _version(doc: Dict[str, Any]) -> int:
    try:
        return int(doc.get("version") or 0)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------- settle
_LIVE_SQL = """
    SELECT decision_id, subject_id, resolution, rationale, policy_name, facts, status, workflow_id
      FROM proc.bp_decision WHERE decision_id = %s AND subject_type = %s FOR UPDATE
"""


def _live_ids(cur, member_ids: List[int]) -> List[int]:
    cur.execute("SELECT DISTINCT (facts->>'liveConflict')::bigint FROM proc.bp_decision "
                "WHERE decision_id = ANY(%s) AND facts ? 'liveConflict' "
                "AND jsonb_typeof(facts->'liveConflict') = 'number'", (list(member_ids),))
    return sorted(int(r[0]) for r in cur.fetchall() if r[0] is not None)


def threshold() -> Optional[int]:
    """The governed precedent count (settings.precedent_count): at N the standing-rule proposal is
    raised, as the engine starts deciding on precedent (ruling R2). None when it cannot be read:
    no proposal then, with a warning, never a number made up here."""
    try:
        return _settings.precedent_count()
    except Exception as exc:  # noqa: BLE001 - LimitUnavailable, or a value that is not a number
        logger.warning("repeat proposal skipped: the precedent count cannot be read (%s)", type(exc).__name__)
        return None


def _credited(cur, approvals, member_ids: List[int], verdict: str, actor: str,
              reason: Optional[str]) -> tuple:
    """(actor, reason) of the member action that decided the group: for a reject the EARLIEST
    member reject not written by system:group (a timeout or a person); for an approve the LAST
    approver. Falls back to the caller's actor when no such row is found."""
    cur.execute(
        "SELECT a.decision, a.actioned_by, a.override_reason FROM proc.bp_decision a "
        "JOIN proc.bp_decision c ON c.subject_type = a.subject_type AND c.subject_id = a.subject_id "
        "WHERE c.decision_id = ANY(%s) AND c.subject_type = %s AND a.status = 'actioned' "
        "AND a.decision IN ('approve','reject') AND a.actioned_by IS NOT NULL ORDER BY a.decision_id",
        (list(member_ids), approvals.SUBJECT_TYPE),
    )
    acts = cur.fetchall()
    if verdict == "reject":
        rejects = [r for r in acts if r[0] == "reject"]
        pick = next((r for r in rejects if r[1] != approvals.GROUP_ACTOR), rejects[0] if rejects else None)
    else:
        pick = next((r for r in reversed(acts) if r[0] == "approve"), None)
    return (pick[1], pick[2]) if pick else (actor, reason)


def _settled_kind(approvals, actor: Optional[str]) -> str:
    """decidedBy.kind of the credited member action: a person, the timeout sweep (exactly
    approvals.TIMEOUT_ACTOR), or any other system actor ("system": a system:group row credited
    because every reject on file is one, or a system caller's own actor). Only "person" counts
    as a person's decision."""
    a = str(actor or "")
    if a == approvals.TIMEOUT_ACTOR:
        return "timeout"
    if not actor or a.startswith("system:"):
        return "system"
    return "person"


def settle_for_group(cur, case: Dict[str, Any], state: Dict[str, int], *, refused: Optional[str], actor: str,
                     reason: Optional[str], now: datetime) -> Optional[int]:
    """The member group of `case` has closed: settle the live case(s) its members name, if still
    open. Returns the (last) action row id, or None when there was nothing to settle."""
    from services.agent_policy import approvals   # lazily: approvals imports us lazily too

    member_ids = approvals._member_ids(cur, int(case["decision_id"]), case["facts"])
    approved = refused is None and state.get("rejected", 0) == 0 and state.get("approved", 0) == state.get("total", 0)
    verdict = "approve" if approved else "reject"
    # Credit the member decision that settled it, never whoever happened to close the group: a
    # sweep can time one member out while a person is approving its sibling (user ruling Q4: a
    # timeout never counts as a person's decision).
    actor, reason = _credited(cur, approvals, member_ids, verdict, actor, reason)
    kind = _settled_kind(approvals, actor)
    by_person = kind == "person"
    out: Optional[int] = None
    for live_id in _live_ids(cur, member_ids):
        cur.execute(_LIVE_SQL, (live_id, SUBJECT_LIVE))
        row = cur.fetchone()
        if not row:
            continue
        live = dict(zip([d[0] for d in cur.description], row))
        if live["status"] != "open":
            continue                                   # block record, standing rule, or settled already
        facts = dict(_j(live["facts"], {}) or {})
        facts["caseId"] = conflict_payload.case_id(live_id)
        facts["memberCases"] = list(member_ids)
        cur.execute("SELECT policy_versions FROM proc.bp_agent_policy_conflict WHERE decision_id = %s", (live_id,))
        pv = cur.fetchone()
        facts["versionsAtDecision"] = dict(_j(pv[0], {}) or {}) if pv else {}
        if kind == "system":
            logger.warning("live conflict %s settled by unexpected system actor %s", live_id, actor)
        facts["decidedBy"] = conflict_cases.decided_by(kind, actor)
        cur.execute(
            """
            INSERT INTO proc.bp_decision (
                subject_type, subject_id, decision, resolution, rationale, policy_id, policy_name,
                facts, evidence, status, actioned_by, actioned_at, override_reason,
                workflow_id, agent, created_by, decision_scope
            ) VALUES (%s,%s,%s,%s,%s,NULL,%s,%s,'[]','actioned',%s,%s,%s,%s,%s,%s,'this_action')
            RETURNING decision_id
            """,
            (SUBJECT_LIVE, live["subject_id"], verdict, live["resolution"], live["rationale"], live["policy_name"],
             json.dumps(facts, default=str), actor, now, reason, live["workflow_id"], _AGENT, actor),
        )
        out = int(cur.fetchone()[0])
        cur.execute("UPDATE proc.bp_decision SET status = 'actioned' WHERE decision_id = %s AND subject_type = %s",
                    (live_id, SUBJECT_LIVE))
        conflict_cases._close_conflict(cur, live_id, outcome=verdict, actor=actor, now=now, by_person=by_person)
        if by_person:
            _propose_safely(cur, live_id, now=now)
    return out


def _propose_safely(cur, live_id: int, *, now: datetime) -> None:
    """maybe_propose in a savepoint: a failed proposal is logged and undone on its own and never
    rolls back the person's approval or the settle (they stay in the caller's transaction).
    N missing, null or 0: no proposal, nothing runs."""
    n = threshold()
    if not n or n <= 0:
        return
    cur.execute("SAVEPOINT live_conflict_propose")
    try:
        maybe_propose(cur, live_id, now=now, threshold=n)
    except Exception as exc:  # noqa: BLE001 - type only: a driver message can quote stored values
        cur.execute("ROLLBACK TO SAVEPOINT live_conflict_propose")
        logger.error("repeat proposal failed for live case %s: %s", live_id, type(exc).__name__)
    cur.execute("RELEASE SAVEPOINT live_conflict_propose")


def _in_key_order(pairs) -> List[tuple]:
    """Pairs in sorted pair-key order, as detect_for takes them: two writers locking the same
    pairs always lock them in one order, so they cannot deadlock."""
    return sorted(pairs, key=lambda p: conflict_detect.pair_key(*p))


def _live_docs(cur, keys: List[str]) -> Dict[str, Dict[str, Any]]:
    """The live version of each key (the proposal is about the policies as they are now)."""
    cur.execute(
        "SELECT p.policy_key, v.compiled FROM proc.bp_agent_policy p JOIN proc.bp_agent_policy_version v "
        "ON v.policy_key = p.policy_key AND v.version = p.live_version "
        "WHERE p.policy_key = ANY(%s) AND p.status = 'live'",
        (list(keys),),
    )
    return {k: d for k, d in ((r[0], _j(r[1], {})) for r in cur.fetchall()) if isinstance(d, dict)}


def maybe_propose(cur, live_id: int, *, now: datetime, threshold: int) -> List[int]:
    """When the last `threshold` person-decided live cases of this same set of policies all went
    one way, raise a policy case per pair proposing a standing rule. Returns the new case ids.

    Timeouts, block records and standing-rule decisions are not person decisions and never count;
    a different outcome among the last N resets the count. Dedup is raise_policy_case's (no second
    case while one is open, while a standing rule covers the pair, or for decided versions)."""
    cur.execute("SELECT pair_key FROM proc.bp_agent_policy_conflict WHERE decision_id = %s AND kind = 'live'",
                (live_id,))
    row = cur.fetchone()
    if not row or threshold <= 0:
        return []
    key = row[0]
    cur.execute(
        "SELECT outcome FROM proc.bp_agent_policy_conflict WHERE kind = 'live' AND pair_key = %s "
        "AND NOT is_open AND by_person ORDER BY decided_at DESC, decision_id DESC LIMIT %s",
        (key, int(threshold)),
    )
    outcomes = [r[0] for r in cur.fetchall()]
    if len(outcomes) < threshold or len(set(outcomes)) != 1:
        return []
    outcome = outcomes[0]
    cur.execute("SELECT facts, evidence FROM proc.bp_decision WHERE decision_id = %s AND subject_type = %s",
                (live_id, SUBJECT_LIVE))
    facts_row = cur.fetchone()
    if not facts_row:
        return []
    facts = _j(facts_row[0], {}) or {}
    evidence = _j(facts_row[1], []) or []
    example = next((e.get("example") for e in evidence if isinstance(e, dict) and e.get("kind") == "overlap"), {})
    pairs = [tuple(p) for p in facts.get("pairs") or [] if isinstance(p, (list, tuple)) and len(p) == 2]
    docs = _live_docs(cur, sorted({k for p in pairs for k in p}))
    mapping = deciders.load_map(cur.connection)
    budget = conflict_cases.cap_of(_settings.load_settings())
    raised: List[int] = []
    for a, b in _in_key_order(pairs):
        if a not in docs or b not in docs:
            logger.info("repeat proposal for %s|%s skipped: a policy is no longer live", a, b)
            continue
        if len(raised) >= budget:
            conflict_cases._cap_reached(budget)
            break
        did = conflict_cases.raise_policy_case(
            cur, docs[a], docs[b], dict(example or {}), raised_by="repeat", now=now, mapping=mapping,
            proposal={"from": "repeat", "count": int(threshold), "outcome": outcome})
        if did is not None:
            raised.append(did)
    return raised


# ---------------------------------------------------------------------------- block record
def raise_block_pairs(cur, lc, *, ctx: Dict[str, Any], now: datetime, mapping) -> List[int]:
    """A live conflict involving a block: a policy case for each pair containing a block, with the
    live action's condition values as the witness. Deduplicated as at design time."""
    by_key = {str(h["id"]): h for h in lc.involved}
    flat = _flat(ctx)
    budget = conflict_cases.cap_of(_settings.load_settings())
    raised: List[int] = []
    for a, b in _in_key_order(lc.pairs):
        ha, hb = by_key.get(a), by_key.get(b)
        if ha is None or hb is None or "block" not in (ha["outcome"], hb["outcome"]):
            continue
        if len(raised) >= budget:
            conflict_cases._cap_reached(budget)
            break
        docs = [ha["policy"], hb["policy"]]
        did = conflict_cases.raise_policy_case(cur, ha["policy"], hb["policy"],
                                               conflict_payload.condition_values(docs, flat),
                                               raised_by="live", now=now, mapping=mapping)
        if did is not None:
            raised.append(did)
    return raised


def rule_text(lc) -> Optional[str]:
    """The standing rules that decided an auto conflict, in words."""
    texts = []
    for r in lc.rules or []:
        t = str(r.get("rule") or f"{r.get('prevails')} takes priority over {r.get('with')}").strip()
        if t and t not in texts:
            texts.append(t)
    return "; ".join(texts) or None
