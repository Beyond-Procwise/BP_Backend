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
decision counts towards repeat-N (maybe_propose); a timeout, a block record or a standing-rule
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
_AGENT = "agent_policy_conflicts"
_AUTO = {"block": (NOT_ALLOWED, "this_action"), "standing_rule": (STANDING_RULE, "standing_rule")}


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
                actor: Optional[str] = None, reason: Optional[str] = None) -> int:
    """Record one live conflict on the caller's cursor. Returns its decision_id.

    status 'open' -> decision 'approve_or_reject', waiting for its member cases;
    status 'actioned' -> decision 'block' | 'standing_rule', closed now by the system."""
    if status not in ("open", "actioned"):
        raise ValueError(f"status must be 'open' or 'actioned', got {status!r}")
    if status == "actioned" and decision not in _AUTO:
        raise ValueError(f"an actioned live case needs decision 'block' or 'standing_rule', got {decision!r}")
    docs = sorted((h["policy"] for h in lc.involved), key=lambda d: str(d.get("id")))
    keys = [str(d.get("id")) for d in docs]
    key = conflict_detect.pair_key(*keys)
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

    open_ = status == "open"
    if open_:
        decision, actor, scope = "approve_or_reject", None, None
        respond_by = now + timedelta(seconds=durations.parse(durations.resolve(within, default_response_time)))
    else:
        actor = actor or _AUTO[decision][0]
        scope = _AUTO[decision][1]
        respond_by = None
    cur.execute(
        """
        INSERT INTO proc.bp_decision (
            subject_type, subject_id, decision, resolution, rationale, status,
            policy_id, policy_name, facts, evidence, workflow_id, agent, created_by,
            options, respond_by, on_timeout, decision_scope, actioned_by, actioned_at, override_reason
        ) VALUES (%s,%s,%s,%s,%s,%s,NULL,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        RETURNING decision_id
        """,
        (cols["subject_type"], key, decision, "escalated" if open_ else "resolved",
         conflict_payload.why_line(docs), "open" if open_ else "actioned", key,
         json.dumps(facts, default=str), json.dumps(cols["evidence"], default=str), action.get("workflowId"),
         action.get("agent"), action.get("userId") or f"agent:{action.get('agent') or 'unknown'}",
         json.dumps(cols["options"]), respond_by, cols["on_timeout"], scope, actor,
         None if open_ else now, None if open_ else reason),
    )
    decision_id = int(cur.fetchone()[0])
    cur.execute("UPDATE proc.bp_decision SET facts = facts || %s::jsonb WHERE decision_id = %s",
                (json.dumps({"caseId": conflict_payload.case_id(decision_id)}), decision_id))
    versions = {str(d.get("id")): _version(d) for d in docs}
    cur.execute(
        "INSERT INTO proc.bp_agent_policy_conflict (decision_id, kind, pair_key, policy_keys, policy_versions, "
        "raised_by, is_open, outcome, decided_by, decided_at, by_person) "
        "VALUES (%s, 'live', %s, %s, %s, 'live', %s, %s, %s, %s, %s)",
        (decision_id, key, keys, json.dumps(versions), open_, None if open_ else decision,
         None if open_ else actor, None if open_ else now, None if open_ else False),
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


def threshold() -> int:
    """The company's repeat-N (settings.live_conflict_repeat); unreadable -> the ruled default."""
    try:
        n = int(_settings.load_settings().get("live_conflict_repeat"))
    except (TypeError, ValueError):
        n = 0
    return n if n > 0 else int(_settings.DEFAULTS["live_conflict_repeat"])


def settle_for_group(cur, case: Dict[str, Any], state: Dict[str, int], *, refused: Optional[str], actor: str,
                     reason: Optional[str], now: datetime) -> Optional[int]:
    """The member group of `case` has closed: settle the live case(s) its members name, if still
    open. Returns the (last) action row id, or None when there was nothing to settle."""
    from services.agent_policy import approvals   # lazily: approvals imports us lazily too

    member_ids = approvals._member_ids(cur, int(case["decision_id"]), case["facts"])
    approved = refused is None and state.get("rejected", 0) == 0 and state.get("approved", 0) == state.get("total", 0)
    verdict = "approve" if approved else "reject"
    by_person = not str(actor or "").startswith("system:")
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
            maybe_propose(cur, live_id, now=now, threshold=threshold())
    return out


# ---------------------------------------------------------------------------- repeat-N
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
    for a, b in pairs:
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
    for a, b in lc.pairs:
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
