"""The policy gate every agent tool call passes through, immediately before its handler runs.

``before_tool`` answers one question: may this call run now? It
1. honours the kill switch (AGENT_POLICY_ENFORCEMENT=off, any case: allow, nothing loaded);
2. loads the valid live policies (none -> allow, nothing written: behaviour identical to today);
3. runs enforcement.check (the one evaluator);
4. writes one firing row per matched policy, the notifications every notify recipient gets
   (whatever the result), and one approval case per matched approve policy, all sharing a
   ``firing_group``. A repeat of the same call while its case is still open reuses that case
   (same policy, tool, args digest and workflow) instead of opening another.

Fail closed: ANY exception (store unavailable, check() raising, a write failing) refuses the
call with reasonCode ``policy_check_unavailable`` and a best-effort firing row (policy_key '*').
Notification text never contains input values.

Live conflicts (stage 4): conflict_engine.classify(verdict) runs right after the check. With no
conflict (kind None) every statement is exactly stage 3's. Otherwise, in the same transaction:
- block_record: the block still blocks; a closed live record and a policy case per block pair;
- auto: a standing rule decides; only the rule's winner(s) get (normal) approval cases;
- human: the action pauses; one member case per approve policy, a conflicting one at its LAST
  escalation level only, all naming the open live record; every member must approve.
- human, on a fresh call: the decision engine (decision_engine.decide_live_conflict) may decide it
  on precedent: the action runs, or is refused with refused_on_precedent; otherwise it goes to
  people as below, with the clash's history attached.
The agent is told the live case as ``conflictCaseId``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from engines import decision_engine as DE
from services.agent_policy import (approvals, conflict_detect, conflict_engine, conflict_history, conflict_live,
                                   conflict_payload, deciders, enforcement, live_policies, settings)

logger = logging.getLogger(__name__)

CHECKPOINT = "tool.call.before"
REASON_MAX = 500
UNAVAILABLE = {"result": "blocked", "reasonCode": "policy_check_unavailable",
               "reason": "Policy checks are unavailable, so this action was not run."}
PRECEDENT_REFUSED = "refused_on_precedent"


@dataclass
class GateResult:
    allow: bool
    to_agent: Optional[Dict[str, Any]] = None
    firing_ids: List[int] = field(default_factory=list)
    case_ids: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------- seams
def _connect():
    from services.db import get_conn

    return get_conn()


def _load_policies() -> List[Dict[str, Any]]:
    return live_policies.load()


def _default_response_time() -> str:
    return settings.load_settings().get("response_time") or "PT4H"


def enforcement_enabled() -> bool:
    return os.getenv("AGENT_POLICY_ENFORCEMENT", "on").strip().lower() != "off"


# ---------------------------------------------------------------------------- helpers
def clean_reason(text: Any) -> Optional[str]:
    """The assistant text that accompanied the tool call: stripped, capped, or absent."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    return text[:REASON_MAX] or None


def args_digest(args: Dict[str, Any]) -> str:
    canon = json.dumps(args or {}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _ctx(tool_name: str, args: Dict[str, Any], agent: Optional[str], reason: Optional[str]) -> Dict[str, Any]:
    ctx: Dict[str, Any] = {"checkpoint": CHECKPOINT, "tool.name": tool_name, "args": dict(args or {})}
    if agent:
        ctx["agent.name"] = agent
    if reason:
        ctx["agent.reason"] = reason
    return ctx


_OUTCOME_TEXT = {"blocked": "was blocked", "paused_for_approval": "is waiting for approval",
                 "allowed": "was allowed to run"}


def _what(policy: Dict[str, Any], tool_name: str) -> str:
    plain = (((policy.get("context") or {}).get("actions") or {}).get("plain") or "").strip()
    what = plain or f"running {tool_name or 'an action'}"
    return what[:1].upper() + what[1:]


def _insert_firing(cur, *, key: str, version: int, tool_name: str, agent, workflow_id, user_id,
                   outcome: str, result: str, matched_values, missing, duration_ms: int,
                   decision_id: Optional[int] = None, reason: Optional[str] = None) -> int:
    cur.execute(
        "INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, agent, "
        "workflow_id, requested_by, outcome, result, matched_values, missing_inputs, decision_id, "
        "reason, duration_ms) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING firing_id",
        (key, version, CHECKPOINT, tool_name or "", agent, workflow_id, user_id, outcome, result,
         json.dumps(matched_values or {}, default=str), list(missing or []), decision_id, reason,
         duration_ms),
    )
    return int(cur.fetchone()[0])


def _notify(cur, firing_id: int, recipients, message: str, link: str) -> None:
    for r in recipients or []:
        r = str(r or "").strip()
        if r:
            cur.execute("INSERT INTO proc.bp_policy_notification (firing_id, recipient, message, link) "
                        "VALUES (%s,%s,%s,%s)", (firing_id, r, message, link))


def record_matches(cur, hits: List[Dict[str, Any]], notifies: List[Dict[str, Any]], *, result: str,
                   tool_name: str, agent, workflow_id, user_id, duration_ms: int,
                   reason: Optional[str] = None, decision_id: Optional[int] = None) -> Dict[int, int]:
    """One firing row per matched policy (all recording `result`, the action's fate) and the
    notifications every notify recipient gets. Returns {id(hit): firing_id}. Used by the gate and
    by the replay's re-check, on the caller's cursor inside the caller's transaction. `decision_id`
    links the rows at insert (a settled row is append-only: it can never be linked afterwards)."""
    firing_of: Dict[int, int] = {}
    for hit in hits:
        if id(hit) in firing_of:
            continue
        firing_of[id(hit)] = _insert_firing(
            cur, key=str(hit["id"]), version=int(hit.get("version") or 0), tool_name=tool_name,
            agent=agent, workflow_id=workflow_id, user_id=user_id, outcome=hit["outcome"], result=result,
            matched_values=hit.get("matched_values"), missing=hit.get("missing"),
            duration_ms=duration_ms, decision_id=decision_id, reason=reason)
    # notify recipients hear about it whatever the result (a block carrying a notify list too)
    for hit in notifies:
        key = str(hit["id"])
        msg = f"{_what(hit['policy'], tool_name)} (policy {key}) {_OUTCOME_TEXT.get(result, 'was checked')}."
        _notify(cur, firing_of[id(hit)], hit.get("notify"), msg, f"agent-policy:{key}")
    return firing_of


def link_paused_notifies(cur, firing_ids: List[int], case_id: int) -> None:
    """A notify row of a paused call is itself paused and linked to one case of the call's group;
    approvals._close_group settles it once the whole group is decided."""
    if firing_ids:
        cur.execute("UPDATE proc.bp_policy_firing SET decision_id = %s "
                    "WHERE firing_id = ANY(%s) AND result = 'paused_for_approval'",
                    (case_id, list(firing_ids)))


def _level_names(policy: Dict[str, Any]) -> List[str]:
    return [str(e.get("name")).strip() for e in
            ((policy.get("enforcement") or {}).get("intervention") or {}).get("escalateTo") or []
            if isinstance(e, dict) and str(e.get("name") or "").strip()]


def notify_last_level(cur, policy: Dict[str, Any], firing_id: int, decision_id: int, tool_name: str) -> None:
    """A live-conflict member case is decided at the policy's last level only: that level hears."""
    levels = _level_names(policy)
    if levels:
        _notify(cur, firing_id, [levels[-1]],
                f"{_what(policy, tool_name)} (policy {policy.get('id')}) needs your decision.",
                f"decision:{decision_id}")


def notify_first_level(cur, policy: Dict[str, Any], firing_id: int, decision_id: int, tool_name: str) -> None:
    """The first level of a newly opened case hears that a decision is waiting (escalations
    notify the next level from the sweep)."""
    levels = [str(e.get("name")).strip() for e in
              ((policy.get("enforcement") or {}).get("intervention") or {}).get("escalateTo") or []
              if isinstance(e, dict) and str(e.get("name") or "").strip()]
    if levels:
        _notify(cur, firing_id, [levels[0]],
                f"{_what(policy, tool_name)} (policy {policy.get('id')}) needs your decision.",
                f"decision:{decision_id}")


def _lock_key(tool_name: str, digest: str, workflow_id, user_id) -> str:
    """The per-call advisory lock: two identical calls (tool, args, workflow, requester) serialise."""
    return f"agent_policy_gate:{tool_name}:{digest}:{workflow_id}:{user_id}"


def _open_case_for(cur, *, key: str, tool_name: str, digest: str, workflow_id,
                   requested_by) -> Optional[Dict[str, Any]]:
    """The open case this exact call (same policy, tool, args, workflow AND requester) made."""
    cur.execute(
        "SELECT decision_id, facts FROM proc.bp_decision WHERE subject_type = %s AND status = 'open' "
        "AND decision = 'approve_or_reject' AND policy_name = %s AND facts->'action'->>'tool' = %s "
        "AND facts->>'argsDigest' = %s AND workflow_id IS NOT DISTINCT FROM %s "
        "AND facts->>'requestedBy' IS NOT DISTINCT FROM %s "
        "ORDER BY decision_id LIMIT 1",
        (approvals.SUBJECT_TYPE, key, tool_name, digest, workflow_id, requested_by),
    )
    row = cur.fetchone()
    if not row:
        return None
    facts = row[1] if isinstance(row[1], dict) else json.loads(row[1] or "{}")
    return {"decision_id": int(row[0]), "facts": facts}


def _refuse(exc: BaseException, *, tool_name, agent, workflow_id, user_id, started: float) -> GateResult:
    # the type only: a driver's message can quote row values (the call's arguments)
    logger.error("agent policy check unavailable for tool %s: %s", tool_name, type(exc).__name__)
    firing_ids: List[int] = []
    try:
        with _connect() as conn:
            with conn.cursor() as cur:
                firing_ids.append(_insert_firing(
                    cur, key="*", version=0, tool_name=tool_name, agent=agent, workflow_id=workflow_id,
                    user_id=user_id, outcome="block", result="error", matched_values={}, missing=[],
                    duration_ms=int((time.monotonic() - started) * 1000),
                    reason=f"policy_check_unavailable: {type(exc).__name__}"))
    except Exception:  # noqa: BLE001 - best effort; the refusal stands either way
        logger.warning("could not log the policy_check_unavailable refusal for %s", tool_name)
    return GateResult(allow=False, to_agent=dict(UNAVAILABLE), firing_ids=firing_ids)


# ---------------------------------------------------------------------------- the gate
def before_tool(*, tool_name: str, args: Dict[str, Any], agent: Optional[str] = None,
                reason: Optional[str] = None, workflow_id: Optional[str] = None,
                user_id: Optional[str] = None) -> GateResult:
    """May this tool call run now? Never raises: any failure refuses (fail closed)."""
    if not enforcement_enabled():
        return GateResult(allow=True)
    started = time.monotonic()
    try:
        return _before_tool(tool_name=tool_name, args=dict(args or {}), agent=agent,
                            reason=clean_reason(reason), workflow_id=workflow_id, user_id=user_id,
                            started=started)
    except Exception as exc:  # noqa: BLE001 - every failure is the same answer: cannot check
        return _refuse(exc, tool_name=tool_name, agent=agent, workflow_id=workflow_id,
                       user_id=user_id, started=started)


def _before_tool(*, tool_name, args, agent, reason, workflow_id, user_id, started) -> GateResult:
    policies = _load_policies()
    if not policies:
        return GateResult(allow=True)
    ctx = _ctx(tool_name, args, agent, reason)
    verdict = enforcement.check(ctx, policies, default_response_time=_default_response_time())
    lc = conflict_engine.classify(verdict)
    matched: List[Dict[str, Any]] = []
    for hit in verdict.blocks + verdict.approvals + verdict.notifies:
        if not any(h is hit for h in matched):
            matched.append(hit)
    if not matched:
        return GateResult(allow=True)

    duration_ms = int((time.monotonic() - started) * 1000)
    digest = args_digest(args)
    firing_ids: List[int] = []
    case_ids: List[int] = []
    now = datetime.now(timezone.utc)
    action = {"tool": tool_name, "args": args, "agent": agent, "workflowId": workflow_id,
              "userId": user_id, "reason": reason}
    live: Dict[str, Optional[int]] = {"id": None}
    precedent = None          # the engine's Decision when it resolved the clash on precedent
    note: Optional[str] = None  # why precedent did not decide a 'human' clash

    # One transaction for every row this call writes (firing rows, notifications, cases): on any
    # failure nothing is left behind -- never an open case for a call the agent was refused.
    with _connect() as conn:
        with approvals._tx(conn):
            if lc.kind == "human" and verdict.result == "paused_for_approval":
                consulted = _consult_precedent(conn, lc, ctx=ctx, digest=digest, tool_name=tool_name,
                                               workflow_id=workflow_id, user_id=user_id, now=now)
                if consulted is not None and consulted.resolution == DE.RESOLVED:
                    precedent = consulted
                elif consulted is not None:
                    note = consulted.rationale
            if precedent is not None:
                live["id"], firing_ids = _record_precedent(
                    conn, lc, verdict, matched, precedent, action=action, ctx=ctx, tool_name=tool_name,
                    agent=agent, workflow_id=workflow_id, user_id=user_id, duration_ms=duration_ms, now=now)
            else:
                with conn.cursor() as cur:
                    firing_of = record_matches(cur, matched, verdict.notifies, result=verdict.result,
                                               tool_name=tool_name, agent=agent, workflow_id=workflow_id,
                                               user_id=user_id, duration_ms=duration_ms)
                firing_ids.extend(firing_of[id(h)] for h in matched)

                if lc.kind == "block_record":
                    live["id"] = _record_block(conn, lc, action=action, ctx=ctx, now=now)

                if verdict.result == "paused_for_approval":
                    case_ids = _open_cases(conn, verdict, firing_of, action=action, ctx=ctx, digest=digest,
                                           tool_name=tool_name, workflow_id=workflow_id, user_id=user_id,
                                           now=now, **({"lc": lc, "live": live, "note": note} if lc.kind else {}))
                    # A notify row of a paused call is itself paused and linked to the group's first
                    # case; approvals._close_group settles it once the whole group is decided.
                    if case_ids:
                        with conn.cursor() as cur:
                            link_paused_notifies(cur, [firing_of[id(h)] for h in verdict.notifies], case_ids[0])

    if precedent is not None:
        return _precedent_answer(precedent, live["id"], firing_ids)
    if verdict.result == "allowed":
        return GateResult(allow=True, firing_ids=firing_ids)
    to_agent = dict(verdict.to_agent or {})
    if verdict.result == "paused_for_approval":
        to_agent["requestIds"] = list(case_ids)
    if live["id"] is not None:
        to_agent["conflictCaseId"] = conflict_payload.case_id(live["id"])
    return GateResult(allow=False, to_agent=to_agent, firing_ids=firing_ids, case_ids=case_ids)


def _record_block(conn, lc, *, action, ctx, now) -> int:
    """A live conflict involving a block (the block still blocks): the closed live record and a
    policy case for every pair containing the block. Returns the live record's id."""
    with conn.cursor() as cur:
        live_id = conflict_live.insert_live(cur, lc, ctx=ctx, action=action, now=now,
                                            default_response_time=_default_response_time(),
                                            status="actioned", decision="block")
        conflict_live.raise_block_pairs(cur, lc, ctx=ctx, now=now, mapping=deciders.load_map(conn))
    return live_id


def _consult_precedent(conn, lc, *, ctx, digest, tool_name, workflow_id, user_id, now):
    """The decision engine on a 'human' clash, under the call's own lock (inside the gate's
    transaction). None when this call repeats one whose member cases are still open: that call is
    already with people, and precedent is not consulted again."""
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (_lock_key(tool_name, digest, workflow_id, user_id),))
        if any(_open_case_for(cur, key=str(h["id"]), tool_name=tool_name, digest=digest,
                              workflow_id=workflow_id, requested_by=user_id) for h in lc.required):
            return None
        return DE.decide_live_conflict(cur, lc, ctx=ctx, now=now)


def _notify_precedent(cur, lc, firing_of, decision, case: str, tool_name: str) -> None:
    """Each involved policy's owner and deciders hear what precedent did. No input values."""
    verb = "ran on precedent" if decision.decision == "approve" else "was refused on precedent"
    n = len(decision.evidence)
    for hit in sorted(lc.involved, key=lambda h: str(h["id"])):
        policy = hit["policy"]
        names: List[str] = []
        for name in [str(policy.get("owner") or "").strip(), *_level_names(policy)]:
            if name and name not in names:
                names.append(name)
        _notify(cur, firing_of[id(hit)], names,
                f"{_what(policy, tool_name)} (policy {hit['id']}) {verb}: decided the same way {n} times "
                f"before ({case}).", f"agent-policy:{hit['id']}")


def _record_precedent(conn, lc, verdict, matched, decision, *, action, ctx, tool_name, agent, workflow_id,
                      user_id, duration_ms, now):
    """A clash the engine decided on precedent, in the gate's transaction: the closed live record
    (system:precedent, citing its cases), the firing rows linked to it, and the notifications.
    Returns (live_id, firing_ids)."""
    approve = decision.decision == "approve"
    cited = [{"kind": "precedent", "caseId": e.reference, "source": e.source, **dict(e.value or {})}
             for e in decision.evidence]
    with conn.cursor() as cur:
        live_id = conflict_live.insert_live(cur, lc, ctx=ctx, action=action, now=now,
                                            default_response_time=_default_response_time(), status="actioned",
                                            decision=decision.decision, actor=conflict_live.PRECEDENT,
                                            reason=decision.rationale, extra_evidence=cited)
        case = conflict_payload.case_id(live_id)
        firing_of = record_matches(cur, matched, verdict.notifies, result="allowed" if approve else "blocked",
                                   tool_name=tool_name, agent=agent, workflow_id=workflow_id, user_id=user_id,
                                   duration_ms=duration_ms, decision_id=live_id,
                                   reason=(f"Decided on precedent ({case})" if approve
                                           else f"{PRECEDENT_REFUSED}: decided on precedent ({case})"))
        fids = [firing_of[id(h)] for h in matched]
        _notify_precedent(cur, lc, firing_of, decision, case, tool_name)
    return live_id, fids


def _precedent_answer(decision, live_id: int, firing_ids: List[int]) -> GateResult:
    case = conflict_payload.case_id(live_id)
    if decision.decision == "approve":
        return GateResult(allow=True, to_agent={"result": "allowed", "conflictCaseId": case, "precedent": True},
                          firing_ids=firing_ids)
    n = len(decision.evidence)
    return GateResult(allow=False, firing_ids=firing_ids, to_agent={
        "result": "blocked", "reasonCode": PRECEDENT_REFUSED,
        "reason": f"People refused this same action {n} times before, so it was refused on precedent ({case}).",
        "conflictCaseId": case, "precedent": True})


def _open_cases(conn, verdict, firing_of, *, action, ctx, digest, tool_name, workflow_id, user_id,
                now, lc=None, live=None, note: Optional[str] = None) -> List[int]:
    """One case per matched approve policy; a still-open case for the same call is reused.

    Runs inside the gate's transaction. A transaction-scoped advisory lock on (tool, digest,
    workflow, requester) serialises two identical calls racing, so the second always finds the
    first's case instead of opening its own.

    `lc` (a live conflict of kind 'human' or 'auto') and `live` ({"id": ...}, filled in) come only
    with a conflict; without them this is exactly stage 3. 'auto' opens cases only for the
    standing rule's winners (lc.required, normal levels) and records the closed live case; the
    losers' firing rows wait on the winners' first case. 'human' records an open live case and
    routes each conflicting policy to its last level. A repeat takes the live id from any reused
    case that carries one (also when some members were decided meanwhile and are opened anew) and
    writes no second live record; one is written only when no reused case carries one.
    """
    conflict = lc is not None and lc.kind in ("human", "auto")
    needed = lc.required if conflict else verdict.approvals
    lock_key = _lock_key(tool_name, digest, workflow_id, user_id)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,))
    reused: Dict[str, Dict[str, Any]] = {}
    with conn.cursor() as cur:
        for hit in needed:
            found = _open_case_for(cur, key=str(hit["id"]), tool_name=tool_name, digest=digest,
                                   workflow_id=workflow_id, requested_by=user_id)
            if found:
                reused[str(hit["id"])] = found
    # new cases join the group of the call they repeat, else a fresh group
    group = None
    for found in reused.values():
        group = found["facts"].get("firing_group") or f"case:{found['decision_id']}"
        break
    group = group or uuid.uuid4().hex
    default_rt = _default_response_time()
    mapping = deciders.load_map(conn)
    extra: Dict[str, Any] = {}
    if conflict:
        live_id = next((f["facts"].get("liveConflict") for f in reused.values()
                        if f["facts"].get("liveConflict") is not None), None)
        # A repeat after one member was decided reuses only the still-open members; the new ones
        # join the live case those carry (one action is one live decision, user ruling Q4).
        if live_id is None and len(reused) < len(needed):
            with conn.cursor() as cur:
                if lc.kind == "human":
                    extra_live: Dict[str, Any] = {"history": conflict_history.raw(
                        cur, pair_key=conflict_detect.pair_key(*[str(h["id"]) for h in lc.involved]),
                        limit=conflict_history.IN_CASE_LIMIT)}
                    if note:
                        extra_live["precedent"] = {"why": note}
                    live_id = conflict_live.insert_live(cur, lc, ctx=ctx, action=action, now=now,
                                                        default_response_time=default_rt, status="open",
                                                        extra_facts=extra_live)
                else:
                    live_id = conflict_live.insert_live(cur, lc, ctx=ctx, action=action, now=now,
                                                        default_response_time=default_rt, status="actioned",
                                                        decision="standing_rule",
                                                        reason=conflict_live.rule_text(lc))
        live["id"] = live_id
        extra = {"liveConflict": live_id}
    out: List[int] = []
    for hit in needed:
        key = str(hit["id"])
        if key in reused:
            did = reused[key]["decision_id"]
            with conn.cursor() as cur:
                # link this attempt's row to the open request it repeats (decision columns only)
                cur.execute("UPDATE proc.bp_policy_firing SET decision_id = %s, reason = %s "
                            "WHERE firing_id = %s AND result = 'paused_for_approval'",
                            (did, f"Repeat call while request {did} is open; no new request opened",
                             firing_of[id(hit)]))
            logger.info("tool %s repeated while request %s is open; reusing it", tool_name, did)
            out.append(did)
            continue
        last_only = conflict and lc.kind == "human" and key in lc.last_level_only
        with conn.cursor() as cur:
            did = approvals._insert_case(
                cur, policy_doc=hit["policy"], firing_id=firing_of[id(hit)], action=action,
                requested_by=user_id, now=now, mapping=mapping, default_response_time=default_rt,
                extra_facts={"firing_group": group, "ctx": ctx, "argsDigest": digest, **extra},
                **({"last_level_only": True} if last_only else {}))
        out.append(did)
        with conn.cursor() as cur:
            (notify_last_level if last_only else notify_first_level)(cur, hit["policy"], firing_of[id(hit)],
                                                                     did, tool_name)
    if conflict and out:
        # the standing rule's losers need no approval of their own: their rows wait on the winners'
        losers = [h for h in verdict.approvals if not any(h is r for r in needed)]
        with conn.cursor() as cur:
            link_paused_notifies(cur, [firing_of[id(h)] for h in losers], out[0])
    return out
