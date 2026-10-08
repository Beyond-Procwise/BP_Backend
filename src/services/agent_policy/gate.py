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

from services.agent_policy import approvals, deciders, enforcement, live_policies, settings

logger = logging.getLogger(__name__)

CHECKPOINT = "tool.call.before"
REASON_MAX = 500
UNAVAILABLE = {"result": "blocked", "reasonCode": "policy_check_unavailable",
               "reason": "Policy checks are unavailable, so this action was not run."}


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


def _row_result(hit: Dict[str, Any], verdict: enforcement.Verdict) -> str:
    """Every matched policy's row records the action's fate: the verdict's overall result.

    A notify row of a paused call is therefore 'paused_for_approval' too; the gate links it to
    the call's first case so that case's decision settles it (approvals._update_firing).
    """
    return verdict.result


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
    logger.exception("agent policy check unavailable for tool %s", tool_name, exc_info=exc)
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

    # One transaction for every row this call writes (firing rows, notifications, cases): on any
    # failure nothing is left behind -- never an open case for a call the agent was refused.
    with _connect() as conn:
        with approvals._tx(conn):
            firing_of: Dict[int, int] = {}
            with conn.cursor() as cur:
                for hit in matched:
                    fid = _insert_firing(
                        cur, key=str(hit["id"]), version=int(hit.get("version") or 0),
                        tool_name=tool_name, agent=agent, workflow_id=workflow_id, user_id=user_id,
                        outcome=hit["outcome"], result=_row_result(hit, verdict),
                        matched_values=hit.get("matched_values"), missing=hit.get("missing"),
                        duration_ms=duration_ms)
                    firing_of[id(hit)] = fid
                    firing_ids.append(fid)

                # notify recipients hear about it whatever the result (block carrying a notify list too)
                for hit in verdict.notifies:
                    key = str(hit["id"])
                    msg = (f"{_what(hit['policy'], tool_name)} (policy {key}) "
                           f"{_OUTCOME_TEXT.get(verdict.result, 'was checked')}.")
                    _notify(cur, firing_of[id(hit)], hit.get("notify"), msg, f"agent-policy:{key}")

            if verdict.result == "paused_for_approval":
                case_ids = _open_cases(conn, verdict, firing_of, action=action, ctx=ctx, digest=digest,
                                       tool_name=tool_name, workflow_id=workflow_id, user_id=user_id,
                                       now=now)
                # A notify row of a paused call is itself paused (_row_result) and linked to the
                # group's first case, so the decision that settles the call settles it too
                # (approvals._update_firing settles every row linked by decision_id).
                notify_rows = [firing_of[id(h)] for h in verdict.notifies if h["outcome"] != "approve"]
                if notify_rows and case_ids:
                    with conn.cursor() as cur:
                        cur.execute("UPDATE proc.bp_policy_firing SET decision_id = %s "
                                    "WHERE firing_id = ANY(%s) AND result = 'paused_for_approval'",
                                    (case_ids[0], notify_rows))

    if verdict.result == "allowed":
        return GateResult(allow=True, firing_ids=firing_ids)
    to_agent = dict(verdict.to_agent or {})
    if verdict.result == "paused_for_approval":
        to_agent["requestIds"] = list(case_ids)
    return GateResult(allow=False, to_agent=to_agent, firing_ids=firing_ids, case_ids=case_ids)


def _open_cases(conn, verdict, firing_of, *, action, ctx, digest, tool_name, workflow_id, user_id,
                now) -> List[int]:
    """One case per matched approve policy; a still-open case for the same call is reused.

    Runs inside the gate's transaction. A transaction-scoped advisory lock on (tool, digest,
    workflow, requester) serialises two identical calls racing, so the second always finds the
    first's case instead of opening its own.
    """
    lock_key = f"agent_policy_gate:{tool_name}:{digest}:{workflow_id}:{user_id}"
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,))
    reused: Dict[str, Dict[str, Any]] = {}
    with conn.cursor() as cur:
        for hit in verdict.approvals:
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
    out: List[int] = []
    for hit in verdict.approvals:
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
        with conn.cursor() as cur:
            did = approvals._insert_case(
                cur, policy_doc=hit["policy"], firing_id=firing_of[id(hit)], action=action,
                requested_by=user_id, now=now, mapping=mapping, default_response_time=default_rt,
                extra_facts={"firing_group": group, "ctx": ctx, "argsDigest": digest})
        out.append(did)
        # the first level hears that a decision is waiting (escalations notify the next)
        levels = [str(e.get("name")).strip() for e in
                  ((hit["policy"].get("enforcement") or {}).get("intervention") or {}).get("escalateTo") or []
                  if isinstance(e, dict) and str(e.get("name") or "").strip()]
        if levels:
            with conn.cursor() as cur:
                _notify(cur, firing_of[id(hit)], [levels[0]],
                        f"{_what(hit['policy'], tool_name)} (policy {key}) needs your decision.",
                        f"decision:{did}")
    return out
