"""Run an approved action: the system replays the stored tool call once every approval is in.

Called by approvals.act() after an approve commits (``run(decision_id)``). The original chat
session is NOT resumed (design §3.8): the tool runs by name with its stored arguments, through
the same ``agentnick_control.build_tools`` the agent used, and the outcome is written down.

Order of work for one call:
1. Find the case's group: every approval case sharing ``facts.firing_group`` (or the case alone).
   Any rejected/timed-out member -> never run. Any member still open -> wait (the last approval
   to commit is the one whose run proceeds).
2. Under a row lock on every member, check that no replay was already recorded for the group
   (exactly-once: two approvals finishing together run the tool once).
3. Re-check against the CURRENT live policies with the stored context:
   a block now matches -> do not run; a new approve policy the group did not cover -> open a
   case for it in the same group and stop (its approval calls run() again).
4. Claim the run by inserting the replay row (subject_type 'agent_policy_replay'), commit,
   then run the tool outside any lock and write the outcome onto that row. A crash mid-run
   leaves the claim behind: the action is run AT MOST once, never twice.

The firing log is append-only: a firing row is only touched while it is still
``paused_for_approval`` (act() has normally moved it to 'approved' already), and only through
the decision columns the guard trigger allows.

``run`` never raises: every failure is logged and, where possible, recorded.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from services.agent_policy import approvals, enforcement, live_policies

logger = logging.getLogger(__name__)

SUBJECT_TYPE = "agent_policy_replay"
APPROVAL = approvals.SUBJECT_TYPE
ACTOR = "system:replay"
CHECKPOINT = "tool.call.before"
SUMMARY_LIMIT = 2000
BLOCKED_REASON = "A policy now forbids this action"
NO_RUNTIME = "no agent runtime available"


# ---------------------------------------------------------------------------- seams
def _connect():
    from services.db import get_conn

    return get_conn()


def _resolve_agent_nick(agent_nick: Any) -> Any:
    """The caller's agent_nick, else the BackendScheduler singleton's (app.state shares it)."""
    if agent_nick is not None:
        return agent_nick
    try:
        from services.backend_scheduler import BackendScheduler

        inst = BackendScheduler._instance
        return getattr(inst, "agent_nick", None) if inst is not None else None
    except Exception:  # noqa: BLE001
        logger.exception("replay could not reach the BackendScheduler for agent_nick")
        return None


def _load_policies() -> List[Dict[str, Any]]:
    return live_policies.load(None, ttl=0)


def _build_tools(agent_nick: Any, *, workflow_id: Optional[str], user_id: Optional[str]):
    from orchestration import agentnick_control

    return agentnick_control.build_tools(agent_nick, workflow_id=workflow_id, user_id=user_id)


# ---------------------------------------------------------------------------- helpers
def _facts(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes)) and raw:
        return json.loads(raw)
    return {}


def _group_of(facts: Dict[str, Any]) -> Optional[str]:
    g = facts.get("firing_group") or facts.get("firingGroup")
    return str(g) if g else None


def _effective_group(case: Dict[str, Any]) -> str:
    """The stored firing_group, else 'case:<id>' (the group a re-check opens for a lone case)."""
    return _group_of(case["facts"]) or f"case:{case['decision_id']}"


def _ctx(facts: Dict[str, Any]) -> Dict[str, Any]:
    """The context the gate checked: stored ctx when present, else rebuilt from the action."""
    stored = facts.get("ctx")
    if isinstance(stored, dict) and stored:
        ctx = dict(stored)
        ctx.setdefault("checkpoint", CHECKPOINT)
        return ctx
    action = facts.get("action") or {}
    return {"checkpoint": CHECKPOINT, "tool.name": action.get("tool"),
            "agent.name": action.get("agent"), "agent.reason": action.get("reason"),
            "args": dict(action.get("args") or {})}


def _sensitive_args(policies: Iterable[Dict[str, Any]]) -> Set[str]:
    """Argument names any policy marks sensitive (``args.<name>`` inputs)."""
    out: Set[str] = set()
    for p in policies or []:
        for i in (p.get("inputs") or []) if isinstance(p, dict) else []:
            f = str((i or {}).get("field") or "") if isinstance(i, dict) else ""
            if i.get("sensitive") and f.startswith("args."):
                out.add(f[len("args."):])
    return out


def _scrub(value: Any, keys: Set[str]) -> Any:
    if isinstance(value, dict):
        return {k: (enforcement.MASK if k in keys else _scrub(v, keys)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v, keys) for v in value]
    return value


def summarise(result: Any, args: Dict[str, Any], sensitive: Set[str]) -> str:
    """A <=2000-char text of the result with sensitive input values masked."""
    try:
        text = result if isinstance(result, str) else json.dumps(_scrub(result, sensitive), default=str)
    except Exception:  # noqa: BLE001
        text = repr(result)
    for name in sensitive:
        val = (args or {}).get(name)
        if val is None:
            continue
        s = str(val)
        if len(s) >= 3 or isinstance(val, str) and s:
            text = text.replace(s, enforcement.MASK)
    return text[:SUMMARY_LIMIT]


def _load_case(cur, decision_id: int) -> Optional[Dict[str, Any]]:
    cur.execute("SELECT decision_id, subject_id, facts, workflow_id FROM proc.bp_decision "
                "WHERE decision_id = %s AND subject_type = %s", (decision_id, APPROVAL))
    row = cur.fetchone()
    if not row:
        return None
    return {"decision_id": int(row[0]), "subject_id": row[1], "facts": _facts(row[2]), "workflow_id": row[3]}


def _members(cur, case: Dict[str, Any], *, lock: bool) -> List[Dict[str, Any]]:
    """The original cases of the group (decision 'approve_or_reject'), in id order."""
    group = _effective_group(case)
    ids = [case["decision_id"]]
    if group.startswith("case:") and group[5:].isdigit():
        ids.append(int(group[5:]))       # a lone case that a re-check later joined with a new one
    cur.execute("SELECT decision_id, subject_id, facts, status FROM proc.bp_decision "
                "WHERE subject_type = %s AND decision = 'approve_or_reject' "
                "AND (facts->>'firing_group' = %s OR facts->>'firingGroup' = %s OR decision_id = ANY(%s)) "
                "ORDER BY decision_id" + (" FOR UPDATE" if lock else ""),
                (APPROVAL, group, group, ids))
    return [{"decision_id": int(r[0]), "subject_id": r[1], "facts": _facts(r[2]), "status": r[3]}
            for r in cur.fetchall()]


def _verdict_of(cur, member: Dict[str, Any]) -> str:
    """'open' | 'approve' | 'reject' — the latest human/timeout action recorded for the case."""
    if member["status"] == "open":
        return "open"
    cur.execute("SELECT decision FROM proc.bp_decision WHERE subject_type = %s AND subject_id = %s "
                "AND decision IN ('approve','reject') AND actioned_by IS NOT NULL "
                "ORDER BY decision_id DESC LIMIT 1", (APPROVAL, member["subject_id"]))
    row = cur.fetchone()
    return row[0] if row else "reject"   # closed with no recorded approval never counts as approved


def _replay_key(members: List[Dict[str, Any]]) -> str:
    return members[0]["subject_id"]      # lowest decision_id: the same for every caller


def _already_replayed(cur, key: str) -> bool:
    cur.execute("SELECT 1 FROM proc.bp_decision WHERE subject_type = %s AND subject_id = %s LIMIT 1",
                (SUBJECT_TYPE, key))
    return cur.fetchone() is not None


def _insert_replay(cur, *, key: str, case: Dict[str, Any], members: List[Dict[str, Any]],
                   facts: Dict[str, Any]) -> int:
    action = case["facts"].get("action") or {}
    full = {"caseIds": [m["decision_id"] for m in members], "firingGroup": _group_of(case["facts"]),
            "tool": action.get("tool"), **facts}
    cur.execute(
        """
        INSERT INTO proc.bp_decision (subject_type, subject_id, decision, resolution, rationale,
            policy_name, facts, evidence, status, actioned_by, actioned_at, workflow_id, agent, created_by)
        VALUES (%s,%s,'replay','resolved',%s,%s,%s,'[]','actioned',%s,%s,%s,%s,%s)
        RETURNING decision_id
        """,
        (SUBJECT_TYPE, key, f"Replay of {action.get('tool') or 'an action'} after approval",
         ",".join(str((m["facts"].get("policy") or {}).get("id") or "") for m in members),
         json.dumps(full, default=str), ACTOR, datetime.now(timezone.utc), case["workflow_id"],
         action.get("agent"), ACTOR),
    )
    return int(cur.fetchone()[0])


def _finish_replay(conn, replay_id: int, facts: Dict[str, Any]) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_decision SET facts = facts || %s::jsonb, actioned_at = %s "
                    "WHERE decision_id = %s AND subject_type = %s",
                    (json.dumps(facts, default=str), datetime.now(timezone.utc), replay_id, SUBJECT_TYPE))


def _firing_ids(members: List[Dict[str, Any]]) -> List[int]:
    return [int(m["facts"]["firingId"]) for m in members
            if str(m["facts"].get("firingId") or "").isdigit()]


def _update_paused_firings(cur, firing_ids: List[int], *, result: str, reason: str) -> None:
    """Only decision columns, only on rows still paused (the guard trigger refuses anything else)."""
    if not firing_ids:
        return
    cur.execute("UPDATE proc.bp_policy_firing SET result = %s, decided_by = %s, decided_at = %s, reason = %s "
                "WHERE firing_id = ANY(%s) AND result = 'paused_for_approval'",
                (result, ACTOR, datetime.now(timezone.utc), reason, firing_ids))


def _open_new_case(conn, cur, hit: Dict[str, Any], case: Dict[str, Any], members, group: str,
                   requested_by: Optional[str]) -> int:
    action = case["facts"].get("action") or {}
    policy = hit["policy"]
    cur.execute(
        "INSERT INTO proc.bp_policy_firing (policy_key, policy_version, checkpoint, action_name, agent, "
        "workflow_id, requested_by, outcome, result, matched_values, missing_inputs, reason) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,'approve','paused_for_approval',%s,%s,%s) RETURNING firing_id",
        (str(hit["id"]), int(hit.get("version") or 0), CHECKPOINT, str(action.get("tool") or ""),
         action.get("agent"), action.get("workflowId"), requested_by,
         json.dumps(hit.get("matched_values") or {}, default=str), list(hit.get("missing") or []),
         "Approval newly required when the approved action was re-checked"),
    )
    firing_id = int(cur.fetchone()[0])
    # open_case commits this connection's transaction (its own _tx on a non-autocommit
    # connection); the member locks are held until then, so a concurrent run sees the new case.
    return approvals.open_case(
        conn, policy_doc=policy, firing_id=firing_id, action=action, requested_by=requested_by,
        now=datetime.now(timezone.utc),
        extra_facts={"firing_group": group, "replayOf": [m["decision_id"] for m in members],
                     "ctx": _ctx(case["facts"])})


# ---------------------------------------------------------------------------- run
def run(decision_id: int, *, agent_nick: Any = None) -> Dict[str, Any]:
    """Replay the approved action of case ``decision_id`` if its whole group is approved.

    Returns {"status": ...} describing what happened; never raises.
    """
    try:
        with _connect() as conn:
            return _run(conn, int(decision_id), agent_nick)
    except Exception as exc:  # noqa: BLE001
        logger.exception("replay of decision %s failed", decision_id)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"[:500]}


def _run(conn, decision_id: int, agent_nick: Any) -> Dict[str, Any]:
    prev = getattr(conn, "autocommit", False)
    conn.autocommit = False
    try:
        out = _claim(conn, decision_id)
    except BaseException:
        conn.rollback()
        raise
    finally:
        try:
            conn.commit()
        except Exception:  # noqa: BLE001 - already rolled back / closed
            pass
        conn.autocommit = prev
    if out.get("status") != "claimed":
        return out
    return _execute(conn, out, agent_nick)


def _claim(conn, decision_id: int) -> Dict[str, Any]:
    """Inside one transaction: group check, exactly-once guard, re-check, claim."""
    with conn.cursor() as cur:
        case = _load_case(cur, decision_id)
        if case is None:
            return {"status": "not_found"}
        members = _members(cur, case, lock=True)
        if not members:
            return {"status": "not_found"}
        verdicts = {m["decision_id"]: _verdict_of(cur, m) for m in members}
        if any(v == "reject" for v in verdicts.values()):
            return {"status": "rejected", "cases": verdicts}
        if any(v == "open" for v in verdicts.values()):
            return {"status": "waiting", "cases": verdicts}
        key = _replay_key(members)
        if _already_replayed(cur, key):
            return {"status": "already_replayed"}

        facts = case["facts"]
        action = facts.get("action") or {}
        ctx = _ctx(facts)
        try:
            # own connection (a failed load must not abort this transaction), no cache: CURRENT
            policies = _load_policies()
            verdict = enforcement.check(ctx, policies)
        except Exception as exc:  # noqa: BLE001 - fail closed: never run unchecked
            logger.exception("replay re-check failed for decision %s", decision_id)
            rid = _insert_replay(cur, key=key, case=case, members=members,
                                 facts={"ok": False, "outcome": "not_run", "resultSummary": None,
                                        "error": f"policy_check_unavailable: {type(exc).__name__}"})
            return {"status": "check_unavailable", "replayId": rid}

        if verdict.blocks:
            rid = _insert_replay(cur, key=key, case=case, members=members,
                                 facts={"ok": False, "outcome": "blocked", "resultSummary": None,
                                        "error": BLOCKED_REASON,
                                        "blockedBy": [h["id"] for h in verdict.blocks]})
            _update_paused_firings(cur, _firing_ids(members), result="rejected", reason=BLOCKED_REASON)
            return {"status": "blocked", "replayId": rid}

        covered = {str((m["facts"].get("policy") or {}).get("id") or "") for m in members}
        new = [h for h in verdict.approvals if str(h["id"]) not in covered]
        if new:
            group = _effective_group(case)
            opened = [_open_new_case(conn, cur, h, case, members, group, facts.get("requestedBy"))
                      for h in new]
            return {"status": "new_approval_required", "caseIds": opened}

        sensitive = _sensitive_args(policies)
        rid = _insert_replay(cur, key=key, case=case, members=members,
                             facts={"ok": None, "outcome": "running", "resultSummary": None, "error": None})
        return {"status": "claimed", "replayId": rid, "action": action, "sensitive": sensitive,
                "firingIds": _firing_ids(members)}


def _execute(conn, claim: Dict[str, Any], agent_nick: Any) -> Dict[str, Any]:
    """Run the tool outside any lock and record the outcome on the claimed replay row."""
    action, rid = claim["action"], claim["replayId"]
    args = dict(action.get("args") or {})
    ok, summary, error = False, None, None
    try:
        nick = _resolve_agent_nick(agent_nick)
        if nick is None:
            error = NO_RUNTIME
        else:
            tools = _build_tools(nick, workflow_id=action.get("workflowId"), user_id=action.get("userId"))
            tool = next((t for t in tools if getattr(t, "name", None) == action.get("tool")), None)
            if tool is None:
                error = f"tool {action.get('tool')!r} is not available"
            else:
                result = tool.handler(**args)
                ok, summary = True, summarise(result, args, claim["sensitive"])
    except Exception as exc:  # noqa: BLE001
        logger.exception("replayed tool %s failed (replay %s)", action.get("tool"), rid)
        error = summarise(f"{type(exc).__name__}: {exc}", args, claim["sensitive"])[:500]

    facts = {"ok": ok, "outcome": "ran" if ok else ("not_run" if error == NO_RUNTIME else "error"),
             "resultSummary": summary, "error": error}
    try:
        _finish_replay(conn, rid, facts)
        if not ok:
            with conn.cursor() as cur:
                _update_paused_firings(cur, claim["firingIds"], result="error", reason=error or "error")
    except Exception:  # noqa: BLE001
        logger.exception("could not record replay outcome %s", rid)
    return {"status": "ran" if ok else "error", "replayId": rid, **facts}
