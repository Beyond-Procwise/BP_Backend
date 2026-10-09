"""Run an approved action: the system replays the stored tool call once every approval is in.

Called by approvals.act() after an approve commits (``run(decision_id)``). The original chat
session is NOT resumed (design §3.8): the tool runs by name with its stored arguments, through
the same ``agentnick_control.build_tools`` the agent used, and the outcome is written down.

Order of work for one call:
0. Resolve the agent runtime first. Without one nothing is claimed or written, so a later
   run (once a runtime exists) can still proceed.
1. Find the case's group: every approval case sharing ``facts.firing_group`` (or the case alone).
   Any rejected/timed-out member -> never run. Any member still open -> wait (the last approval
   to commit is the one whose run proceeds).
2. Under a row lock on every member, re-read the group with a fresh statement (a concurrent run
   may have added a case while we waited) and check that no replay was already recorded for the
   group (exactly-once: two approvals finishing together run the tool once).
3. Re-check against the CURRENT live policies with the stored context:
   a block now matches -> do not run; a new approve policy the group did not cover -> open a
   case for it in the same group, in this same transaction, notify its first level, and stop
   (its approval calls run() again). Matched blocks and notifies get firing rows and
   notifications as at the gate (gate.record_matches), in the same transaction. If the policies
   cannot be loaded or checked, nothing is written and the retry sweeper tries again; only its
   last attempt records the action as not run and tells the approvers and the requester.
4. Claim the run by inserting the replay row (subject_type 'agent_policy_replay'), commit,
   then run the tool outside any lock and write the outcome onto that row. A crash mid-run
   leaves the claim behind: the action is run AT MOST once, never twice.

Where the outcome lives (controller ruling): the firing row records the HUMAN decision
('approved', set by act()); the replay outcome lives only on the agent_policy_replay row. Replay
never updates the decided firing rows (the re-check only inserts its own).

``run`` never raises. Logs carry exception TYPES and masked summaries only, never input values.
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set

from services.agent_policy import approvals, deciders, enforcement, gate, live_policies, replay_retry

logger = logging.getLogger(__name__)

SUBJECT_TYPE = "agent_policy_replay"
APPROVAL = approvals.SUBJECT_TYPE
ACTOR = "system:replay"
CHECKPOINT = "tool.call.before"
SUMMARY_LIMIT = 2000
ERROR_LIMIT = 500
BLOCKED_REASON = "A policy now forbids this action"
NO_RUNTIME = "no agent runtime available"
RECHECK_REASON = "Matched when the approved action was re-checked"


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
    except Exception as exc:  # noqa: BLE001
        logger.warning("replay could not reach the BackendScheduler for agent_nick: %s", type(exc).__name__)
        return None


def _load_policies() -> List[Dict[str, Any]]:
    """CURRENT live policies: own connection (a failed load never aborts our transaction), no cache."""
    return live_policies.load(None, ttl=0)


def _build_tools(agent_nick: Any, *, workflow_id: Optional[str], user_id: Optional[str]):
    from orchestration import agentnick_control

    return agentnick_control.build_tools(agent_nick, workflow_id=workflow_id, user_id=user_id)


def _stored_policy_docs(cur, members: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The compiled policy versions the group's cases were opened under (for their sensitive inputs)."""
    pairs = []
    for m in members:
        p = m["facts"].get("policy") or {}
        if p.get("id") and str(p.get("version") or "").isdigit():
            pairs.append((str(p["id"]), int(p["version"])))
    docs = []
    for key, version in sorted(set(pairs)):
        cur.execute("SELECT compiled FROM proc.bp_agent_policy_version WHERE policy_key = %s AND version = %s",
                    (key, version))
        row = cur.fetchone()
        doc = _facts(row[0]) if row else None
        if isinstance(doc, dict):
            docs.append(doc)
    return docs


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


def _sensitive_args(policies: Iterable[Any]) -> Set[str]:
    """Argument names any policy marks sensitive (``args.<name>`` inputs)."""
    out: Set[str] = set()
    for p in policies or []:
        if not isinstance(p, dict):
            continue
        for i in p.get("inputs") or []:
            if not isinstance(i, dict) or not i.get("sensitive"):
                continue
            f = str(i.get("field") or "")
            if f.startswith("args."):
                out.add(f[len("args."):])
    return out


def _scrub(value: Any, keys: Set[str]) -> Any:
    if isinstance(value, dict):
        return {k: (enforcement.MASK if k in keys else _scrub(v, keys)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v, keys) for v in value]
    return value


def _leaves(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for v in value.values():
            yield from _leaves(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _leaves(v)
    elif value is not None:
        s = str(value)
        if s:
            yield s


def mask_text(text: str, args: Dict[str, Any], sensitive: Set[str]) -> str:
    """Mask every sensitive argument value (any type, any length) where it stands as a token."""
    vals = sorted({s for name in sensitive for s in _leaves((args or {}).get(name))}, key=len, reverse=True)
    for s in vals:
        # a whole token, so a sensitive 5 masks "5" but never the 5 inside "2025"
        text = re.sub(r"(?<![\w.])" + re.escape(s) + r"(?![\w])", enforcement.MASK, text)
    return text


def summarise(result: Any, args: Dict[str, Any], sensitive: Set[str], limit: int = SUMMARY_LIMIT) -> str:
    """A capped text of the result with sensitive input values masked (by key and by value)."""
    try:
        text = result if isinstance(result, str) else json.dumps(_scrub(result, sensitive), default=str)
    except Exception:  # noqa: BLE001
        text = repr(_scrub(result, sensitive))
    return mask_text(text, args, sensitive)[:limit]


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


def _open_new_case(conn, cur, hit: Dict[str, Any], case: Dict[str, Any], members, group: str,
                   requested_by: Optional[str], mapping) -> int:
    """A paused firing row + an approval case, on the caller's cursor (no commit here)."""
    action = case["facts"].get("action") or {}
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
    # argsDigest (with requestedBy, which _insert_case stores) lets the gate's repeat-call reuse
    # find this case, as it finds a case the gate opened itself
    digest = case["facts"].get("argsDigest") or gate.args_digest(action.get("args") or {})
    did = approvals._insert_case(
        cur, policy_doc=hit["policy"], firing_id=firing_id, action=action, requested_by=requested_by,
        now=datetime.now(timezone.utc), mapping=mapping,
        extra_facts={"firing_group": group, "replayOf": [m["decision_id"] for m in members],
                     "ctx": _ctx(case["facts"]), "argsDigest": digest})
    gate.notify_first_level(cur, hit["policy"], firing_id, did, str(action.get("tool") or ""))
    return did


def _approvers(cur, members: List[Dict[str, Any]]) -> List[str]:
    """Who approved the group's cases (their sign-in names), in decision order."""
    cur.execute("SELECT DISTINCT ON (actioned_by) actioned_by FROM proc.bp_decision "
                "WHERE subject_type = %s AND subject_id = ANY(%s) AND decision = 'approve' "
                "AND actioned_by IS NOT NULL ORDER BY actioned_by, decision_id",
                (APPROVAL, [m["subject_id"] for m in members]))
    return [r[0] for r in cur.fetchall()]


def _check_unavailable(cur, decision_id: int, case: Dict[str, Any], members: List[Dict[str, Any]],
                       key: str, exc: BaseException) -> Dict[str, Any]:
    """The re-check could not load or evaluate the live policies. Never run unchecked; but an
    outage must not consume the claim either: nothing is written, so the retry sweeper
    (replay_retry, which counts its attempts in facts.replayAttempts) tries again. Only on its
    LAST attempt is the action recorded as not run and the approvers and requester told."""
    facts = case["facts"]
    attempts = int(facts.get("replayAttempts") or 0)
    if attempts < replay_retry.MAX_ATTEMPTS:
        logger.warning("replay re-check unavailable for decision %s (attempt %s of %s, will retry): %s",
                       decision_id, attempts, replay_retry.MAX_ATTEMPTS, type(exc).__name__)
        return {"status": "check_unavailable", "retry": True}
    logger.error("replay re-check unavailable for decision %s on the last attempt: %s",
                 decision_id, type(exc).__name__)
    rid = _give_up(cur, case, members, key, error=f"policy_check_unavailable: {type(exc).__name__}",
                   why="policy checks were unavailable")
    return {"status": "check_unavailable", "replayId": rid}


def _give_up(cur, case: Dict[str, Any], members: List[Dict[str, Any]], key: str, *, error: str, why: str) -> int:
    """The last retry attempt failed for a reason outside the action itself: record the replay as
    not run and tell the approvers and the requester, so an approved action never vanishes."""
    rid = _insert_replay(cur, key=key, case=case, members=members,
                         facts={"ok": False, "outcome": "not_run", "resultSummary": None, "error": error})
    facts = case["facts"]
    who = _approvers(cur, members) + [facts.get("requestedBy")]
    approvals._notify(cur, facts.get("firingId"), list(dict.fromkeys(w for w in who if w)),
                      f"{approvals._action_text(facts)} was approved but could not be run: {why}.",
                      case["decision_id"])
    return rid


def _no_runtime_give_up(decision_id: int) -> Dict[str, Any]:
    """No agent runtime was found. Normally nothing is written (a later run can proceed); but when
    this was the retry sweeper's LAST attempt there is no later run, so say so loudly (the I3 path)."""
    base = {"status": "no_runtime", "error": NO_RUNTIME}
    with _connect() as conn:
        prev = getattr(conn, "autocommit", False)
        conn.autocommit = False
        try:
            with conn.cursor() as cur:
                case = _load_case(cur, decision_id)
                if case is None or int(case["facts"].get("replayAttempts") or 0) < replay_retry.MAX_ATTEMPTS:
                    conn.commit()
                    return base
                if not _members(cur, case, lock=True):
                    conn.commit()
                    return base
                members = _members(cur, case, lock=False)
                if (any(m["status"] == "open" for m in members)
                        or any(_verdict_of(cur, m) != "approve" for m in members)):
                    conn.commit()
                    return base
                key = _replay_key(members)
                if _already_replayed(cur, key):
                    conn.commit()
                    return base
                logger.error("replay of decision %s gave up on its last attempt: %s", decision_id, NO_RUNTIME)
                rid = _give_up(cur, case, members, key, error=NO_RUNTIME, why="no agent runtime was available")
            conn.commit()
            return {**base, "replayId": rid}
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.autocommit = prev


def _record_recheck(cur, verdict, *, result: str, action: Dict[str, Any], requested_by: Optional[str],
                    duration_ms: int) -> Dict[int, int]:
    """Firing rows and notifications for the blocks and notifies matched at the re-check, the
    way the gate writes them (gate.record_matches), inside _claim's transaction. Approve
    policies the group already covers were decided by people and get no new row."""
    hits = list(verdict.blocks) + [h for h in verdict.notifies if not any(h is b for b in verdict.blocks)]
    if not hits:
        return {}
    return gate.record_matches(cur, hits, verdict.notifies, result=result,
                               tool_name=str(action.get("tool") or ""), agent=action.get("agent"),
                               workflow_id=action.get("workflowId"), user_id=requested_by,
                               duration_ms=duration_ms, reason=RECHECK_REASON)


# ---------------------------------------------------------------------------- run
def run(decision_id: int, *, agent_nick: Any = None) -> Dict[str, Any]:
    """Replay the approved action of case ``decision_id`` if its whole group is approved.

    Returns {"status": ...} describing what happened; never raises.
    """
    try:
        nick = _resolve_agent_nick(agent_nick)
        if nick is None:
            # nothing claimed, nothing written: a later run with a runtime can still proceed
            logger.warning("replay of decision %s not attempted: %s", decision_id, NO_RUNTIME)
            return _no_runtime_give_up(decision_id)
        with _connect() as conn:
            return _run(conn, int(decision_id), nick)
    except Exception as exc:  # noqa: BLE001
        # type only: a driver/DB message can quote stored arguments
        logger.error("replay of decision %s failed: %s", decision_id, type(exc).__name__)
        return {"status": "error", "error": type(exc).__name__}


def _run(conn, decision_id: int, agent_nick: Any) -> Dict[str, Any]:
    """_claim in ONE transaction (locks, re-check, new cases, claim), then the tool outside it."""
    prev = getattr(conn, "autocommit", False)
    conn.autocommit = False
    try:
        out = _claim(conn, decision_id)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.autocommit = prev
    if out.get("status") != "claimed":
        return out
    return _execute(conn, out, agent_nick)


def _claim(conn, decision_id: int) -> Dict[str, Any]:
    """Inside the caller's transaction: group check, exactly-once guard, re-check, claim."""
    with conn.cursor() as cur:
        case = _load_case(cur, decision_id)
        if case is None:
            return {"status": "not_found"}
        if not _members(cur, case, lock=True):
            return {"status": "not_found"}
        # Fresh statement AFTER the lock: a run that held the lock may have committed a new case
        # into this group, which the locking statement's snapshot cannot see.
        members = _members(cur, case, lock=False)
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
        started = time.monotonic()
        try:
            policies = _load_policies()
            verdict = enforcement.check(_ctx(facts), policies)
        except Exception as exc:  # noqa: BLE001 - fail closed: never run unchecked
            return _check_unavailable(cur, decision_id, case, members, key, exc)
        duration_ms = int((time.monotonic() - started) * 1000)

        covered = {str((m["facts"].get("policy") or {}).get("id") or "") for m in members}
        new = [h for h in verdict.approvals if str(h["id"]) not in covered]
        result = "blocked" if verdict.blocks else "paused_for_approval" if new else "allowed"
        firing_of = _record_recheck(cur, verdict, result=result, action=action,
                                    requested_by=facts.get("requestedBy"), duration_ms=duration_ms)

        if verdict.blocks:
            rid = _insert_replay(cur, key=key, case=case, members=members,
                                 facts={"ok": False, "outcome": "blocked", "resultSummary": None,
                                        "error": BLOCKED_REASON,
                                        "blockedBy": [h["id"] for h in verdict.blocks]})
            return {"status": "blocked", "replayId": rid}

        if new:
            mapping = deciders.load_map(conn)
            group = _effective_group(case)
            opened = [_open_new_case(conn, cur, h, case, members, group, facts.get("requestedBy"), mapping)
                      for h in new]
            # the re-check's notify rows wait with the group, as the gate's do
            gate.link_paused_notifies(cur, [firing_of[id(h)] for h in verdict.notifies], opened[0])
            return {"status": "new_approval_required", "caseIds": opened}

        sensitive = _sensitive_args(policies) | _sensitive_args(_stored_policy_docs(cur, members))
        rid = _insert_replay(cur, key=key, case=case, members=members,
                             facts={"ok": None, "outcome": "running", "resultSummary": None, "error": None})
        return {"status": "claimed", "replayId": rid, "action": action, "sensitive": sensitive}


def _execute(conn, claim: Dict[str, Any], agent_nick: Any) -> Dict[str, Any]:
    """Run the tool outside any lock and record the outcome on the claimed replay row."""
    action, rid = claim["action"], claim["replayId"]
    args = dict(action.get("args") or {})
    ok, summary, error = False, None, None
    try:
        tools = _build_tools(agent_nick, workflow_id=action.get("workflowId"), user_id=action.get("userId"))
        tool = next((t for t in tools if getattr(t, "name", None) == action.get("tool")), None)
        if tool is None:
            error = f"tool {action.get('tool')!r} is not available"
        else:
            result = tool.handler(**args)
            ok, summary = True, summarise(result, args, claim["sensitive"])
    except Exception as exc:  # noqa: BLE001
        error = summarise(f"{type(exc).__name__}: {exc}", args, claim["sensitive"], ERROR_LIMIT)
        # no exc_info: a traceback's message and locals can carry the raw arguments
        logger.warning("replayed tool %s failed (replay %s): %s", action.get("tool"), rid, error)

    facts = {"ok": ok, "outcome": "ran" if ok else "error", "resultSummary": summary, "error": error}
    try:
        _finish_replay(conn, rid, facts)
    except Exception as exc:  # noqa: BLE001
        logger.error("could not record replay outcome %s: %s", rid, type(exc).__name__)
    return {"status": "ran" if ok else "error", "replayId": rid, **facts}
