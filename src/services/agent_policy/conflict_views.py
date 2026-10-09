"""What the conflicts screen reads (stage 4, Task 8): policy conflict cases, read-only.

Only POLICY cases (subject_type 'policy_conflict', decision 'resolve_conflict') are listed or
read here; a live case is read through its member approval cases (approval_views' conflict
block), and an action row is never a case.

Who decides (user ruling Q3): anyone linked to EITHER owner name; Admin is not automatic.
Masking (global constraint, stage 3 rule): the overlap example is the witness, and it can hold
an action's own values (raise_block_pairs and maybe_propose copy a live action's condition
values into it). A sensitive field shows enforcement.MASK unless the caller may decide the case
(linked to an owner; status aside, as approval_views.can_decide). The sensitive set is stage 3's:
the union over the case's own policy versions and every live policy (approval_views.sensitive_for).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.agent_policy import approval_views as AV
from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_history as CH
from services.agent_policy import conflict_payload as CP
from services.agent_policy import deciders

CASE_LIMIT = 200
_COLS = ("d.decision_id, d.subject_id, d.rationale, d.facts, d.evidence, d.status, d.options, d.created_at, "
         "c.raised_by")
_WHERE = {"open": "AND d.status = 'open'", "closed": "AND d.status <> 'open'", "all": ""}


def _case(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    row["facts"] = AV._j(row.get("facts"), {}) or {}
    row["options"] = AV._j(row.get("options"), []) or []
    return row


def _select(cur, where: str, params: tuple, limit: int) -> List[Dict[str, Any]]:
    cur.execute(f"SELECT {_COLS} FROM proc.bp_decision d "
                "LEFT JOIN proc.bp_agent_policy_conflict c ON c.decision_id = d.decision_id "
                f"WHERE d.subject_type = %s AND d.decision = 'resolve_conflict' {where} "
                "ORDER BY d.created_at DESC, d.decision_id DESC LIMIT %s",
                (CC.SUBJECT_POLICY, *params, limit))
    return [_case(r) for r in AV._rows(cur)]


def may_decide(principal, case: Dict[str, Any], mapping: deciders.Mapping) -> bool:
    """Linked to either owner (status aside: masking uses this too)."""
    return any(deciders.eligible(principal, o, mapping) for o in CC._case_owners(case["facts"]))


def _pairs(cases: List[Dict[str, Any]]):
    out = []
    for c in cases:
        for p in c["facts"].get("policies") or []:
            v = (p or {}).get("version")
            if (p or {}).get("id") and str(v or "").isdigit():
                out.append((str(p["id"]), int(v)))
    return out


def _heads(cur, keys: List[str]) -> Dict[str, Dict[str, Any]]:
    if not keys:
        return {}
    cur.execute("SELECT policy_key, latest_version, status FROM proc.bp_agent_policy WHERE policy_key = ANY(%s)",
                (sorted(set(keys)),))
    return {r[0]: {"latestVersion": int(r[1]) if r[1] is not None else None, "status": r[2]}
            for r in cur.fetchall()}


def _actions(cur, cases: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """{case decision_id: its action rows as returned decisions, oldest first}."""
    ids = [CP.case_id(c["decision_id"]) for c in cases]
    if not ids:
        return {}
    cur.execute("SELECT decision_id, facts->>'caseId', decision, decision_scope, actioned_by, actioned_at, "
                "override_reason FROM proc.bp_decision WHERE subject_type = %s AND status = 'actioned' "
                "AND actioned_by IS NOT NULL AND decision <> 'resolve_conflict' AND facts->>'caseId' = ANY(%s) "
                "ORDER BY decision_id", (CC.SUBJECT_POLICY, ids))
    out: Dict[int, List[Dict[str, Any]]] = {}
    for action_id, cid, decision, scope, by, at, reason in cur.fetchall():
        did = CP.parse_case_id(cid)
        got = CP.returned_decision({"decision_id": did, "decision": decision, "decision_scope": scope,
                                    "actioned_by": by, "actioned_at": AV._iso(at), "override_reason": reason})
        got["actionId"] = int(action_id)
        out.setdefault(did, []).append(got)
    return out


def view(case: Dict[str, Any], *, unmasked: bool, decidable: bool, sensitive, heads: Dict[str, Dict[str, Any]],
         actions: List[Dict[str, Any]]) -> Dict[str, Any]:
    facts = case["facts"]
    example = AV.overlap_example(case.get("evidence"))
    if not unmasked:
        example = AV.mask_witness(example, sensitive)
    policies = []
    for p in facts.get("policies") or []:
        p = dict(p or {})
        head = heads.get(str(p.get("id"))) or {}
        p["latestVersion"], p["status"] = head.get("latestVersion"), head.get("status")
        policies.append(p)
    options = list(case["options"])
    return {
        "caseId": CP.case_id(case["decision_id"]),
        "decisionId": int(case["decision_id"]),
        "pairKey": case.get("subject_id"),
        "status": case["status"],
        "raisedAt": AV._iso(case.get("created_at")),
        "raisedBy": case.get("raised_by"),
        "policies": policies,
        "example": example,
        # rebuilt from the (masked) example: the stored summary line quotes the raw values
        "actionPlain": CP._action_plain("policy", None, example),
        "why": case.get("rationale") or (facts.get("summary") or {}).get("why"),
        "prior": facts.get("priorDecisions"),
        "options": options,
        "optionLabels": {o: CC.option_label(o) for o in options},
        "respondWithin": (facts.get("summary") or {}).get("respondWithin"),
        "unroutable": list(facts.get("unroutable") or []),
        "proposal": facts.get("proposal"),
        "canDecide": bool(decidable and case["status"] == "open"),
        "decision": actions[-1] if actions else None,
    }


def _shown_actions(case: Dict[str, Any], actions: List[Dict[str, Any]], *, full: bool,
                   sensitive) -> List[Dict[str, Any]]:
    """The case's action rows as this reader may see them: a reason quoting a sensitive value of the
    case's example is masked in place unless the reader sees the history in full (conflict_history's
    viewer rule, design §3.1), so `decision`, `history` and `conflictHistory` always agree."""
    if full:
        return actions
    args = CH.example_args(AV.overlap_example(case.get("evidence")))
    return [{**a, "reason": CH.mask_reason(a.get("reason"), args, sensitive)} for a in actions]


def _views(cur, cases: List[Dict[str, Any]], principal, mapping, *, is_admin: bool = False,
           with_history: bool = False) -> List[Dict[str, Any]]:
    sensitive = AV.sensitive_for(cur, _pairs(cases))
    heads = _heads(cur, [str((p or {}).get("id")) for c in cases for p in c["facts"].get("policies") or []])
    acts = _actions(cur, cases)
    reader = CH.Viewer(principal, bool(is_admin), mapping)
    out = []
    for c in cases:
        ok = may_decide(principal, c, mapping)
        full = CH.may_see(reader, CH.parties(c["facts"].get("policies") or []))
        shown = _shown_actions(c, acts.get(int(c["decision_id"]), []), full=full, sensitive=sensitive)
        v = view(c, unmasked=ok, decidable=ok, sensitive=sensitive, heads=heads, actions=shown)
        if with_history:
            v["history"] = shown
        out.append(v)
    return out


def list_conflicts(conn, principal, *, status: str = "open", limit: int = CASE_LIMIT,
                   is_admin: bool = False) -> List[Dict[str, Any]]:
    """Policy cases, newest first; canDecide and masking per caller."""
    mapping = deciders.load_map(conn)
    with conn.cursor() as cur:
        cases = _select(cur, _WHERE[status], (), limit)
        return _views(cur, cases, principal, mapping, is_admin=is_admin)


def get_conflict(conn, decision_id: int, principal, *, is_admin: bool = False) -> Optional[Dict[str, Any]]:
    """One policy case with its history (every action row, oldest first) and the whole history of
    its pair through the one reader; None for a live case, an action row or an unknown id."""
    mapping = deciders.load_map(conn)
    with conn.cursor() as cur:
        cases = _select(cur, "AND d.decision_id = %s", (decision_id,), 1)
        if not cases:
            return None
        [out] = _views(cur, cases, principal, mapping, is_admin=is_admin, with_history=True)
        out["conflictHistory"] = CH.read(cur, pair_key=str(cases[0]["subject_id"]),
                                         viewer=CH.Viewer(principal, bool(is_admin), mapping))
    return out
