"""Conflict history: every decision on a conflict between agent policies, complete and readable.

One reader for every screen and export -- the policy page, the Conflicts detail, the approval
card of a paused clash and the CSV -- so they can never disagree (the stage 4 D1 class of bug).
An entry is one conflict case, newest first:

    {caseId, kind (policy|live), isOpen, raisedAt, policies [{id, version}], example,
     decision {option, scope, decidedBy {kind, name}, decidedAt, reason} | None,
     citedCases [caseId], proposal}

Who sees what (design §3.1): full reasons and example values go to anyone linked (decider map)
to an owner or a decider of any of the case's policies, and to the Admin role. Everyone else
reads the same entry with sensitive values replaced by the mask, in the example and wherever the
reason quotes one of the action's values: the reason is masked, never dropped.

raw() reads (with the private _owners/_deciders/_args masking needs), shown() masks and strips
them, read() is both for one viewer. Nothing here writes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from services.agent_policy import conflict_payload, deciders

SUBJECT_POLICY = "policy_conflict"     # conflict_cases.SUBJECT_POLICY (not imported: it pulls in approvals)
SUBJECT_LIVE = "live_conflict"
HISTORY_LIMIT = 1000                   # the newest cases a page or an export reads
IN_CASE_LIMIT = 20                     # the newest cases copied into a paused clash for its approvers
_PRIVATE = ("_owners", "_deciders", "_args")


@dataclass(frozen=True)
class Viewer:
    principal: Any = None
    is_admin: bool = False
    mapping: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)


ANONYMOUS = Viewer()


def viewer(conn, principal, *, is_admin: bool) -> Viewer:
    return Viewer(principal=principal, is_admin=bool(is_admin), mapping=deciders.load_map(conn))


def _av():
    from services.agent_policy import approval_views   # lazily: approval_views reaches the gate
    return approval_views


def _j(v: Any, default=None):
    if v is None:
        return default
    if isinstance(v, (str, bytes)):
        return json.loads(v) if v else default
    return v


def _iso(v: Any) -> Any:
    return v.isoformat() if isinstance(v, datetime) else v


_CASES_SQL = """
    SELECT c.decision_id, c.kind, c.is_open, c.policy_versions, c.created_at, c.outcome, c.decided_by,
           c.decided_at, d.facts, d.evidence
      FROM proc.bp_agent_policy_conflict c
      JOIN proc.bp_decision d ON d.decision_id = c.decision_id
     WHERE {where}
     ORDER BY c.created_at DESC, c.decision_id DESC
     LIMIT %s
"""
# The row that closed each case: a person's or the system's action row, or a live record closed
# when it was written (block, standing rule, precedent). Open originals have no actioned_by.
_ACTIONS_SQL = """
    SELECT facts->>'caseId', decision, decision_scope, actioned_by, actioned_at, override_reason,
           facts->'decidedBy', evidence
      FROM proc.bp_decision
     WHERE subject_type IN (%s, %s) AND status = 'actioned' AND actioned_by IS NOT NULL
       AND facts->>'caseId' = ANY(%s)
     ORDER BY actioned_at, decision_id
"""


def raw(cur, *, policy_key: Optional[str] = None, pair_key: Optional[str] = None,
        limit: int = HISTORY_LIMIT) -> List[Dict[str, Any]]:
    """Every conflict case naming the policy, or of exactly this pair, newest first, unmasked."""
    if (policy_key is None) == (pair_key is None):
        raise ValueError("give exactly one of policy_key or pair_key")
    where, arg = (("c.policy_keys @> ARRAY[%s]::text[]", policy_key) if policy_key is not None
                  else ("c.pair_key = %s", pair_key))
    cur.execute(_CASES_SQL.format(where=where), (arg, int(limit)))
    cases = cur.fetchall()
    ids = [conflict_payload.case_id(r[0]) for r in cases]
    acts: Dict[str, tuple] = {}
    if ids:
        cur.execute(_ACTIONS_SQL, (SUBJECT_POLICY, SUBJECT_LIVE, ids))
        for row in cur.fetchall():
            acts[row[0]] = tuple(row[1:])          # oldest first: the latest closing row wins
    return [_entry(r, cid, acts.get(cid)) for r, cid in zip(cases, ids)]


def _entry(row, cid: str, act: Optional[tuple]) -> Dict[str, Any]:
    _did, kind, is_open, versions, created, outcome, by, at, facts, evidence = row
    facts = _j(facts, {}) or {}
    example = _av().overlap_example(evidence)
    pols = [p for p in facts.get("policies") or [] if isinstance(p, dict)]
    decision: Optional[Dict[str, Any]] = None
    cited: List[str] = []
    if act is not None:
        option, scope, actor, acted_at, reason, decided, act_evidence = act
        decided = _j(decided, {}) or {}
        decision = {"option": option, "scope": scope, "decidedBy": {"kind": decided.get("kind"), "name": actor},
                    "decidedAt": _iso(acted_at), "reason": reason}
        cited = [str(e["caseId"]) for e in _j(act_evidence, []) or []
                 if isinstance(e, dict) and e.get("kind") == "precedent" and e.get("caseId")]
    elif not is_open and outcome is not None:
        # closed with no closing row the reader knows (written before it existed): the index says
        decision = {"option": outcome, "scope": None, "decidedBy": {"kind": None, "name": by},
                    "decidedAt": _iso(at), "reason": None}
    if kind == "live":
        args = dict(((facts.get("action") or {}).get("args")) or {})
    else:
        args = example_args(example)
    return {
        "caseId": cid, "kind": kind, "isOpen": bool(is_open), "raisedAt": _iso(created),
        "policies": [{"id": k, "version": int(v)} for k, v in sorted((_j(versions, {}) or {}).items())],
        "example": example, "decision": decision, "citedCases": cited, "proposal": facts.get("proposal"),
        **parties(pols),
        "_args": args,
    }


def parties(policies: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """{_owners, _deciders}: the names may_see() links a reader to, from a case's facts.policies."""
    pols = [p for p in policies or [] if isinstance(p, dict)]
    return {"_owners": sorted({str(p.get("owner")).strip() for p in pols if str(p.get("owner") or "").strip()}),
            "_deciders": sorted({str(n).strip() for p in pols for n in p.get("deciders") or []
                                 if str(n or "").strip()})}


def example_args(example: Dict[str, Any]) -> Dict[str, Any]:
    """A policy case's action values: the args.* fields of its (unmasked) overlap example."""
    return {k[len("args."):]: v for k, v in (example or {}).items() if k.startswith("args.")}


def mask_reason(reason: Any, args: Dict[str, Any], sensitive: Set[str]) -> Any:
    """Free text with every sensitive action value it quotes masked in place (never dropped)."""
    if not isinstance(reason, str):
        return reason
    arg_names = {f[len("args."):] for f in sensitive if f.startswith("args.")}
    return _av().mask_text(reason, dict(args or {}), arg_names)


def may_see(v: Viewer, entry: Dict[str, Any]) -> bool:
    """Admin, or linked to an owner or a decider of any of the case's policies."""
    if v.is_admin:
        return True
    names = list(entry.get("_owners") or []) + list(entry.get("_deciders") or [])
    return any(deciders.eligible(v.principal, n, v.mapping) for n in names)


def shown(entries: List[Dict[str, Any]], *, sensitive: Set[str],
          unmasked_for: Callable[[Dict[str, Any]], bool]) -> List[Dict[str, Any]]:
    """The entries as a reader may see them: private keys gone, and for an entry the reader may
    not see in full, the example and the reason masked. Never edits its input."""
    av = _av()
    out = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        view = {k: v for k, v in e.items() if k not in _PRIVATE}
        view["example"] = dict(e.get("example") or {})
        view["decision"] = dict(e["decision"]) if isinstance(e.get("decision"), dict) else None
        if not unmasked_for(e):
            view["example"] = av.mask_witness(view["example"], sensitive)
            d = view["decision"]
            if d:
                d["reason"] = mask_reason(d.get("reason"), dict(e.get("_args") or {}), sensitive)
        out.append(view)
    return out


def sensitive_of(cur, entries: List[Dict[str, Any]]) -> Set[str]:
    """Stage 3's union: every live policy's sensitive fields and those of the versions named."""
    pairs = [(p["id"], p["version"]) for e in entries for p in e.get("policies") or [] if p.get("id")]
    return _av().sensitive_for(cur, pairs)


def read(cur, *, policy_key: Optional[str] = None, pair_key: Optional[str] = None, viewer: Viewer,
         limit: int = HISTORY_LIMIT) -> List[Dict[str, Any]]:
    entries = raw(cur, policy_key=policy_key, pair_key=pair_key, limit=limit)
    return shown(entries, sensitive=sensitive_of(cur, entries), unmasked_for=lambda e: may_see(viewer, e))
