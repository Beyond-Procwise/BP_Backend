"""Build the `policy-conflict/1` case payload and the plain-language summary approvers read. Pure: no DB, no model, no clock."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from services.agent_policy import conditions
from services.agent_policy.conflict_detect import deciders_of

SCHEMA = "policy-conflict/1"
_PREFIX = "pc_"
_KINDS = {"policy": "policy_conflict", "live": "live_conflict"}


def case_id(decision_id: int) -> str:
    return f"{_PREFIX}{int(decision_id)}"


def parse_case_id(s: Any) -> Optional[int]:
    if not isinstance(s, str):
        return None
    digits = s[len(_PREFIX):] if s.startswith(_PREFIX) else s
    return int(digits) if digits.isascii() and digits.isdigit() else None


def _outcome(doc: Dict[str, Any]) -> Optional[str]:
    return (doc.get("enforcement") or {}).get("outcome")


def policy_entry(doc: Dict[str, Any]) -> Dict[str, Any]:
    area = doc.get("businessArea") or {}
    src = doc.get("source") or {}
    out: Dict[str, Any] = {
        "id": doc.get("id"), "version": doc.get("version"), "outcome": _outcome(doc),
        "situation": (doc.get("trigger") or {}).get("plain"),
    }
    if _outcome(doc) == "approve":
        out["deciders"] = list(deciders_of(doc))
    out["owner"] = doc.get("owner")
    out["businessArea"] = f"{area.get('primary')} / {area.get('subArea')}"
    out["source"] = {"document": src.get("document"), "reference": src.get("reference"),
                     "excerpt": src.get("excerpt")}
    return out


def _fields(docs: List[Dict[str, Any]]) -> set:
    out: set = set()
    for d in docs:
        out |= conditions.condition_fields((d.get("trigger") or {}).get("condition"))
    return out


def condition_args(docs: List[Dict[str, Any]], args: Dict[str, Any]) -> Dict[str, Any]:
    used = {f[len("args."):] for f in _fields(docs) if f.startswith("args.")}
    return {k: v for k, v in (args or {}).items() if k in used}


def condition_values(docs: List[Dict[str, Any]], flat_ctx: Dict[str, Any]) -> Dict[str, Any]:
    fields = _fields(docs)
    return {k: v for k, v in (flat_ctx or {}).items() if k in fields}


def why_line(policies: List[Dict[str, Any]]) -> str:
    outcomes = [_outcome(p) for p in policies]
    if len(policies) == 2 and sorted(o or "" for o in outcomes) == ["approve", "block"]:
        approver = policies[outcomes.index("approve")]
        return (f"One policy needs approval from {' then '.join(deciders_of(approver))}; "
                "the other does not allow this at all.")
    if len(policies) == 2 and outcomes == ["approve", "approve"]:
        a, b = (" then ".join(deciders_of(p)) for p in policies)
        return f"The policies name different approvers: {a} and {b}."
    return "The policies say different things about this action."


def policy_options(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    ia, ib = a.get("id"), b.get("id")
    keep = [f"keep_both:{ia}", f"keep_both:{ib}"]
    blocks = [i for i, d in ((ia, a), (ib, b)) if _outcome(d) == "block"]
    if len(blocks) == 1:   # Q1: a standing rule can never let a policy beat a block
        keep = [f"keep_both:{blocks[0]}"]
    out = list(keep)
    for verb in ("change", "limit", "retire"):
        out += [f"{verb}:{ia}", f"{verb}:{ib}"]
    return out


def scope_of(option: str) -> str:
    return "standing_rule" if str(option).startswith("keep_both:") else "this_action"


def _action_plain(kind: str, action: Optional[Dict[str, Any]], example: Dict[str, Any]) -> str:
    if kind == "live" and action and action.get("plain"):
        return action["plain"]
    bits = ", ".join(f"{k} {v}" for k, v in (example or {}).items())
    return f"Example action that triggers both policies: {bits}" if bits else "Example action that triggers both policies"


def build(kind: str, *, raised_at: str, policies: List[Dict[str, Any]], overlap_example: Dict[str, Any],
          standing_rules: List[Dict[str, Any]], prior: Dict[str, Any], options: List[str],
          respond_within: Optional[str], on_timeout: str, action: Optional[Dict[str, Any]] = None,
          case: Optional[str] = None) -> Dict[str, Any]:
    if kind not in _KINDS:
        raise ValueError(f"kind must be 'policy' or 'live', got {kind!r}")
    # why_line works on policy entries too: it needs outcome + deciders, both present there
    entry_docs = [{"enforcement": {"outcome": p.get("outcome"),
                                   "intervention": {"escalateTo": [{"name": n} for n in p.get("deciders") or []]}}}
                  for p in policies]
    summary = {
        "actionPlain": _action_plain(kind, action, overlap_example),
        "why": why_line(entry_docs),
        "policies": [{"id": p.get("id"), "situation": p.get("situation"), "outcome": p.get("outcome"),
                      "owner": p.get("owner"), "excerpt": (p.get("source") or {}).get("excerpt")}
                     for p in policies],
        "prior": dict(prior),
        "options": list(options),
        "respondWithin": respond_within,
    }
    payload: Dict[str, Any] = {"schema": SCHEMA, "caseId": case, "kind": kind, "raisedAt": raised_at}
    if kind == "live":
        payload["action"] = action
    payload.update({
        "policies": list(policies), "overlap": {"example": overlap_example},
        "standingRules": list(standing_rules), "priorDecisions": dict(prior),
        "options": list(options), "respondWithin": respond_within, "onTimeout": on_timeout,
        "summary": summary,
    })
    return payload


def to_columns(payload: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("action", "policies", "standingRules", "priorDecisions", "summary", "schema")
    return {
        "subject_type": _KINDS[payload["kind"]],
        "facts": {k: payload[k] for k in keys if k in payload},
        "evidence": [{"kind": "overlap", "example": payload["overlap"]["example"]}],
        "options": list(payload["options"]),
        "on_timeout": payload["onTimeout"],
    }


def returned_decision(row: Dict[str, Any]) -> Dict[str, Any]:
    return {"caseId": case_id(row["decision_id"]) if row.get("decision_id") is not None else None,
            "decision": row.get("decision"), "scope": row.get("decision_scope"),
            "decidedBy": row.get("actioned_by"), "decidedAt": row.get("actioned_at"),
            "reason": row.get("override_reason")}
