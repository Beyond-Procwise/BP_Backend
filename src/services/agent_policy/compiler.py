"""Form state -> hard-policy/2. Pure: no database, no model, no clock, no mutation of the input.

Only the current outcome's fields are emitted, so switching outcome in the form can never
leave a stale level, response time or approver block in the JSON.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

_ON_MATCH = {"approve": "paused_for_approval", "block": "blocked", "notify": "allowed"}


def on_missing_for(form: Dict[str, Any], settings: Dict[str, Any]) -> str:
    explicit = (form.get("hidden") or {}).get("onMissingData")
    if explicit:
        return explicit
    return (settings.get("on_missing_data") or {}).get(form.get("outcome") or "", "fail_closed")


def response_time(form: Dict[str, Any], settings: Dict[str, Any]) -> Tuple[str, str]:
    override = form.get("responseTime")
    if override:
        return override, "policy"
    return settings["response_time"], "company_default"


def _roles(names: List[str]) -> List[Dict[str, str]]:
    return [{"type": "role", "name": n} for n in names if str(n).strip()]


def compile_policy(form: Dict[str, Any], *, policy_key: str, version: int, status: str,
                   settings: Dict[str, Any], never_suggest: bool,
                   conflicts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    f = copy.deepcopy(form)
    h = f.get("hidden") or {}
    outcome = f.get("outcome")
    inputs = h.get("inputs") or []
    checkpoint = h.get("checkpoint")
    limit = f.get("limit") or {}
    checked = f.get("checked") or {}

    json_inputs = []
    for i in inputs:
        row = {"name": i.get("name"), "field": i.get("field"), "type": i.get("type"),
               "from": i.get("from"), "showApprover": bool(i.get("showApprover")),
               "sensitive": bool(i.get("sensitive"))}
        if i.get("unit"):
            row["unit"] = i["unit"]
        json_inputs.append(row)

    to_agent: Dict[str, Any] = {"onMatch": _ON_MATCH.get(outcome),
                                "reasonCode": f"{policy_key}.{h.get('reasonCode') or 'policy'}",
                                "reason": f.get("messageForAgent"),
                                "messageForPerson": f.get("messageForPerson")}
    if outcome == "approve":
        to_agent["whilePaused"] = h.get("whilePaused") or "no_retry"
        # keep the brief's key order: onMatch, reasonCode, reason, whilePaused, messageForPerson
        to_agent = {k: to_agent[k] for k in ("onMatch", "reasonCode", "reason", "whilePaused", "messageForPerson")}

    enforcement: Dict[str, Any] = {"outcome": outcome}
    to_approver = None
    to_notify = None
    if outcome == "approve":
        deciders = [d for d in (f.get("deciders") or []) if str(d).strip()]
        within, source = response_time(f, settings)
        enforcement["intervention"] = {
            "escalateTo": _roles(deciders),
            "sla": {"source": source, "respondWithin": within,
                    "onTimeout": "escalate_next" if len(deciders) > 1 else "reject"}}
        to_approver = {"show": [i["field"] for i in inputs if i.get("showApprover")],
                       "options": ["approve", "reject"], "reasonRequiredOn": ["reject"]}
    elif outcome in ("block", "notify"):
        told = [n for n in (f.get("notify") or []) if str(n).strip()]
        if told:
            enforcement["notify"] = _roles(told)
            to_notify = {"to": told}

    return {
        "schema": "hard-policy/2",
        "id": policy_key,
        "version": version,
        "status": status,
        "title": f.get("name"),
        "category": f.get("category"),
        "owner": f.get("owner"),
        "businessArea": {"primary": f.get("businessArea"), "subArea": f.get("subArea")},
        "effective": {"from": f.get("effectiveFrom") or None, "reviewBy": f.get("reviewBy") or None},
        "scope": {"agents": ["*"], "tools": ["*"], "skills": ["*"],
                  "limit": (limit.get("text") or None) if limit.get("on") else None},
        "source": f.get("source"),
        "context": {"checkpoint": checkpoint, "actions": h.get("actions"),
                    "timeWindow": h.get("timeWindow"), "units": h.get("units")},
        "inputs": json_inputs,
        "outputs": {"toAgent": to_agent, "toApprover": to_approver, "toNotify": to_notify,
                    "audit": {"logInputs": True,
                              "mask": [i["field"] for i in inputs if i.get("sensitive")]}},
        "trigger": {"plain": f.get("situation"),
                    "events": [checkpoint] if checkpoint else [],
                    "condition": h.get("condition"),
                    "onMissingData": on_missing_for(f, settings),
                    "setBy": h.get("setBy") or "person",
                    "checkedBy": checked.get("by"),
                    "checkedAt": checked.get("at")},
        "enforcement": enforcement,
        "conflicts": list(conflicts or []),
        "learning": {"eligible": outcome == "approve" and not never_suggest},
    }
