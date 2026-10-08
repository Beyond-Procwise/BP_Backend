"""ProposedPolicy (what the extraction agent said) -> stage 1's form state.

The agent's output is a proposal: checked stays None, setBy is "extraction_agent", and every
name it used is re-checked against the registry and the taxonomy here. The code verifies; it
never trusts the model.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

from services.agent_policy.extraction_schema import ProposedPolicy
from services.agent_policy.registry import RegistrySnapshot

# Lookups and totals are not registered in this release, so their inputs are never
# available at a checkpoint and the form shows "Can't be enforced yet".
_FROM = {"action": "action", "lookup": "lookup:unregistered", "total": "total:unregistered"}
_ALWAYS = (
    {"name": "Tool", "field": "tool.name", "type": "string", "isAmount": False, "from": "action",
     "showApprover": False, "sensitive": False},
    {"name": "Agent's reason", "field": "agent.reason", "type": "string", "isAmount": False, "from": "action",
     "showApprover": True, "sensitive": False},
)


def _unique(items: Iterable[str]) -> List[str]:
    seen: List[str] = []
    for i in items:
        if i and i not in seen:
            seen.append(i)
    return seen


def _leaf(rule) -> Dict[str, Any]:
    leaf: Dict[str, Any] = {"field": rule.field, "op": rule.op}
    if rule.op == "exists":
        return leaf
    if rule.op in ("in", "not_in"):
        leaf["value"] = list(rule.value_list)
    elif rule.value_number is not None:
        leaf["value"] = rule.value_number
    else:
        leaf["value"] = rule.value_text
    return leaf


def _area(p: ProposedPolicy, taxonomy: List[Mapping[str, Any]], notes: List[str]):
    areas = {a.get("areaName"): list(a.get("subAreas") or []) for a in taxonomy or []}
    area: Optional[str] = p.business_area
    sub: Optional[str] = p.sub_area
    if area not in areas:
        notes.append(f"The agent proposed business area '{p.business_area}', which is not in the taxonomy.")
        area = None
    if area is None or sub not in areas[area]:
        notes.append(f"The agent proposed sub-area '{p.sub_area}', which is not in the taxonomy.")
        sub = None
    return area, sub


def _input(spec) -> Dict[str, Any]:
    row = {"name": spec.name, "field": spec.field, "type": spec.type, "isAmount": spec.is_amount,
           "from": _FROM[spec.source], "showApprover": spec.show_approver, "sensitive": spec.sensitive}
    if spec.unit:
        row["unit"] = spec.unit
    return row


def _rule_input(fld: str, p: ProposedPolicy, registry: RegistrySnapshot) -> Dict[str, Any]:
    """An input for a condition field the agent forgot to list (the contract needs every one)."""
    row = registry.input_row(p.checkpoint, fld)
    if row:
        return {"name": row.get("plain") or fld, "field": fld, "type": row.get("type") or "string",
                "isAmount": False, "from": row.get("source") or "action",
                "showApprover": True, "sensitive": False}
    numeric = any(r.field == fld and r.value_number is not None for r in p.rules)
    return {"name": fld, "field": fld, "type": "number" if numeric else "string", "isAmount": False,
            "from": "action", "showApprover": True, "sensitive": False}


def to_form(p: ProposedPolicy, *, document_title: str, document_version: Optional[int],
            registry: RegistrySnapshot, taxonomy: List[Mapping[str, Any]]) -> Dict[str, Any]:
    cp = p.checkpoint
    leaves = [_leaf(r) for r in p.rules]
    added_tool_leaf = bool(p.action_tools) and not any(r.field == "tool.name" for r in p.rules)
    if added_tool_leaf:
        leaves.insert(0, {"field": "tool.name", "op": "in", "value": list(p.action_tools)})

    inputs = [_input(i) for i in p.inputs]
    listed = {i["field"] for i in inputs}
    for always in _ALWAYS:
        if always["field"] not in listed:
            inputs.append(dict(always))
            listed.add(always["field"])
    for leaf in leaves:
        if leaf["field"] not in listed:
            inputs.append(_rule_input(leaf["field"], p, registry))
            listed.add(leaf["field"])

    rule_tools = [v for r in p.rules if r.field == "tool.name"
                  for v in (r.value_list or ([r.value_text] if r.value_text else []))]
    unknown = _unique(
        list(p.unknown_names)
        + [r.field for r in p.rules if registry.input_row(cp, r.field) is None]
        + [i.field for i in p.inputs if i.source == "action" and registry.input_row(cp, i.field) is None]
        + [t for t in list(p.action_tools) + rule_tools if not registry.knows_action(cp, t)])

    examples = []
    for ex in p.examples:
        values = {v.field: (v.value_number if v.value_number is not None else v.value_text) for v in ex.values}
        if added_tool_leaf and "tool.name" not in values:
            # The tool leaf was added here, not by the agent: an example that omits the tool
            # would otherwise be judged on a missing field instead of on its own values.
            values = {"tool.name": p.action_tools[0], **values}
        examples.append({"input": values, "agentExpected": ex.expected, "flipped": False})

    notes: List[str] = []
    area, sub = _area(p, taxonomy, notes)
    window = None
    if p.time_window_from or p.time_window_to:
        window = {"from": p.time_window_from, "to": p.time_window_to, "timeZone": p.time_zone}

    hidden = {
        "checkpoint": cp,
        "actions": {"tools": list(p.action_tools), "plain": p.action_plain},
        "timeWindow": window,
        "units": {"currency": p.currency, "convertOther": "rate_on_action_date",
                  "amountsIncludeTax": p.amounts_include_tax},
        "inputs": inputs,
        "missingInputs": [m.model_dump() for m in p.missing_inputs],
        "unknownNames": unknown,
        "condition": {p.match: leaves} if leaves else None,
        "onMissingData": None,
        "reasonCode": p.reason_code,
        "whilePaused": "no_retry",
        "setBy": "extraction_agent",
        "agentNotes": notes,
    }
    return {
        "name": p.name, "category": p.category,
        "businessArea": area, "subArea": sub,
        "situation": p.situation,
        "source": {"document": document_title, "documentVersion": document_version,
                   "reference": p.reference, "excerpt": p.excerpt},
        "outcome": p.outcome,
        "outcomeBecause": p.outcome_phrase,
        "deciders": list(p.deciders),
        "responseTime": None,
        "notify": list(p.notify),
        "limit": {"on": False, "text": ""},
        "owner": p.owner, "effectiveFrom": None, "reviewBy": None,
        "messageForAgent": p.message_for_agent,
        "messageForPerson": p.message_for_person,
        "hidden": hidden,
        "examples": examples,
        "checked": None,
        "changeNote": "",
    }
