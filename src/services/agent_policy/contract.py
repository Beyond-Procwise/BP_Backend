"""Is a hard-policy/2 document one the orchestrator can enforce without guessing?

JSON Schema covers shape. The cross-field rules the brief adds (§7 "Contract") cannot be
expressed in JSON Schema, so they are checked here, in the same call: one function, one
answer, used by activation AND by the orchestrator feed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft202012Validator

from services.agent_policy import conditions
from services.agent_policy.registry import RegistrySnapshot

_SCHEMA = json.loads((Path(__file__).with_name("hard-policy-2.schema.json")).read_text())
# The schema says "format": "date"; without a format checker that is only a comment.
_VALIDATOR = Draft202012Validator(_SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER)


def validate(doc: Dict[str, Any], registry: RegistrySnapshot) -> List[str]:
    problems = [f"{'/'.join(str(p) for p in e.path) or 'document'}: {e.message}"
                for e in _VALIDATOR.iter_errors(doc)]
    trigger = doc.get("trigger") or {}
    context = doc.get("context") or {}
    cp = context.get("checkpoint")
    input_fields = {i.get("field") for i in doc.get("inputs") or []}

    if trigger.get("events") != ([cp] if cp else []):
        problems.append("trigger.events must equal [context.checkpoint]")
    if not registry.knows_checkpoint(cp):
        problems.append(f"context.checkpoint {cp!r} is not a checkpoint the orchestrator recognises")

    cond = trigger.get("condition")
    try:
        conditions.to_engine(cond)
    except conditions.ConditionError as exc:
        problems.append(f"trigger.condition cannot be read: {exc}")
    for fld in sorted(conditions.condition_fields(cond) - input_fields):
        problems.append(f"condition field {fld} is not listed in inputs")
    for tool in sorted(conditions.tool_names(cond)):
        if not registry.knows_action(cp, tool):
            problems.append(f"tool {tool} is not something the orchestrator recognises")
    for tool in (context.get("actions") or {}).get("tools") or []:
        if not registry.knows_action(cp, tool):
            problems.append(f"tool {tool} is not something the orchestrator recognises")

    show = ((doc.get("outputs") or {}).get("toApprover") or {}).get("show") or []
    for fld in show:
        if fld not in input_fields:
            problems.append(f"approver field {fld} is not listed in inputs")
    for i in doc.get("inputs") or []:
        if not registry.available(cp, i.get("field")):
            problems.append(f"input {i.get('field')} is not available at {cp}")

    if doc.get("status") == "live" and not trigger.get("checkedBy"):
        problems.append("a live policy needs trigger.checkedBy")
    return sorted(set(problems))
