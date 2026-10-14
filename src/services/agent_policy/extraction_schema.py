"""What the extraction agent returns, and the grammar it is held to.

Flat and union-free on purpose. Ollama's JSON-schema converter does not honour ``oneOf`` /
``anyOf`` (measured in services/rga/compose.py: a union collapses to its loosest member), so
the schema sent as ``format=`` comes from grammar_schema(), which rewrites every
``Optional[X]`` (pydantic emits ``anyOf: [X, null]``) to plain ``X`` and leaves the field
optional. Validation on the way back still uses these models, so ``null`` stays legal.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict

OUTCOMES = Literal["approve", "block", "notify"]
OPS = Literal["gt", "gte", "lt", "lte", "eq", "ne", "in", "not_in", "exists"]
RESULTS = Literal["approve", "block", "notify", "none"]
CHECKPOINTS = Literal["tool.call.before", "message.send.before", "data.egress.before", "record.write.before"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Rule(_Strict):
    field: str
    op: OPS
    value_number: Optional[float] = None
    value_text: Optional[str] = None
    value_list: List[str] = []


class ExampleValue(_Strict):
    field: str
    value_number: Optional[float] = None
    value_text: Optional[str] = None


class Example(_Strict):
    values: List[ExampleValue]
    expected: RESULTS


class InputSpec(_Strict):
    name: str
    field: str
    type: Literal["string", "number", "boolean", "date", "list"]
    is_amount: bool = False
    unit: Optional[str] = None
    source: Literal["action", "lookup", "total"] = "action"
    show_approver: bool = True
    sensitive: bool = False


class Missing(_Strict):
    name: str
    reason: str


class ProposedPolicy(_Strict):
    name: str
    category: str
    business_area: str
    sub_area: str
    situation: str
    match: Literal["all", "any"]
    rules: List[Rule]
    outcome: OUTCOMES
    outcome_phrase: str
    deciders: List[str] = []
    notify: List[str] = []
    reference: str
    excerpt: str
    examples: List[Example]
    checkpoint: CHECKPOINTS
    action_tools: List[str]
    action_plain: str
    time_window_from: Optional[str] = None
    time_window_to: Optional[str] = None
    time_zone: Optional[str] = None
    currency: Optional[str] = None
    amounts_include_tax: Optional[bool] = None
    inputs: List[InputSpec]
    missing_inputs: List[Missing] = []
    unknown_names: List[str] = []
    reason_code: str
    message_for_agent: str
    message_for_person: Optional[str] = None
    owner: Optional[str] = None


class NotEnforceable(_Strict):
    reference: str
    excerpt: str
    reason: str


class ChunkResult(_Strict):
    policies: List[ProposedPolicy]
    not_enforceable: List[NotEnforceable]


def _flatten(node: Any) -> Any:
    if isinstance(node, list):
        return [_flatten(n) for n in node]
    if not isinstance(node, dict):
        return node
    out = {k: _flatten(v) for k, v in node.items()}
    union = out.get("anyOf") or out.get("oneOf")
    if union is not None:
        kept = [m for m in union if m != {"type": "null"}]
        if len(kept) != 1:
            # A real union (two non-null members) cannot be expressed to the grammar.
            raise ValueError(f"schema has a union the grammar cannot express: {union}")
        out.pop("anyOf", None)
        out.pop("oneOf", None)
        if out.get("default", "") is None:
            out.pop("default")
        out = {**kept[0], **out}
    return out


def grammar_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """The model's JSON schema with every nullable union reduced to its one real type."""
    return _flatten(copy.deepcopy(model.model_json_schema()))
