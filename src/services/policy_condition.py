"""Decide whether a policy's ``condition`` holds for one request.

A policy row names the actions it covers in ``details.applies_to``. That says
*which* action, not *when*: "refunds over $500" had to be written as "refunds".
``details.condition`` closes that. It is data, not code, so a policy row can
carry it and nothing in a row can execute.

Grammar (a condition is exactly one of these)::

    {"field": "amount", "op": ">", "value": 500}
    {"field": "note", "op": "exists"}
    {"all": [<condition>, ...]}       every branch holds
    {"any": [<condition>, ...]}       at least one branch holds
    {"not": <condition>}

``field`` is a dotted path into the request context (``order.age_days``).
Operators: ``== != > >= < <= in not_in exists``.

Three outcomes, and the third is the point of this module:

* holds        -> ``True``
* does not     -> ``False``
* cannot tell  -> :class:`MissingField` (the request did not carry a field the
  condition needs) or :class:`ConditionError` (the condition is not well formed
  or compares things that cannot be compared).

"Cannot tell" must never collapse into ``False``. A deny whose condition reads
a field the caller forgot to send would otherwise stop denying, which is how a
missing argument becomes a permission. Three-valued logic is used for
``all``/``any``/``not`` so an unknown branch only matters when the known
branches do not already settle the answer.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Optional

_COMPARE = {">", ">=", "<", "<="}
_EQUALITY = {"==", "!="}
_MEMBER = {"in", "not_in"}
_OPS = _COMPARE | _EQUALITY | _MEMBER | {"exists"}
_COMBINATORS = ("all", "any", "not")

_MISSING = object()


class ConditionError(ValueError):
    """The condition is malformed, or compares values that cannot be compared."""


class MissingField(LookupError):
    """The request context lacks a field the condition needs."""

    def __init__(self, field: str) -> None:
        super().__init__(f"context has no field {field!r}")
        self.field = field


def _number(value: Any, where: str) -> Decimal:
    # bool is an int subclass; `True > 500` is False and `True < 500` is True,
    # neither of which anyone meant. Exact Decimal avoids float drift at the
    # boundary ($500.00 vs 500.0000001).
    if isinstance(value, bool):
        raise ConditionError(f"{where}: a boolean is not a number")
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        raise ConditionError(f"{where}: {value!r} is not a number") from None


def _lookup(context: Any, path: str) -> Any:
    current = context
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return _MISSING if current is None else current


def validate(condition: Any) -> None:
    """Raise :class:`ConditionError` unless ``condition`` is well formed."""

    if not isinstance(condition, dict) or not condition:
        raise ConditionError("a condition must be a non-empty object")

    combinators = [k for k in _COMBINATORS if k in condition]
    if combinators:
        if len(condition) != 1:
            raise ConditionError(
                f"a combinator stands alone; got keys {sorted(condition)}"
            )
        kind = combinators[0]
        if kind == "not":
            validate(condition["not"])
            return
        branches = condition[kind]
        if not isinstance(branches, list) or not branches:
            raise ConditionError(f"{kind!r} needs a non-empty list of conditions")
        for branch in branches:
            validate(branch)
        return

    extra = set(condition) - {"field", "op", "value"}
    if extra:
        raise ConditionError(f"unknown keys {sorted(extra)}")
    field = condition.get("field")
    op = condition.get("op")
    if not isinstance(field, str) or not field.strip():
        raise ConditionError("a comparison needs a 'field'")
    if op not in _OPS:
        raise ConditionError(f"unknown operator {op!r}")
    if op == "exists":
        if "value" in condition:
            raise ConditionError("'exists' takes no value")
        return
    if "value" not in condition:
        raise ConditionError(f"operator {op!r} needs a 'value'")
    value = condition["value"]
    if op in _MEMBER and (not isinstance(value, list) or not value):
        raise ConditionError(f"operator {op!r} needs a non-empty list")
    if op in _COMPARE:
        _number(value, f"value of {field!r}")


def _eval(condition: Dict[str, Any], context: Any) -> Optional[bool]:
    """True / False, or None for 'cannot tell yet' (a missing field)."""

    if "all" in condition:
        unknown = False
        for branch in condition["all"]:
            result = _eval(branch, context)
            if result is False:
                return False
            unknown = unknown or result is None
        return None if unknown else True
    if "any" in condition:
        unknown = False
        for branch in condition["any"]:
            result = _eval(branch, context)
            if result is True:
                return True
            unknown = unknown or result is None
        return None if unknown else False
    if "not" in condition:
        result = _eval(condition["not"], context)
        return None if result is None else not result

    field, op = condition["field"], condition["op"]
    actual = _lookup(context, field)
    if op == "exists":
        return actual is not _MISSING
    if actual is _MISSING:
        return None

    expected = condition["value"]
    if op in _MEMBER:
        hit = actual in expected
        return hit if op == "in" else not hit
    if op in _EQUALITY:
        if isinstance(expected, (int, float, Decimal)) and not isinstance(expected, bool):
            equal = _number(actual, field) == _number(expected, field)
        else:
            equal = actual == expected
        return equal if op == "==" else not equal

    left, right = _number(actual, field), _number(expected, field)
    return {">": left > right, ">=": left >= right, "<": left < right, "<=": left <= right}[op]


def _first_missing(condition: Dict[str, Any], context: Any) -> str:
    for key in ("all", "any"):
        for branch in condition.get(key, []):
            if _eval(branch, context) is None:
                return _first_missing(branch, context)
    if "not" in condition:
        return _first_missing(condition["not"], context)
    return str(condition.get("field"))


def evaluate(condition: Any, context: Any) -> bool:
    """Whether ``condition`` holds for ``context``.

    Raises :class:`MissingField` or :class:`ConditionError` rather than
    returning ``False`` when the answer cannot be established.
    """

    validate(condition)
    result = _eval(condition, context)
    if result is None:
        raise MissingField(_first_missing(condition, context))
    return result
