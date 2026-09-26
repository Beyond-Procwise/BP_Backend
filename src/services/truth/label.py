"""One extraction record in, one verdict per field out.

The glue between ``verify`` (which knows about text) and ``build_set`` (which
knows about corpora), so neither has to know the other's shape.
"""
from __future__ import annotations

from typing import Any, Optional

from src.services.truth.derived import is_derived
from src.services.truth.verify import UNVERIFIABLE, Verdict, verify_field


def _judge(field: str, value: Any, source_text: Optional[str]) -> Verdict:
    if is_derived(field):
        return Verdict(UNVERIFIABLE, "derived-field")
    return verify_field(field, value, source_text)


def label_record(extracted: dict, source_text: Optional[str]) -> dict:
    """Walk header and line items, returning a verdict and rule for every field.

    A line-item field is keyed ``line_items[<i>].<field>`` so a verdict can be
    traced back to the row it came from.
    """
    fields: dict[str, dict] = {}

    for field, value in (extracted.get("header") or {}).items():
        v = _judge(field, value, source_text)
        fields[field] = {"value": value, "outcome": v.outcome, "rule": v.rule}

    for i, line in enumerate(extracted.get("line_items") or []):
        if not isinstance(line, dict):
            continue
        for field, value in line.items():
            v = _judge(field, value, source_text)
            fields[f"line_items[{i}].{field}"] = {
                "value": value, "outcome": v.outcome, "rule": v.rule}

    counts = {"verified": 0, "unsupported": 0, "unverifiable": 0}
    for f in fields.values():
        counts[f["outcome"]] = counts.get(f["outcome"], 0) + 1
    return {"fields": fields, "counts": counts}
