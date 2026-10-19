"""The wording of a sales opportunity's reason, read from the governed prompt library.

The detector decides WHAT is true (a relation the distributor stated, an end-of-life date); the
sentence that says so lives in ``proc.bp_prompt`` where a person can reword it without a deploy.
A prompt row holds ``prompts_desc = {"prompt_template": "...{field}...", "phrases": {field: {value: words}}}``.
``phrases`` turns a coded fact (``replaced_by``) into words; a ``default`` entry covers the values
nobody listed. Nothing is invented here: a missing prompt, a missing field or an unlisted value with
no default raises ``ReasonPromptError``, and the caller treats that as "no evidence, no opportunity".
"""
from __future__ import annotations

import string
from typing import Any, Dict, Optional, Tuple

from src.services.sell_side._db import dict_cursor

RELATION = "sales_reason_relation"
END_OF_LIFE_DATED = "sales_reason_end_of_life_dated"
END_OF_LIFE_UNDATED = "sales_reason_end_of_life_undated"


class ReasonPromptError(RuntimeError):
    """The wording for a reason cannot be produced from the prompt library."""


def _load(conn: Any, name: str) -> Tuple[str, Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT prompts_desc FROM proc.bp_prompt "
        "WHERE prompt_name = %s AND COALESCE(prompts_status, 1) = 1 "
        "ORDER BY version DESC NULLS LAST, prompt_id DESC LIMIT 1", (name,))
    row = cur.fetchone()
    desc = row["prompts_desc"] if row else None
    template = desc.get("prompt_template") if isinstance(desc, dict) else None
    if not template:
        raise ReasonPromptError(f"no active prompt named {name!r} with a prompt_template")
    phrases = desc.get("phrases") or {}
    return template, phrases


def render(conn: Any, name: str, facts: Dict[str, Any]) -> str:
    """The reason sentence for ``facts`` under the prompt ``name``."""
    template, phrases = _load(conn, name)
    values: Dict[str, Any] = dict(facts)
    for field, words in phrases.items():
        if field not in facts:
            continue
        key = str(facts[field])
        if key in words:
            values[field] = words[key]
        elif "default" in words:
            values[field] = words["default"]
        else:
            raise ReasonPromptError(f"prompt {name!r} has no wording for {field}={key!r}")
    needed = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    missing = needed - values.keys()
    if missing:
        raise ReasonPromptError(f"prompt {name!r} needs {sorted(missing)} which the detector did not supply")
    return template.format_map(values)
