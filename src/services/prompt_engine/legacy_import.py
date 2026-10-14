"""Bring the prompts already in ``proc.bp_prompt`` into the registry.

Each active row becomes version 1 of a prompt whose key is the row's name,
kept exactly as it was (``syntax = 'legacy'``), with the live pointer on it.
Nothing about how the product behaves changes: the older table is not written
to, and the registry simply now knows what is in use.

Safe to run again. A prompt already in the registry is left alone, and one
whose older row has since drifted from the registry's live version is reported,
never overwritten.

Refused, and reported: a name that is not a usable key, and a name two active
rows share (the key must mean one prompt).
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Dict, List, Tuple

from services.db import get_conn
from services.prompt_engine import registry as R

IMPORTER = "import"


def _loaded(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


def _template_of(desc: Any) -> str:
    if isinstance(desc, str):
        return desc
    if isinstance(desc, dict):
        text = desc.get("prompt_template") or desc.get("template")
        return text if isinstance(text, str) else ""
    return ""


def read_rows() -> List[Dict[str, Any]]:
    """The active rows of the older table, in definition order."""
    columns = ("prompt_id", "prompt_name", "prompt_type", "prompt_linked_agents",
               "prompts_desc", "version", "created_by", "last_modified_by")
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            f"SELECT {', '.join(columns)} FROM proc.bp_prompt "
            "WHERE COALESCE(prompts_status, 1) = 1 ORDER BY prompt_id")
        rows = cur.fetchall()
        cur.close()
    return [dict(zip(columns, r)) for r in rows]


def plan(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """``(what would be imported, why anything was refused)``. Reads nothing."""
    names = Counter(str(r.get("prompt_name") or "") for r in rows)
    items: List[Dict[str, Any]] = []
    problems: List[str] = []
    for row in rows:
        name = str(row.get("prompt_name") or "")
        ident = f"prompt_id {row.get('prompt_id')} ({name!r})"
        try:
            R._check_key(name)
        except R.InvalidKey:
            problems.append(f"{ident}: the name is not a usable key")
            continue
        if names[name] > 1:
            problems.append(f"{ident}: {names[name]} active rows share this name")
            continue
        desc = _loaded(row.get("prompts_desc"))
        items.append({
            "key": name,
            "template": _template_of(desc),
            "author": (row.get("last_modified_by") or row.get("created_by") or IMPORTER),
            "settings": {
                R.LEGACY_DESC: desc,
                "legacy_type": row.get("prompt_type"),
                "legacy_linked_agents": row.get("prompt_linked_agents"),
            },
            "change_note": (f"imported from proc.bp_prompt as it stood "
                            f"(its version {row.get('version')})"),
        })
    return items, problems


def run(apply: bool = False) -> Dict[str, Any]:
    """Import, or with ``apply=False`` only say what an import would do."""
    items, problems = plan(read_rows())
    report: Dict[str, Any] = {"imported": [], "already_there": [], "drifted": [],
                              "refused": problems, "applied": apply}
    for item in items:
        key = item["key"]
        if R._kind_of(key) is not None:
            try:
                live = R.resolve(key)
            except R.PointerUnresolved:
                report["already_there"].append(key)
                continue
            same = (live["settings"].get(R.LEGACY_DESC, {"prompt_template": live["template"]})
                    == item["settings"][R.LEGACY_DESC])
            report["already_there" if same else "drifted"].append(key)
            continue
        if apply:
            R.adopt(key, template=item["template"], by=item["author"], change="imported",
                    settings=item["settings"], change_note=item["change_note"],
                    note="in use at the time of import")
        report["imported"].append(key)
    return report
