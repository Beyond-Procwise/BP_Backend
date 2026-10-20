"""Comparing landed costs: like with like.

A component enters a comparison only if it has a value for EVERY option being compared.
Otherwise an option that states less looks cheaper because it said less, not because it
costs less. A cost counted for one option and left out for the others is shown against that
option, never silently added or dropped.

``included`` (the Incoterm or quote puts it in the goods price) is a value of 0, so an option
that includes freight is compared with one that states it. ``missing`` and ``not_stated`` are
not values.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

from .landed_cost import COUNTED, MISSING


def compare_landed(options: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare options given as ``{"name": ..., "landed": <landed_cost result or None>}``."""
    unassessed = [str(o.get("name")) for o in options if o.get("landed") is None]
    priced = [o for o in options if o.get("landed") is not None]
    if len(options) < 2:
        raise ValueError("a comparison needs at least two options")
    currencies = {o["landed"]["currency"] for o in priced}
    if len(currencies) > 1:
        raise ValueError(f"options are in different currencies ({', '.join(sorted(currencies))}): "
                         "convert them to one currency first")

    names: List[str] = []
    for o in priced:
        for c in o["landed"]["components"]:
            if c["name"] not in names:
                names.append(c["name"])

    def comp(option: Mapping[str, Any], name: str) -> Mapping[str, Any]:
        return next(c for c in option["landed"]["components"] if c["name"] == name)

    comparable = [n for n in names if priced and all(comp(o, n)["status"] in COUNTED
                                                     for o in priced)]
    missing_for_all = [n for n in names if priced and all(comp(o, n)["status"] == MISSING
                                                          for o in priced)]

    rows: List[Dict[str, Any]] = []
    excluded_any = False
    for o in priced:
        excluded = [{"name": n, "amount": comp(o, n)["amount"],
                     "why": "another option has no value for it"}
                    for n in names if n not in comparable and comp(o, n)["status"] in COUNTED]
        excluded_any = excluded_any or bool(excluded)
        rows.append({
            "name": o.get("name"),
            "comparable_total": sum(comp(o, n)["amount"] for n in comparable),
            "own_landed_total": o["landed"]["landed_total"],
            "landed_status": o["landed"]["landed_status"],
            "excluded": excluded,
        })

    lowest: List[str] = []
    if rows:
        best = min(r["comparable_total"] for r in rows)
        lowest = [r["name"] for r in rows if abs(r["comparable_total"] - best) < 1e-9]

    return {
        "basis": "components that have a value for every option",
        "currency": next(iter(currencies)) if currencies else None,
        "comparable_components": comparable,
        "options": rows,
        "lowest": lowest,
        # Partial: a cost was set aside for some option, or one that is owed is missing from all.
        "partial": excluded_any or bool(missing_for_all),
        "missing_for_all": missing_for_all,
        "unassessed": unassessed,
    }
