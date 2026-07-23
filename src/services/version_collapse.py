"""Collapse quote version rounds to a single current bid.

28 quotes in the analysis batch are really 12 distinct proposals; 20 carry a
``(V<n>...)`` suffix ("CPS-Q-3380 (V3 (BAFO))"). Counting rounds as separate
competing bids inflates supplier counts and corrupts bid-spread maths, so this
is a prerequisite for correct clustering. Pure — no DB.
"""
from __future__ import annotations

import re

# Trailing "(V2)" / "(V3 (BAFO))" — capture the integer after the leading V.
_VERSION_RE = re.compile(r"\s*\(\s*v(\d+).*\)\s*$", re.IGNORECASE)


def base_reference(quote_id: str) -> str:
    """The quote id with a trailing (V<n>...) round suffix removed."""
    return _VERSION_RE.sub("", str(quote_id)).strip()


def version_ordinal(quote_id: str) -> int:
    """Round number: 1 when unversioned, else the integer inside (V<n>...)."""
    m = _VERSION_RE.search(str(quote_id))
    return int(m.group(1)) if m else 1


def collapse_versions(quotes: list[dict]) -> list[dict]:
    """Group quotes by base reference; return one bid per group (highest round).

    The returned bid is the highest-version member's dict, plus:
      base_reference — the shared base id
      version        — its round ordinal
      rounds         — every member quote_id, sorted
    """
    groups: dict[str, list[dict]] = {}
    for q in quotes:
        groups.setdefault(base_reference(q["quote_id"]), []).append(q)

    bids: list[dict] = []
    for base, members in groups.items():
        current = max(members, key=lambda q: version_ordinal(q["quote_id"]))
        bids.append({
            **current,
            "base_reference": base,
            "version": version_ordinal(current["quote_id"]),
            "rounds": sorted(m["quote_id"] for m in members),
        })
    return bids
