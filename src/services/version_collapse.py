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


# --- Shared by every analysis over quotes (2026-10-08) ----------------------------------------
# A bid is a supplier's quote FAMILY -- one base reference, from one supplier -- and its current
# value is its latest version; earlier versions are history. Analyses that summed, averaged,
# counted or ranked every version as a separate bid gave a three-supplier, three-round deal nine
# "bids" and a value of all nine added together. Twin of the gateway's spendiq.quote-version.ts
# and of bp_deal_overview's superseded_quote.

def QUOTE_VERSION_SQL(col: str) -> str:
    """SQL: the version number of a quote id column (unversioned = 1)."""
    return f"COALESCE((regexp_match({col}, '\\(\\s*V(\\d+)', 'i'))[1]::int, 1)"


def QUOTE_BASE_SQL(col: str) -> str:
    """SQL: the quote id with its version marker removed."""
    return f"regexp_replace({col}, '\\s*\\(\\s*v(\\d+).*\\)\\s*$', '', 'i')"


def latest_quote_pred(alias: str, table: str = "proc.bp_quote_trgt") -> str:
    """SQL predicate: row `alias` is its quote family's latest version.

    Correlated over `table` (the same table the alias reads), so it works inside any query
    without restructuring it: `... WHERE q.deal_id = %s AND <pred>`."""
    return (
        f"NOT EXISTS (SELECT 1 FROM {table} _lv "
        f"WHERE COALESCE(_lv.supplier_id, '') = COALESCE({alias}.supplier_id, '') "
        f"AND {QUOTE_BASE_SQL('_lv.quote_id')} = {QUOTE_BASE_SQL(f'{alias}.quote_id')} "
        f"AND ({QUOTE_VERSION_SQL('_lv.quote_id')}, _lv.quote_id) > "
        f"({QUOTE_VERSION_SQL(f'{alias}.quote_id')}, {alias}.quote_id))"
    )


def latest_per_family(rows: list[dict], id_key: str = "quote_id",
                      supplier_key: str = "supplier_id") -> list[dict]:
    """One row per bid -- each (supplier, base reference) family's highest version -- in
    first-seen order. A row with no supplier is still a bid; two suppliers who number a quote
    the same are two bids. Pure."""
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    for r in rows:
        qid = str(r.get(id_key) or "")
        key = (r.get(supplier_key) or "", base_reference(qid))
        if key not in best:
            order.append(key)
            best[key] = r
        elif (version_ordinal(qid), qid) > (version_ordinal(str(best[key].get(id_key) or "")),
                                            str(best[key].get(id_key) or "")):
            best[key] = r
    return [best[k] for k in order]
