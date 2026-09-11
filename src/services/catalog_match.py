"""Which catalog SKU is which item in purchase history (spec §4.6).

A match is a claim about two datasets, so it is proposed and a person decides.
Exact methods first -- they carry no confidence, because a number there would
imply a judgement nobody made -- then one best fuzzy candidate per SKU that no
exact method resolved.

History carries one identifier, item_id, and no MPN column exists on any _trgt
line table. mpn_exact and sku_exact both compare against item_id for that reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence

from rapidfuzz import fuzz, process

from src.services.governed_limits import limit as _governed_limit
from src.services.sell_side._db import NotFound, StateConflict, dict_cursor


def _FUZZY_MIN() -> float:
    return _governed_limit("reseller_catalog", "fuzzy_propose_min")


@dataclass(frozen=True)
class Proposal:
    distributor_sku: str
    item_id: str
    match_method: str
    confidence: Optional[Decimal]


def _key(value: Optional[str]) -> str:
    return " ".join((value or "").split()).casefold()


def propose(catalog: Sequence[Dict[str, Any]], history: Sequence[Dict[str, Any]],
            *, fuzzy_min: float) -> List[Proposal]:
    by_id = {_key(h["item_id"]): h["item_id"] for h in history if h.get("item_id")}
    out: List[Proposal] = []
    unresolved: List[Dict[str, Any]] = []
    for item in catalog:
        sku = item["distributor_sku"]
        found = set()
        mpn = _key(item.get("mpn"))
        if mpn and mpn in by_id:
            out.append(Proposal(sku, by_id[mpn], "mpn_exact", None))
            found.add(by_id[mpn])
        sku_hit = by_id.get(_key(sku))
        if sku_hit and sku_hit not in found:
            out.append(Proposal(sku, sku_hit, "sku_exact", None))
            found.add(sku_hit)
        if not found:
            unresolved.append(item)

    if unresolved and history:
        choices = [_key(h["item_description"]) for h in history]
        for item in unresolved:
            best = process.extractOne(_key(item["item_description"]), choices,
                                      scorer=fuzz.token_sort_ratio, score_cutoff=fuzzy_min)
            if best:
                _, score, idx = best
                out.append(Proposal(item["distributor_sku"], history[idx]["item_id"],
                                    "description_fuzzy",
                                    Decimal(str(round(score / 100.0, 4)))))
    return out


_HISTORY_SQL = """
SELECT item_id, MIN(item_description) AS item_description FROM (
    SELECT item_id, item_description FROM proc.bp_invoice_line_items_trgt WHERE item_id IS NOT NULL
    UNION ALL
    SELECT item_id, item_description FROM proc.bp_po_line_items_trgt WHERE item_id IS NOT NULL
) h GROUP BY item_id
"""


def propose_matches(conn: Any, distributor_id: str) -> Dict[str, int]:
    """Propose matches for this distributor's current catalog. Re-running is safe:
    a pair already proposed, confirmed or rejected is never re-proposed."""
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT distributor_sku, mpn, item_description FROM proc.bp_catalog_item "
        "WHERE distributor_id = %s AND valid_to IS NULL", (distributor_id,))
    catalog = cur.fetchall() or []
    cur.execute(_HISTORY_SQL)
    history = cur.fetchall() or []
    proposals = propose(catalog, history, fuzzy_min=_FUZZY_MIN())

    counts = {"proposed": 0, "exact": 0, "fuzzy": 0, "already_known": 0}
    for p in proposals:
        cur.execute(
            "INSERT INTO proc.bp_catalog_item_match (distributor_id, distributor_sku, item_id, "
            "match_method, confidence) VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (distributor_id, distributor_sku, item_id) DO NOTHING",
            (distributor_id, p.distributor_sku, p.item_id, p.match_method, p.confidence))
        if cur.rowcount:
            counts["proposed"] += 1
            counts["fuzzy" if p.match_method == "description_fuzzy" else "exact"] += 1
        else:
            counts["already_known"] += 1
    conn.commit()
    return counts


def list_matches(conn: Any, *, distributor_id: Optional[str] = None,
                 status: str = "proposed", limit: int = 100) -> List[Dict[str, Any]]:
    cur = dict_cursor(conn)
    cur.execute(
        "SELECT m.*, c.item_description AS catalog_description "
        "FROM proc.bp_catalog_item_match m "
        "LEFT JOIN proc.bp_catalog_item c ON c.distributor_id = m.distributor_id "
        " AND c.distributor_sku = m.distributor_sku AND c.valid_to IS NULL "
        "WHERE (%s IS NULL OR m.distributor_id = %s) AND (%s = 'all' OR m.status = %s) "
        "ORDER BY m.confidence DESC NULLS FIRST, m.match_id LIMIT %s",
        (distributor_id, distributor_id, status, status, max(1, min(limit, 500))))
    return [dict(r) for r in (cur.fetchall() or [])]


def _decide(conn: Any, match_id: int, reviewer: Optional[str], status: str) -> Dict[str, Any]:
    """confirmed_by / confirmed_at record whoever DECIDED -- a rejection too."""
    cur = dict_cursor(conn)
    cur.execute("SELECT status FROM proc.bp_catalog_item_match WHERE match_id = %s FOR UPDATE",
                (match_id,))
    row = cur.fetchone()
    if row is None:
        conn.rollback()
        raise NotFound(f"match {match_id} does not exist")
    if row["status"] != "proposed":
        conn.rollback()
        raise StateConflict(f"match {match_id} is already {row['status']}")
    cur.execute(
        "UPDATE proc.bp_catalog_item_match SET status = %s, confirmed_by = %s, "
        "confirmed_at = now() WHERE match_id = %s RETURNING *",
        (status, reviewer, match_id))
    out = dict(cur.fetchone())
    conn.commit()
    return out


def confirm_match(conn: Any, match_id: int, reviewer: Optional[str]) -> Dict[str, Any]:
    return _decide(conn, match_id, reviewer, "confirmed")


def reject_match(conn: Any, match_id: int, reviewer: Optional[str]) -> Dict[str, Any]:
    return _decide(conn, match_id, reviewer, "rejected")


def record_human_match(conn: Any, *, distributor_id: str, distributor_sku: str,
                       item_id: str, reviewer: Optional[str]) -> Dict[str, Any]:
    """A person asserts a match no method found. Confirmed on write; replaces any
    earlier machine proposal or rejection for the same pair."""
    cur = dict_cursor(conn)
    cur.execute("SELECT 1 FROM proc.bp_catalog_item WHERE distributor_id = %s "
                "AND distributor_sku = %s AND valid_to IS NULL", (distributor_id, distributor_sku))
    if cur.fetchone() is None:
        conn.rollback()
        raise NotFound(f"{distributor_id}/{distributor_sku} is not a current catalog SKU")
    cur.execute(
        "INSERT INTO proc.bp_catalog_item_match (distributor_id, distributor_sku, item_id, "
        "match_method, confidence, status, confirmed_by, confirmed_at) "
        "VALUES (%s, %s, %s, 'human', NULL, 'confirmed', %s, now()) "
        "ON CONFLICT (distributor_id, distributor_sku, item_id) DO UPDATE SET "
        "match_method = 'human', confidence = NULL, status = 'confirmed', "
        "confirmed_by = EXCLUDED.confirmed_by, confirmed_at = now() RETURNING *",
        (distributor_id, distributor_sku, item_id, reviewer))
    out = dict(cur.fetchone())
    conn.commit()
    return out
