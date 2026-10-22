"""classify_spend: an item against the organisation's own category taxonomy and product mapping. Read-only.

Two sources, both the organisation's: the category master (``proc.bp_category_master``, five levels with
UNSPSC codes, owners and status) and the product-to-category mapping (``proc.bp_category_product_map``).
An item whose description is mapped is "recorded" with its five levels. An item that is not mapped is
"unclassified": the tool offers up to three taxonomy rows whose level names share words with the
description, labelled as name matches, which are a draft for a person to confirm, never a classification.
Coverage is the recorded share, so a skill can state it against its threshold. Nothing is written.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services.contract_tools import store

log = logging.getLogger(__name__)

MAX_ITEMS = 200
MAX_CANDIDATES = 3
MAX_BROWSE = 40
LEVELS = ("category_level_1", "category_level_2", "category_level_3", "category_level_4", "category_level_5")

_MAP_SQL = """
    SELECT lower(product), category_level_1, category_level_2, category_level_3, category_level_4, category_level_5
      FROM proc.bp_category_product_map
     WHERE product IS NOT NULL
"""
_TAXONOMY_SQL = """
    SELECT category_level_1_id, category_level_1, category_level_2_id, category_level_2, category_level_3_id,
           category_level_3, category_level_4_id, category_level_4, category_level_5_id, category_level_5,
           unspsc_code, category_status, category_owner_name, esg_impact
      FROM proc.bp_category_master
"""
_WORD = re.compile(r"[a-z0-9]+")
# Words that describe how something is bought, not what it is: they must not match a category on their own.
_STOP = frozenset({"advanced", "essential", "standard", "certified", "modular", "enterprise", "bundle", "unit",
                   "installation", "subscription", "services", "service", "supplies", "general", "other", "misc",
                   "heavy", "duty", "basic", "premium", "professional", "pack", "per", "with", "and", "the"})


def _words(text: Any) -> set:
    return {w for w in _WORD.findall(str(text or "").lower()) if len(w) >= 4 and w not in _STOP}


def _row(r: tuple) -> Dict[str, Any]:
    return {"category_level_1_id": r[0], "category_level_1": r[1], "category_level_2_id": r[2], "category_level_2": r[3],
            "category_level_3_id": r[4], "category_level_3": r[5], "category_level_4_id": r[6], "category_level_4": r[7],
            "category_level_5_id": r[8], "category_level_5": r[9], "unspsc_code": r[10], "status": r[11], "owner": r[12],
            "esg_impact": r[13] if len(r) > 13 else None}


def _candidates(description: str, taxonomy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Taxonomy rows whose level-3 to level-5 names share words with the description, best first.
    A name match says nothing about what was bought: these are offered, never applied."""
    want = _words(description)
    if not want:
        return []
    scored = []
    for cat in taxonomy:
        names = " ".join(str(cat.get(k) or "") for k in LEVELS[2:])
        hit = want & _words(names)
        if hit:
            scored.append((len(hit), cat, sorted(hit)))
    scored.sort(key=lambda s: (-s[0], s[1]["category_level_5_id"] or ""))
    return [{**cat, "matched_words": hit, "basis": "name match only: a draft for a person to confirm, not a classification"}
            for _n, cat, hit in scored[:MAX_CANDIDATES]]


def classify(items: Any = None, query: Any = None, *,
             rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None) -> Dict[str, Any]:
    read = rows or store.rows
    descriptions = [str(i).strip() for i in (items or []) if str(i or "").strip()] if isinstance(items, (list, tuple)) else (
        [str(items).strip()] if isinstance(items, str) and items.strip() else [])
    words = str(query or "").strip()
    if not descriptions and not words:
        return {"found": False, "note": "Give items (a list of item descriptions to classify) or query (words to look up in "
                                        "the category taxonomy)."}
    if len(descriptions) > MAX_ITEMS:
        return {"found": False, "note": f"At most {MAX_ITEMS} items per call."}
    try:
        taxonomy = [_row(r) for r in read(_TAXONOMY_SQL, [])]
        mapping = {r[0]: dict(zip(LEVELS, r[1:6])) for r in read(_MAP_SQL, [])} if descriptions else {}
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The category taxonomy could not be read just now. This does not mean the items are unclassified."}
    if not taxonomy:
        return {"found": False, "loaded": False,
                "note": "No category taxonomy is loaded, so nothing can be classified; use the user's labels and mark "
                        "them draft classification."}
    out: Dict[str, Any] = {"found": True, "taxonomy_rows": len(taxonomy), "recorded_mappings": len(mapping), "written": False}
    if words:
        want = _words(words) or {words.lower()}
        hits = [c for c in taxonomy if want & _words(" ".join(str(c.get(k) or "") for k in LEVELS))
                or words.lower() in " ".join(str(c.get(k) or "") for k in LEVELS).lower()]
        out["categories"] = hits[:MAX_BROWSE]
        out["category_count"] = len(hits)
        out["note"] = ("Taxonomy rows whose names contain the words, as the organisation recorded them (status and owner "
                       "as stored). Not a classification of any purchase.")
        if not descriptions:
            return out
    results = []
    recorded = 0
    for d in descriptions:
        hit = mapping.get(d.lower())
        if hit:
            recorded += 1
            results.append({"item": d, "status": "recorded", "classification": hit,
                            "source": "proc.bp_category_product_map"})
        else:
            results.append({"item": d, "status": "unclassified", "classification": None,
                            "candidates": _candidates(d, taxonomy)})
    out["items"] = results
    out["classified"] = recorded
    out["coverage_pct"] = round(100.0 * recorded / len(descriptions), 1)
    out["note"] = ("'recorded' items carry the organisation's own product-to-category mapping. 'unclassified' items have "
                   "no mapping: their candidates are name matches against the taxonomy, a draft for a person to confirm, "
                   "and must be labelled draft classification, never applied as fact. Coverage counts recorded items only.")
    if not mapping:
        out["note"] += " No product-to-category mappings are recorded at all, so every item is unclassified."
    return out
