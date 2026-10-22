"""similar_items: items bought under near-identical descriptions, with the evidence behind each. Read-only.

The demand-aggregation skill needs to find "the same thing bought separately": by different buyers, from
different suppliers, under slightly different wording. This reads every distinct item description on the
organisation's PO, invoice and quote lines, with how many lines, suppliers and buyers carry it and the
price range as billed, and ranks the ones whose wording is close to the one asked about.

Closeness is name similarity only (shared descriptive words after dropping buying words and item codes),
reported as a number so a skill can hold it against its own threshold. It never says two items ARE the
same: specification and unit of measure must be confirmed by a person. Prices are as billed, not
normalised for currency, Incoterm or order size.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.services.contract_tools import store

log = logging.getLogger(__name__)

DEFAULT_MIN_SIMILARITY = 0.6
DEFAULT_LIMIT = 20
MAX_LIMIT = 50
MIN_QUERY_CHARS = 3

# Per source: description, item_id, unit_of_measure, lines, suppliers, buyers, price_min, price_max,
# currency (one if all lines agree, else NULL), currencies, last_date.
_PO_SQL = """
    SELECT l.item_description, MIN(l.item_id), MIN(l.unit_of_measure), COUNT(*)::int,
           COUNT(DISTINCT p.supplier_id)::int, COUNT(DISTINCT p.buyer_id)::int,
           MIN(l.unit_price), MAX(l.unit_price),
           CASE WHEN COUNT(DISTINCT UPPER(l.currency)) = 1 THEN MIN(UPPER(l.currency)) END,
           COUNT(DISTINCT UPPER(l.currency))::int, MAX(p.order_date)
      FROM proc.bp_po_line_items_trgt l
      JOIN proc.bp_purchase_order_trgt p USING (po_id)
     WHERE l.item_description IS NOT NULL
     GROUP BY l.item_description
"""
_INVOICE_SQL = """
    SELECT l.item_description, MIN(l.item_id), MIN(l.unit_of_measure), COUNT(*)::int,
           COUNT(DISTINCT h.supplier_id)::int, COUNT(DISTINCT h.buyer_id)::int,
           MIN(l.unit_price), MAX(l.unit_price),
           CASE WHEN COUNT(DISTINCT UPPER(l.currency)) = 1 THEN MIN(UPPER(l.currency)) END,
           COUNT(DISTINCT UPPER(l.currency))::int, MAX(h.invoice_date)
      FROM proc.bp_invoice_line_items_trgt l
      JOIN proc.bp_invoice_trgt h ON h.invoice_id = l.invoice_id
     WHERE l.item_description IS NOT NULL
     GROUP BY l.item_description
"""
_QUOTE_SQL = """
    SELECT l.item_description, MIN(l.item_id), MIN(l.unit_of_measure), COUNT(*)::int,
           COUNT(DISTINCT q.supplier_id)::int, COUNT(DISTINCT q.buyer_id)::int,
           MIN(l.unit_price), MAX(l.unit_price),
           CASE WHEN COUNT(DISTINCT UPPER(l.currency)) = 1 THEN MIN(UPPER(l.currency)) END,
           COUNT(DISTINCT UPPER(l.currency))::int, MAX(q.quote_date)
      FROM proc.bp_quote_line_items_trgt l
      JOIN proc.bp_quote_trgt q ON q.quote_id = l.quote_id
     WHERE l.item_description IS NOT NULL
     GROUP BY l.item_description
"""
SOURCES = (("po", _PO_SQL), ("invoice", _INVOICE_SQL), ("quote", _QUOTE_SQL))

_WORD = re.compile(r"[a-z0-9]+")
_CODE = re.compile(r"^(?:[a-z]{2,4}\d{3,}|\d{5,})$")   # ITM003969, PROD00031, 520561: an item code, not a word
# Words that say how something is bought or packaged, not what it is.
_BUYING = frozenset({"advanced", "essential", "standard", "certified", "modular", "enterprise", "compact", "basic",
                     "premium", "professional", "heavy", "duty", "bundle", "kit", "unit", "module", "component",
                     "package", "pack", "installation", "subscription", "service", "services", "with", "and",
                     "the", "per", "incl", "including", "set", "pcs", "each"})


def _plain(v: Any) -> Any:
    if isinstance(v, (datetime, date)):
        return v.isoformat()
    if isinstance(v, Decimal):
        return float(v)
    return v


def _singular(word: str) -> str:
    """'computers' and 'computer' are one word for matching; 'laptops' and 'laptop' too."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def core_words(text: Any) -> frozenset:
    """The words that describe the thing itself: no item codes, no buying words, no one-letter tokens,
    plurals folded to the singular."""
    return frozenset(_singular(w) for w in _WORD.findall(str(text or "").lower())
                     if len(w) >= 2 and not _CODE.match(w) and w not in _BUYING)


def item_code(text: Any) -> Optional[str]:
    codes = [w for w in _WORD.findall(str(text or "").lower()) if _CODE.match(w)]
    return codes[-1].upper() if codes else None


def similarity(a: Any, b: Any) -> float:
    """Jaccard overlap of the descriptive words, 0 to 1. 1.0 means the same words, not the same item."""
    wa, wb = core_words(a), core_words(b)
    if not wa or not wb:
        return 0.0
    return round(len(wa & wb) / len(wa | wb), 3)


def _evidence(r: tuple) -> Dict[str, Any]:
    (_d, item_id, uom, lines, suppliers, buyers, pmin, pmax, ccy, ccys, last) = r
    out = {"lines": int(lines or 0), "suppliers": int(suppliers or 0), "buyers": int(buyers or 0),
           "price_min": _plain(pmin), "price_max": _plain(pmax), "currency": ccy, "last_date": _plain(last)}
    if (ccys or 0) > 1:
        out["currency"] = None
        out["currencies"] = int(ccys)
        out["price_note"] = "billed in more than one currency; the range is not comparable"
    return out


def _number(value: Any, default: float, lo: float, hi: float) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def find(description: Any = None, min_similarity: Any = None, limit: Any = None, *,
         rows: Optional[Callable[[str, Sequence[Any]], List[tuple]]] = None) -> Dict[str, Any]:
    query = str(description or "").strip()
    if len(query) < MIN_QUERY_CHARS:
        return {"found": False, "note": f"Give the item description to look for, at least {MIN_QUERY_CHARS} letters."}
    want = core_words(query)
    if not want:
        return {"found": False, "note": "The description has no descriptive words once item codes and buying words "
                                        "(bundle, kit, standard ...) are dropped; give what the item is."}
    threshold = _number(min_similarity, DEFAULT_MIN_SIMILARITY, 0.0, 1.0)
    cap = int(_number(limit, DEFAULT_LIMIT, 1, MAX_LIMIT))
    read = rows or store.rows
    by_desc: Dict[str, Dict[str, Any]] = {}
    try:
        for source, sql in SOURCES:
            for r in read(sql, []):
                d = str(r[0]).strip()
                if not d or not (want & core_words(d)):
                    continue  # no shared descriptive word: cannot be close
                entry = by_desc.setdefault(d, {"description": d, "item_id": r[1], "unit_of_measure": r[2], "sources": {}})
                entry["sources"][source] = _evidence(r)
                entry["item_id"] = entry["item_id"] or r[1]
                entry["unit_of_measure"] = entry["unit_of_measure"] or r[2]
    except store.StoreUnavailable:
        return {"found": False, "available": False,
                "note": "The line records could not be read just now. This does not mean no similar item exists."}
    query_code = item_code(query)
    matches = []
    for entry in by_desc.values():
        score = similarity(query, entry["description"])
        if score < threshold:
            continue
        code = item_code(entry["description"]) or (str(entry["item_id"]).upper() if entry["item_id"] else None)
        src = entry["sources"]
        entry.update({
            "name_similarity": score,
            "same_description": entry["description"].strip().lower() == query.lower(),
            "same_item_code": bool(query_code and code and query_code == code),
            "item_code": code,
            "lines_total": sum(s["lines"] for s in src.values()),
            "suppliers_max": max(s["suppliers"] for s in src.values()),
            "buyers_max": max(s["buyers"] for s in src.values()),
        })
        matches.append(entry)
    matches.sort(key=lambda m: (-m["name_similarity"], -m["lines_total"], m["description"]))
    shown = matches[:cap]
    out: Dict[str, Any] = {"found": bool(shown), "query": query, "descriptive_words": sorted(want),
                           "min_similarity": threshold, "match_count": len(matches), "matches": shown,
                           "note": ("name_similarity is the share of descriptive words two descriptions share; it is not "
                                    "identity. Treat a match as the same item only after a person confirms the "
                                    "specification and unit of measure (or the item codes are the same). Line counts, "
                                    "supplier and buyer counts are per source as recorded; suppliers and buyers are not "
                                    "de-duplicated across sources. Prices are as billed, not normalised for currency, "
                                    "Incoterm, payment terms or order size.")}
    if not shown:
        out["note"] = (f"No recorded description shares enough descriptive words with '{query}' at similarity "
                       f"{threshold}. This covers only PO, invoice and quote lines on record. " + out["note"])
    elif len(matches) > cap:
        out["note"] += f" Showing the closest {cap} of {len(matches)}; raise min_similarity to narrow."
    return out
