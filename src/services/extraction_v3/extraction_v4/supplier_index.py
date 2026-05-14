"""Supplier name index for the v4 hybrid extractor.

The reference pipeline scans extracted text for known supplier names to anchor
vendor/supplier extraction. Upstream that index lives in supplier_names.json on
disk; here we hydrate it from proc.bp_supplier on first use and write through
when the extractor encounters a new supplier name.

Public API mirrors the reference's module-level helpers so engine.py can drop
in `from .supplier_index import _load_supplier_names, _match_supplier,
_scan_text_for_supplier` with no behaviour change.

Cache contract:
- First call hydrates from DB (proc.bp_supplier.supplier_name + trading_name).
- A copy is written to resources/supplier_names.json so cold-start without a
  DB is still useful (smoke tests, CI).
- Calling `add_supplier_to_cache(name)` appends in-memory and writes through to
  the JSON file. Persistence to proc.bp_supplier is handled separately by
  supplier_resolver.resolve_or_create_supplier() during DB writes.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Iterable

log = logging.getLogger(__name__)

_RESOURCES_DIR = Path(__file__).parent / "resources"
_JSON_PATH = _RESOURCES_DIR / "supplier_names.json"

_SUPPLIER_NAMES: list[str] = []
_SUPPLIER_NAMES_LOWER: dict[str, str] = {}
_SUPPLIER_NAMES_NORM: dict[str, str] = {}
_SUPPLIER_CORE_TOKENS: dict[str, set[str]] = {}
_HYDRATED: bool = False

_BIZ_SUFFIXES = re.compile(
    r"\b(ltd|limited|llc|inc|plc|corp|corporation|gmbh|"
    r"s\.?a\.?|pty|co\.?|company|group|partners|llp|lp|"
    r"and\s+sons|& sons)\b\.?",
    re.IGNORECASE,
)
_NOISE_WORDS = {"and", "the", "of", "for", "a", "an", "&", "etc"}


def _normalize(name: str) -> str:
    s = name.lower().strip()
    s = _BIZ_SUFFIXES.sub("", s)
    s = re.sub(r"[,.\-&'\"]+", " ", s)
    return " ".join(s.split()).strip()


def _core_tokens(name: str) -> set[str]:
    norm = _normalize(name)
    return {t for t in norm.split() if t and t not in _NOISE_WORDS}


def _rebuild_indexes(names: Iterable[str]) -> None:
    global _SUPPLIER_NAMES, _SUPPLIER_NAMES_LOWER, _SUPPLIER_NAMES_NORM, _SUPPLIER_CORE_TOKENS
    _SUPPLIER_NAMES = sorted({n.strip() for n in names if n and n.strip()})
    _SUPPLIER_NAMES_LOWER = {n.lower(): n for n in _SUPPLIER_NAMES}
    _SUPPLIER_NAMES_NORM = {_normalize(n): n for n in _SUPPLIER_NAMES if _normalize(n)}
    _SUPPLIER_CORE_TOKENS = {n: _core_tokens(n) for n in _SUPPLIER_NAMES}


def _hydrate_from_db() -> list[str]:
    """Pull all supplier names from proc.bp_supplier. Returns [] on any failure."""
    try:
        from src.services.db import get_conn
    except Exception as exc:
        log.debug("supplier_index: db module unavailable (%s)", exc)
        return []
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT supplier_name, trading_name FROM proc.bp_supplier"
                )
                names: set[str] = set()
                for sn, tn in cur.fetchall():
                    if sn:
                        names.add(str(sn).strip())
                    if tn:
                        names.add(str(tn).strip())
                return sorted(n for n in names if n)
    except Exception as exc:
        log.warning("supplier_index: DB hydration failed (%s); falling back to JSON", exc)
        return []


def _load_from_json() -> list[str]:
    if not _JSON_PATH.exists():
        return []
    try:
        with open(_JSON_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return list(data.get("suppliers") or [])
        if isinstance(data, list):
            return list(data)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("supplier_index: JSON load failed (%s)", exc)
    return []


def _write_json(names: list[str]) -> None:
    try:
        _RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
        with open(_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump({"suppliers": names}, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        log.debug("supplier_index: JSON write-through failed (%s)", exc)


def _load_supplier_names() -> None:
    """Hydrate the index. Idempotent — safe to call repeatedly.

    Order: DB → JSON cache → empty. Writes DB result through to JSON cache so
    next cold start is fast even without DB connectivity.
    """
    global _HYDRATED
    if _HYDRATED:
        return

    db_names = _hydrate_from_db()
    if db_names:
        _rebuild_indexes(db_names)
        _write_json(db_names)
        log.info("supplier_index: hydrated %d names from DB", len(db_names))
        _HYDRATED = True
        return

    json_names = _load_from_json()
    if json_names:
        _rebuild_indexes(json_names)
        log.info("supplier_index: hydrated %d names from JSON cache", len(json_names))
        _HYDRATED = True
        return

    _rebuild_indexes([])
    log.warning("supplier_index: starting with empty supplier list")
    _HYDRATED = True


def add_supplier_to_cache(name: str) -> None:
    """Append a newly-discovered supplier name to the in-memory + JSON cache.

    Does NOT write to proc.bp_supplier — that happens in
    supplier_resolver.resolve_or_create_supplier during persist.
    """
    if not name or not name.strip():
        return
    name = name.strip()
    if name.lower() in _SUPPLIER_NAMES_LOWER:
        return
    new_list = sorted(set(_SUPPLIER_NAMES) | {name})
    _rebuild_indexes(new_list)
    _write_json(new_list)


def reset_for_tests() -> None:
    """Clear hydration state. Tests only."""
    global _HYDRATED
    _HYDRATED = False
    _rebuild_indexes([])


def _match_supplier(extracted_name: str) -> str:
    """Reference-compatible: find best canonical name for an extracted string.

    Returns "" if no match above threshold.
    """
    if not extracted_name:
        return ""
    _load_supplier_names()
    if not _SUPPLIER_NAMES:
        return ""
    query = extracted_name.strip()
    query_lower = query.lower()
    query_norm = _normalize(query)
    query_core = _core_tokens(query)
    if query_lower in _SUPPLIER_NAMES_LOWER:
        return _SUPPLIER_NAMES_LOWER[query_lower]
    best_contain = ""
    best_contain_len = 0
    for s_norm, s_orig in _SUPPLIER_NAMES_NORM.items():
        if not s_norm or len(s_norm) < 3:
            continue
        if (
            re.search(r"(?:^|\s)" + re.escape(s_norm) + r"(?:\s|$)", query_norm)
            or re.search(r"(?:^|\s)" + re.escape(query_norm) + r"(?:\s|$)", s_norm)
        ):
            if len(s_norm) > best_contain_len:
                best_contain = s_orig
                best_contain_len = len(s_norm)
    if best_contain:
        return best_contain
    if not query_core:
        return ""
    best_score = 0.0
    best_name = ""
    for s_orig, s_tokens in _SUPPLIER_CORE_TOKENS.items():
        if not s_tokens:
            continue
        overlap = query_core & s_tokens
        n_overlap = len(overlap)
        if n_overlap == 0:
            continue
        all_query_matched = query_core <= s_tokens
        all_supplier_matched = s_tokens <= query_core
        if n_overlap == 1 and not (all_query_matched or all_supplier_matched):
            continue
        union = len(query_core | s_tokens)
        jaccard = n_overlap / union if union else 0.0
        score = jaccard
        if all_query_matched and len(query_core) >= 2:
            score = max(score, 0.7)
        if score > best_score:
            best_score = score
            best_name = s_orig
    if best_score >= 0.5:
        return best_name
    return ""


def _scan_text_for_supplier(text: str) -> str:
    """Reference-compatible: scan free text for any known supplier name."""
    if not text:
        return ""
    _load_supplier_names()
    if not _SUPPLIER_NAMES:
        return ""
    text_lower = text.lower()
    best_match = ""
    best_len = 0
    for s_lower, s_orig in _SUPPLIER_NAMES_LOWER.items():
        if len(s_lower) < 4:
            continue
        pattern = r"\b" + re.escape(s_lower) + r"\b"
        if re.search(pattern, text_lower):
            if len(s_lower) > best_len:
                best_match = s_orig
                best_len = len(s_lower)
    if best_match:
        return best_match
    for s_orig in _SUPPLIER_NAMES:
        s_norm = _normalize(s_orig)
        if not s_norm or len(s_norm) < 4:
            continue
        norm_words = [w for w in s_norm.split() if w not in _NOISE_WORDS]
        if len(norm_words) < 2:
            continue
        pattern = r"\b" + re.escape(s_norm) + r"\b"
        if re.search(pattern, text_lower):
            if len(s_norm) > best_len:
                best_match = s_orig
                best_len = len(s_norm)
    return best_match
