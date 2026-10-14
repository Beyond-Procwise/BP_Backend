"""Which contract replaced which?

parent_contract_id is populated on 1,561 contracts and resolves on 0: the
references were minted in a different namespace (C1543 against actual C00002).
Rather than repair ids by hand, reconstruct the chain from evidence -- same
supplier, adjacent terms, same category, comparable value, similar title.

Renewal uplift then follows, in ONE currency. Converting across currencies to
report an uplift would fabricate an FX rate.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Optional

from src.services import linking_engine as _le
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "contract_succession"
VERSION = "1.0.0"

#: A renewal starts near the predecessor's end. Wider than a day to survive
#: signature lag; narrow enough that an unrelated later contract is not a renewal.
ADJACENCY_DAYS = 120


def _to_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _cmp_derived_supplier(src, tgt) -> tuple[float, str]:
    p = src.get("_same_entity_p") or tgt.get("_same_entity_p")
    if p is None:
        return 0.5, "MISSING"
    return float(p), "OK"


def _cmp_adjacency(src, tgt) -> tuple[float, str]:
    prev_end = _to_date(src.get("contract_end_date"))
    next_start = _to_date(tgt.get("contract_start_date"))
    if prev_end is None or next_start is None:
        return 0.5, "MISSING"
    if next_start < prev_end - timedelta(days=ADJACENCY_DAYS):
        return 0.0, "CONFLICT"      # starts well before the predecessor ended
    gap = abs((next_start - prev_end).days)
    if gap <= ADJACENCY_DAYS:
        return 1.0, "OK"
    return 0.0, "CONFLICT"


def _cmp_category(src, tgt) -> tuple[float, str]:
    a = (src.get("spend_category") or "").strip().lower()
    b = (tgt.get("spend_category") or "").strip().lower()
    if not a or not b:
        return 0.5, "MISSING"
    return (1.0, "OK") if a == b else (0.0, "CONFLICT")


def _cmp_value(src, tgt) -> tuple[float, str]:
    try:
        a, b = float(src["total_contract_value"]), float(tgt["total_contract_value"])
    except (TypeError, ValueError, KeyError):
        return 0.5, "MISSING"
    if a <= 0 or b <= 0 or src.get("currency") != tgt.get("currency"):
        return 0.5, "MISSING"
    ratio = min(a, b) / max(a, b)
    if ratio >= 0.75:
        return 1.0, "OK"
    if ratio >= 0.4:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


def _cmp_title(src, tgt) -> tuple[float, str]:
    ta = set((src.get("contract_title") or "").lower().split())
    tb = set((tgt.get("contract_title") or "").lower().split())
    if not ta or not tb:
        return 0.5, "MISSING"
    j = len(ta & tb) / len(ta | tb)
    if j >= 0.7:
        return 1.0, "OK"
    if j >= 0.35:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


_le.register_signal("csx_supplier", lambda s, t, sl, tl: _cmp_derived_supplier(s, t))
_le.register_signal("csx_adjacent", lambda s, t, sl, tl: _cmp_adjacency(s, t))
_le.register_signal("csx_category", lambda s, t, sl, tl: _cmp_category(s, t))
_le.register_signal("csx_value", lambda s, t, sl, tl: _cmp_value(s, t))
_le.register_signal("csx_title", lambda s, t, sl, tl: _cmp_title(s, t))

SIGNALS = [
    {"id": "supplier_same", "cluster": "identity",    "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "csx_supplier",
     "reads": ["_same_entity_p"]},
    {"id": "adjacency",     "cluster": "temporal",    "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "csx_adjacent",
     "reads": ["contract_end_date", "contract_start_date"]},
    {"id": "category",      "cluster": "category",    "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "csx_category",
     "reads": ["spend_category"]},
    {"id": "value",         "cluster": "commercial",  "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "csx_value",
     "reads": ["total_contract_value", "currency"]},
    {"id": "title",         "cluster": "description", "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "csx_title",
     "reads": ["contract_title"]},
]

# DECLARED UNMEASURED, like contract_coverage: no labelled succession sample
# exists while every parent_contract_id dangles.
_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.35, "floor": 0.55,
    "signals": SIGNALS, "date_field": "contract_start_date",
})


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    sid, tid = str(src.get("contract_id")), str(tgt.get("contract_id"))
    out: dict[str, frozenset] = {}
    for spec in SIGNALS:
        obs: set[Observation] = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out


def score(src: dict, tgt: dict) -> dict:
    overrides = remap_clusters(SIGNALS, observations_for(src, tgt))
    return _le.score_link(src, tgt, PROFILE, cluster_overrides=overrides)


def uplift(prev: dict, nxt: dict) -> Optional[dict]:
    """Renewal uplift, or None when it cannot be stated without inventing a rate."""
    if prev.get("currency") != nxt.get("currency"):
        return None
    try:
        a, b = float(prev["total_contract_value"]), float(nxt["total_contract_value"])
    except (TypeError, ValueError, KeyError):
        return None
    if a <= 0:
        return None
    return {"currency": prev.get("currency"), "previous": a, "current": b,
            "delta": round(b - a, 2), "pct": round((b - a) / a * 100.0, 4)}
