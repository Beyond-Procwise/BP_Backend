"""Are these two lines the same product?

item_id is present on 55,421 of 55,483 invoice lines, so the reference signal
carries most cases; the rest is descriptive drift across suppliers. Connecting
193,857 line nodes through a shared Item is what turns price comparison from
string-matching into a graph question.

UoM is RECORDED, never converted: a pack of 10 and 10 each are related, not
equal, and bp_uom_canonical (38 rows) is the only UoM authority.

Calibrated 2026-09-10 (Task 7, scripts/graph_resolution/calibrate_item_equivalence.py)
against live proc.bp_invoice_line_items_trgt: 4,755 distinct item_id values
appear on more than one of the 55,483 lines (measured). Sample: 379 lines --
all lines (capped at 5/item) from 60 randomly chosen repeated-item_id groups,
plus 150 random singleton-item_id lines for cross-item variety -- scored
pairwise (item_id withheld from the scored records) = 71,631 pairs
(n_same=562, n_diff=71,069). Grid swept in three passes, coarse to fine:
p0 in {0.01,0.02,0.05,0.08} x alpha in {0.35..2.00} (12 pts), then
p0 in {0.005,0.01} x alpha up to 10.0 to find where separation turns over,
then p0 in {0.003,0.005,0.007,0.01} x alpha in {2.2..3.5} around the peak.
Separation is not monotone in alpha: it peaks and then collapses (both
same- and diff-pair scores saturate toward 1 as alpha grows, closing the
gap; by alpha=6-10 same_F and diff_F invert). Chosen: p0=0.003, alpha=3.0 --
separation=62.13 (min same_F=71.83, max diff_F=9.69, median same_F=80.71,
median diff_F=0.0), zero false auto_links across the whole grid tested.
supplier_same carries weight=1, not a symmetric-identity 3: no SAME_ENTITY
edge exists anywhere in this corpus, so the signal is permanently MISSING in
practice, and a heavier weight would permanently tax coverage C for every
pair regardless of alpha (see SIGNALS comment below for the worked ceiling).
"""
from __future__ import annotations

import hashlib
from typing import List, Optional

from src.services import linking_engine as _le
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "item_equivalence"
VERSION = "1.0.0"


def _norm(v) -> Optional[str]:
    if v is None:
        return None
    s = " ".join(str(v).split()).strip().lower()
    return s or None


def _tokens(v) -> set:
    n = _norm(v)
    return set(n.split()) if n else set()


def _cmp_item_id(a, b) -> tuple[float, str]:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


def _cmp_desc(a, b) -> tuple[float, str]:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.5, "MISSING"
    j = len(ta & tb) / len(ta | tb)
    if j >= 0.8:
        return 1.0, "OK"
    if j >= 0.5:
        return 0.7, "WEAK"
    return j, "CONFLICT" if j < 0.2 else "WEAK"


def _cmp_uom(a, b) -> tuple[float, str]:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    if na == nb:
        return 1.0, "OK"
    return 0.0, "CONFLICT"


def _cmp_price(a, b) -> tuple[float, str]:
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return 0.5, "MISSING"
    if fa <= 0 or fb <= 0:
        return 0.5, "MISSING"
    ratio = min(fa, fb) / max(fa, fb)
    if ratio >= 0.95:
        return 1.0, "OK"
    if ratio >= 0.70:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


def _cmp_derived_supplier(src, tgt) -> tuple[float, str]:
    """Reads a SAME_ENTITY edge's stored P_raw, stamped on the row by the caller.

    An absent edge is not weak evidence, it is no evidence: q=0, contributing
    nothing in either direction.
    """
    p = src.get("_same_entity_p")
    if p is None:
        p = tgt.get("_same_entity_p")
    if p is None:
        return 0.5, "MISSING"
    return float(p), "OK"


_le.register_signal("ie_item_id", lambda s, t, sl, tl: _cmp_item_id(
    s.get("item_id"), t.get("item_id")))
_le.register_signal("ie_desc", lambda s, t, sl, tl: _cmp_desc(
    s.get("item_description"), t.get("item_description")))
_le.register_signal("ie_uom", lambda s, t, sl, tl: _cmp_uom(
    s.get("unit_of_measure"), t.get("unit_of_measure")))
_le.register_signal("ie_price", lambda s, t, sl, tl: _cmp_price(
    s.get("unit_price"), t.get("unit_price")))
_le.register_signal("ie_supplier", lambda s, t, sl, tl: _cmp_derived_supplier(s, t))

SIGNALS = [
    {"id": "item_id",       "cluster": "reference",   "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "ie_item_id",
     "reads": ["item_id"]},
    {"id": "desc",          "cluster": "description", "tier": 2, "weight": 4, "appl": 1.0, "cap": 0.70, "kind": "ie_desc",
     "reads": ["item_description"]},
    {"id": "uom",           "cluster": "description", "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "ie_uom",
     "reads": ["unit_of_measure"]},
    {"id": "price",         "cluster": "commercial",  "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "ie_price",
     "reads": ["unit_price"]},
    # weight=1, not the 3 a symmetric identity signal would otherwise carry:
    # no SAME_ENTITY edge exists anywhere in this corpus (see module docstring),
    # so this signal is permanently MISSING in practice. A heavier weight would
    # permanently tax the coverage term C for every pair ever scored -- pairs
    # that agree on every other signal would still be capped under the
    # auto_link band because of a comparator that never has anything to say.
    {"id": "supplier_same", "cluster": "identity",    "tier": 2, "weight": 1, "appl": 1.0, "cap": 0.70, "kind": "ie_supplier",
     "reads": ["_same_entity_p"]},
]

_le.register_profile(PROFILE, {
    "p0": 0.003, "alpha": 3.0, "floor": 0.55,
    "signals": SIGNALS, "date_field": "delivery_date",
})


def _line_id(row: dict) -> str:
    return str(row.get("invoice_line_id") or row.get("po_line_id")
               or row.get("quote_line_id") or row.get("item_id") or "?")


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    sid, tid = _line_id(src), _line_id(tgt)
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


def item_key(members: List[dict]) -> str:
    """Digest of the class's canonical member: the lowest item_id, else the
    lowest normalised description.

    The choice of canonical member is arbitrary; its DETERMINISM is not. A
    rebuild must reproduce the same item_key or every OF_ITEM edge, and every
    finding citing one, breaks on the next pass.
    """
    ids = sorted(_norm(m.get("item_id")) for m in members if _norm(m.get("item_id")))
    if ids:
        basis = f"id:{ids[0]}"
    else:
        descs = sorted(_norm(m.get("item_description")) for m in members
                       if _norm(m.get("item_description")))
        basis = f"desc:{descs[0]}" if descs else "unknown"
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:24]
