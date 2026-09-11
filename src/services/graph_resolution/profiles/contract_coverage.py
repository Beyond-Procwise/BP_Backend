"""Is this document covered by a contract?

The contract side is ready: 958 Active contracts, 957 with a full term window,
892 distinct suppliers. What was missing was a way to reach the supplier --
contract_id is populated on 0 of 38,498 transaction rows and the supplier
keyspaces have zero overlap. SAME_ENTITY supplies that reach.

This is a SCOPE statement, never a PRICE statement: contract.yaml declares
db_lines_table: null, so no contracted unit price exists to compare against.
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from src.services import linking_engine as _le
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "contract_coverage"
VERSION = "1.0.0"


def _to_date(v) -> Optional[date]:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(str(v)[:10], fmt).date()
        except ValueError:
            continue
    return None


def _cmp_contract_ref(a, b) -> tuple[float, str]:
    na = str(a).strip().lower() if a else None
    nb = str(b).strip().lower() if b else None
    if not na or not nb:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


def _cmp_derived(src, tgt, field) -> tuple[float, str]:
    p = src.get(field)
    if p is None:
        p = tgt.get(field)
    if p is None:
        return 0.5, "MISSING"
    return float(p), "OK"


def _cmp_in_term(src, tgt) -> tuple[float, str]:
    d = _to_date(src.get("invoice_date") or src.get("order_date"))
    start = _to_date(tgt.get("contract_start_date"))
    end = _to_date(tgt.get("contract_end_date"))
    if d is None or start is None or end is None:
        return 0.5, "MISSING"
    if start <= d <= end:
        return 1.0, "OK"
    return 0.0, "CONFLICT"


def _cmp_category(src, tgt) -> tuple[float, str]:
    a = (src.get("spend_category") or "").strip().lower()
    b = (tgt.get("spend_category") or "").strip().lower()
    if not a or not b:
        return 0.5, "MISSING"
    return (1.0, "OK") if a == b else (0.0, "CONFLICT")


def _cmp_within_value(src, tgt) -> tuple[float, str]:
    try:
        amt = float(src.get("invoice_total_incl_tax") or src.get("total_amount"))
        val = float(tgt.get("total_contract_value"))
    except (TypeError, ValueError):
        return 0.5, "MISSING"
    if val <= 0:
        return 0.5, "MISSING"
    if (src.get("currency") or "") != (tgt.get("currency") or ""):
        # Never convert to compare. FX is populated on 10 of 12,408 invoices.
        return 0.5, "MISSING"
    return (1.0, "OK") if amt <= val else (0.3, "WEAK")


_le.register_signal("cc_ref", lambda s, t, sl, tl: _cmp_contract_ref(
    s.get("contract_id"), t.get("contract_id")))
_le.register_signal("cc_supplier", lambda s, t, sl, tl: _cmp_derived(s, t, "_same_entity_p"))
_le.register_signal("cc_term", lambda s, t, sl, tl: _cmp_in_term(s, t))
_le.register_signal("cc_category", lambda s, t, sl, tl: _cmp_category(s, t))
_le.register_signal("cc_value", lambda s, t, sl, tl: _cmp_within_value(s, t))
_le.register_signal("cc_item", lambda s, t, sl, tl: _cmp_derived(s, t, "_item_under_contract_p"))

SIGNALS = [
    {"id": "contract_ref",  "cluster": "reference",  "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "cc_ref",
     "reads": ["contract_id"]},
    {"id": "supplier_same", "cluster": "identity",   "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "cc_supplier",
     "reads": ["_same_entity_p"]},
    {"id": "date_in_term",  "cluster": "temporal",   "tier": 1, "weight": 4, "appl": 1.0, "cap": 0.45, "kind": "cc_term",
     "reads": ["invoice_date", "contract_start_date", "contract_end_date"]},
    {"id": "category",      "cluster": "category",   "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "cc_category",
     "reads": ["spend_category"]},
    {"id": "amount_value",  "cluster": "commercial", "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "cc_value",
     "reads": ["invoice_total_incl_tax", "total_contract_value", "currency"]},
    {"id": "item_covered",  "cluster": "line",       "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.70, "kind": "cc_item",
     "reads": ["_item_under_contract_p"]},
]

# p0/alpha DECLARED UNMEASURED: zero contract nodes existed to calibrate
# against. edge_writer refuses auto_link for this profile until that changes.
_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.30, "floor": 0.55,
    "signals": SIGNALS, "date_field": "invoice_date",
})


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    sid = str(src.get("invoice_id") or src.get("po_id") or "?")
    tid = str(tgt.get("contract_id") or "?")
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


def uncovered_reason(result: dict) -> str:
    """Why this document is not covered, in words a supplier conversation
    survives. Absence of evidence and conflicting evidence are different
    findings and are reported differently."""
    by_id = {s["id"]: s for s in result["signals"]}
    parts = []
    if by_id["date_in_term"]["status"] == "CONFLICT":
        parts.append("dated outside every active contract term for this supplier")
    elif by_id["date_in_term"]["status"] == "MISSING":
        parts.append("no usable contract term dates")
    if by_id["supplier_same"]["status"] == "MISSING":
        parts.append("supplier could not be resolved to a contracted entity")
    if by_id["contract_ref"]["status"] == "MISSING":
        parts.append("no contract reference on the document")
    return "; ".join(parts) or "no corroborating evidence above the reporting floor"
