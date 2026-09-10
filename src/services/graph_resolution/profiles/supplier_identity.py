"""Are these two supplier records the same company?

Transactions key suppliers as SUP-<Name> (3,510 distinct), contracts as S####
(2,545). Measured overlap: zero. Nothing joins today, which is why every
contract question is unanswerable. Registration identifiers are the bridge:
bp_supplier_master carries VAT, registration number and DUNS on 1,009 of 1,009
rows.
"""
from __future__ import annotations

from typing import Optional

from src.services import linking_engine as _le
from ..composition import remap_clusters
from ..observations import Observation

PROFILE = "supplier_identity"
VERSION = "1.0.0"


def _norm(v) -> Optional[str]:
    if v is None:
        return None
    s = " ".join(str(v).split()).strip().lower()
    return s or None


def _cmp_exact(a, b) -> tuple[float, str]:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    return (1.0, "OK") if na == nb else (0.0, "CONFLICT")


def _cmp_name(a, b) -> tuple[float, str]:
    na, nb = _norm(a), _norm(b)
    if na is None or nb is None:
        return 0.5, "MISSING"
    if na == nb:
        return 1.0, "OK"
    short, lng = sorted([na, nb], key=len)
    if len(short) >= 8 and lng.startswith(short):
        return 0.7, "WEAK"
    return 0.0, "CONFLICT"


def _cmp_address(src, tgt) -> tuple[float, str]:
    pa, pb = _norm(src.get("postal_code")), _norm(tgt.get("postal_code"))
    ca, cb = _norm(src.get("country")), _norm(tgt.get("country"))
    if pa is None or pb is None:
        return 0.5, "MISSING"
    if pa == pb and ca == cb:
        return 1.0, "OK"
    if ca == cb:
        return 0.6, "WEAK"
    return 0.0, "CONFLICT"


_le.register_signal("si_name", lambda s, t, sl, tl: _cmp_name(
    s.get("supplier_name"), t.get("supplier_name")))
_le.register_signal("si_vat", lambda s, t, sl, tl: _cmp_exact(
    s.get("vat_number"), t.get("vat_number")))
_le.register_signal("si_reg", lambda s, t, sl, tl: _cmp_exact(
    s.get("registration_number"), t.get("registration_number")))
_le.register_signal("si_duns", lambda s, t, sl, tl: _cmp_exact(
    s.get("duns_number"), t.get("duns_number")))
_le.register_signal("si_addr", lambda s, t, sl, tl: _cmp_address(s, t))
_le.register_signal("si_bank", lambda s, t, sl, tl: _cmp_exact(
    s.get("bank_account_number"), t.get("bank_account_number")))

# Registration identifiers share a cluster: a company matching on VAT usually
# matches on registration number too, and dampening stops that reading as three
# independent confirmations.
SIGNALS = [
    {"id": "name",    "cluster": "identity",     "tier": 2, "weight": 4, "appl": 1.0, "cap": 0.65, "kind": "si_name",
     "reads": ["supplier_name"]},
    {"id": "vat",     "cluster": "registration", "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "si_vat",
     "reads": ["vat_number"]},
    {"id": "reg_no",  "cluster": "registration", "tier": 1, "weight": 5, "appl": 1.0, "cap": 0.45, "kind": "si_reg",
     "reads": ["registration_number"]},
    {"id": "duns",    "cluster": "registration", "tier": 2, "weight": 3, "appl": 1.0, "cap": 0.65, "kind": "si_duns",
     "reads": ["duns_number"]},
    {"id": "addr",    "cluster": "context",      "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "si_addr",
     "reads": ["postal_code", "country"]},
    {"id": "bank",    "cluster": "financial",    "tier": 1, "weight": 4, "appl": 1.0, "cap": 0.45, "kind": "si_bank",
     "reads": ["bank_account_number"]},
]

# p0/alpha are MEASURED in Task 5 against bp_supplier_master. These are the
# starting values the calibration script refines; they are not a borrowed guess
# left in place.
_le.register_profile(PROFILE, {
    "p0": 0.02, "alpha": 0.30, "floor": 0.55,
    "signals": SIGNALS, "date_field": "created_date",
})


def observations_for(src: dict, tgt: dict) -> dict[str, frozenset]:
    """Which (document, field) pairs each signal actually read."""
    sid, tid = str(src.get("supplier_id")), str(tgt.get("supplier_id"))
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
