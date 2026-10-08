"""Score a pair on the evidence BOTH documents can supply.

Why this exists. The engine's coverage term is
    C = floor + (1 - floor) * (sum of weight*r) / (sum of weight)
and the denominator counts every signal in the profile, observed or not. A
profile that carries eight corroborating signals most contracts leave empty
therefore taxes every document for data it was never going to have: measured
2026-10-08, an exact-reference SOW fell 96.9 -> 75.7 and a same-supplier SOW
with no reference fell 75.6 -> 61.4, under the 65.0 proposal floor.

The engine already has the idea this needs -- applicability (`appl`, `q`) -- but
fixes it per profile. Here it is decided per PAIR, outside the engine: an
optional signal joins only when its comparator can actually be evaluated on this
pair (status != MISSING). The pair is then scored under a variant profile named
base + "+" + sorted optional ids, registered on first use.

linking_engine.py is not edited; this uses only register_profile and score_link.
"""
from __future__ import annotations

from typing import Sequence

from src.services import linking_engine as _le
from .composition import remap_clusters


def applicable(optional: Sequence[dict], src: dict, tgt: dict, date_field: str) -> list[dict]:
    """The optional signals this pair can actually be evaluated on."""
    keep = []
    for spec in optional:
        _s, status = _le._signal_match(spec["kind"], src, tgt, [], [], date_field)
        if status != "MISSING":
            keep.append(spec)
    return keep


def variant_name(base: str, extra: Sequence[dict]) -> str:
    """Deterministic: independent of the order the optional signals were declared in."""
    return base if not extra else base + "+" + "+".join(sorted(s["id"] for s in extra))


def _observations(specs: Sequence[dict], src: dict, tgt: dict) -> dict:
    sid, tid = str(src.get("contract_id", id(src))), str(tgt.get("contract_id", id(tgt)))
    out = {}
    for spec in specs:
        obs = set()
        for field in spec["reads"]:
            if src.get(field) is not None:
                obs.add((sid, field))
            if tgt.get(field) is not None:
                obs.add((tid, field))
        out[spec["id"]] = frozenset(obs)
    return out


def score_pair(base_name: str, base_specs: Sequence[dict],
               optional: Sequence[dict], src: dict, tgt: dict) -> dict:
    """Score src -> tgt under base_specs plus the optional signals that apply.

    The base profile must already be registered. Its p0/alpha/floor/date_field
    are read from the registry on EVERY call and the variant is re-registered
    from them, never cached: scripts/graph_resolution/calibrate.py tunes a
    profile by changing those values, and a variant holding the copy taken at
    first use would go on scoring with the old ones without anyone noticing.
    Re-registering is one dict assignment.

    The result is the engine's full auditable score_link result with
    ``profile`` set to the BASE name, so callers and the UNCALIBRATED_PROFILES
    check never see a variant name.
    """
    params = {k: v for k, v in _le.PROFILES[base_name].items() if k != "signals"}
    extra = applicable(optional, src, tgt, params["date_field"])
    name = variant_name(base_name, extra)
    specs = list(base_specs) + extra
    _le.register_profile(name, {**params, "signals": specs})
    overrides = remap_clusters(specs, _observations(specs, src, tgt))
    result = _le.score_link(src, tgt, name, cluster_overrides=overrides)
    result["profile"] = base_name
    return result
