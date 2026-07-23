# src/services/requirement_similarity.py
"""Rivalry correlation: the quote_rival signal set.

The existing engine models CONTINUITY (a supplier's own quote -> PO -> invoice:
same supplier, same price). A competitive sourcing event is the opposite
signature — different suppliers, different prices, SAME products and quantities —
so scoring rivals with the continuity profile actively rejects them (cmp_supplier
returns CONFLICT on the highest-weighted signal). This module supplies the second
relation: supplier/price divergence is expected, not penalised. Pure — no DB.
"""
from __future__ import annotations

from src.services.linking_engine import _tokens, _to_float, cmp_exact_ref


def _agg_tokens(lines: list[dict]) -> set:
    toks: set = set()
    for ln in lines or []:
        toks |= _tokens(ln.get("item_description"))
    return toks


def cmp_desc_overlap(src_lines, tgt_lines) -> tuple[float, str]:
    """Same-requirement signal: token Jaccard over all line descriptions."""
    a, b = _agg_tokens(src_lines), _agg_tokens(tgt_lines)
    if not a or not b:
        return 0.5, "MISSING"
    inter = len(a & b)
    union = len(a | b)
    s = inter / union if union else 0.0
    return s, ("OK" if s >= 0.5 else "WEAK" if s > 0 else "CONFLICT")


def _total_qty(lines: list[dict]):
    qs = [_to_float(ln.get("quantity")) for ln in (lines or [])]
    qs = [q for q in qs if q is not None]
    return sum(qs) if qs else None


def cmp_volume(src_lines, tgt_lines) -> tuple[float, str]:
    """Same-requirement volume signal, graded by quantity-total ratio."""
    a, b = _total_qty(src_lines), _total_qty(tgt_lines)
    if a is None or b is None:
        return 0.5, "MISSING"
    hi = max(a, b)
    if hi == 0:
        return (1.0, "OK") if a == b else (0.0, "CONFLICT")
    s = min(a, b) / hi
    return s, ("OK" if s >= 0.9 else "WEAK" if s >= 0.5 else "CONFLICT")


def _amount(row: dict):
    return _to_float(row.get("converted_amount_usd")) or _to_float(row.get("total_amount"))


def cmp_price_prox(a_row: dict, b_row: dict) -> tuple[float, str]:
    """Price PROXIMITY (not equality). Graded ratio min/max with NO cutoff: rivals
    at 1.08x score high; freight-vs-IT at 312x contributes ~0 but is never a hard
    reject (spec: no signal is a gate)."""
    a, b = _amount(a_row), _amount(b_row)
    if a is None or b is None or a <= 0 or b <= 0:
        return 0.5, "MISSING"
    s = min(a, b) / max(a, b)
    return s, ("OK" if s >= 0.85 else "WEAK" if s >= 0.5 else "CONFLICT")


def cmp_buyer(a_row: dict, b_row: dict) -> tuple[float, str]:
    """Tier-4 corroboration only: same buying entity. A match cannot discriminate
    (uniform across the batch); a mismatch is meaningful."""
    return cmp_exact_ref(a_row.get("buyer_id"), b_row.get("buyer_id"))


from src.services import linking_engine as _le

# Register rivalry signal kinds against the engine's dispatch seam.
_le.register_signal("desc_overlap", lambda src, tgt, sl, tl: cmp_desc_overlap(sl, tl))
_le.register_signal("volume",       lambda src, tgt, sl, tl: cmp_volume(sl, tl))
_le.register_signal("price_prox",   lambda src, tgt, sl, tl: cmp_price_prox(src, tgt))
_le.register_signal("buyer",        lambda src, tgt, sl, tl: cmp_buyer(src, tgt))

# quote_rival: the tier 1-4 signal set. NO supplier_id, NO exact amount — divergence on
# those is the SIGNATURE of a competitive event, not a defect. Tier 2 = requirement
# (description + volume). Tier 3 = price proximity. Tier 4 = corroboration (buyer,
# currency, location) which may raise/lower confidence but must never discriminate.
# CONSTANTS BELOW ARE CALIBRATED against tests/fixtures/deal_clustering/golden_batch.py
# (Task 5 test + Task 6 cluster test) — see the tuning notes below the table.
_RIVAL_SIGNALS = [
    {"id": "desc",     "cluster": "requirement", "tier": 2, "weight": 5, "appl": 1.0, "cap": 0.90, "kind": "desc_overlap"},
    {"id": "volume",   "cluster": "requirement", "tier": 2, "weight": 4, "appl": 1.0, "cap": 0.90, "kind": "volume"},
    {"id": "price",    "cluster": "commercial",  "tier": 3, "weight": 2, "appl": 1.0, "cap": 0.90, "kind": "price_prox"},
    {"id": "buyer",    "cluster": "context",     "tier": 4, "weight": 1, "appl": 1.0, "cap": 0.95, "kind": "buyer"},
    {"id": "currency", "cluster": "context",     "tier": 4, "weight": 1, "appl": 1.0, "cap": 0.95, "kind": "currency"},
]
# Tuning (calibrated against tests/fixtures/deal_clustering/golden_batch.py):
# p0=0.03, alpha=0.55 (brief's starting point was p0=0.10/alpha=0.35 — both raised).
# Weights/caps are UNCHANGED from the brief; only p0/alpha (steepness) were tuned, per the
# brief's own guidance ("adjust alpha/floor before weights").
#
# Why: the fixture's same-event pairs cluster in a narrow evidence band (total weighted
# cluster score ~9.2-11.3 across all four events — descriptions are near-identical within
# an event, final negotiated prices converge), while cross-category pairs sit far outside
# it (desc/volume conflict pushes the score deeply negative). p0=0.10/alpha=0.35 undershoots
# (same-event pairs land ~75-85%, below the required >=0.90 floor for a tight rival pair);
# p0=0.03/alpha=0.55 lands same-event pairs at ~91-96% (comfortably >=0.90) while still
# driving freight<->IT to ~1.6% and freight<->consultancy to ~0.04% (both near-zero, no
# hard cutoff needed — the log-odds curve alone does the job). floor stays at 0.55 (matches
# the continuity profiles); it is inert in this fixture because every quote_rival signal is
# always observable (no MISSING path is exercised in-batch), but keeps the profile's
# coverage behaviour sane if a real quote is ever missing e.g. its buyer_id.
_le.register_profile("quote_rival", {
    "p0": 0.03, "alpha": 0.55, "floor": 0.55,
    "signals": _RIVAL_SIGNALS, "date_field": "quote_date",
})


def rivalry_score(bid_a: dict, bid_b: dict, lines_a: list, lines_b: list) -> dict:
    """Correlation between two collapsed bids using the quote_rival profile.
    Returns score_link's full auditable result plus correlation = F/100 in [0,1]."""
    result = _le.score_link(bid_a, bid_b, "quote_rival", lines_a, lines_b)
    result["correlation"] = round(result["F"] / 100.0, 4)
    return result
