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
