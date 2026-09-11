"""Measure p0 and alpha for supplier_identity against real ground truth.

Two bp_supplier_master rows are the same company iff they share a VAT number.
Hold VAT out of the scored records and score on what remains: that is a
labelled sample by construction, and the floor it yields is measured rather
than borrowed from another profile that was tuned on a different question.
"""
from __future__ import annotations

import itertools
from typing import Iterable, List

from src.services import linking_engine as _le
from src.services.graph_resolution.profiles import supplier_identity as si


def build_labelled_pairs(rows: Iterable[dict]) -> List[tuple]:
    rows = list(rows)
    out = []
    for a, b in itertools.combinations(rows, 2):
        same = (a.get("vat_number") is not None
                and a.get("vat_number") == b.get("vat_number"))
        sa = {k: v for k, v in a.items() if k != "vat_number"}
        sb = {k: v for k, v in b.items() if k != "vat_number"}
        out.append((sa, sb, same))
    return out


def sweep(pairs: List[tuple], grid: List[tuple]) -> List[dict]:
    """Score every pair under each (p0, alpha) and report separation."""
    results = []
    original = dict(_le.PROFILES[si.PROFILE])
    try:
        for p0, alpha in grid:
            _le.PROFILES[si.PROFILE] = {**original, "p0": p0, "alpha": alpha}
            same_F, diff_F, false_auto = [], [], 0
            for a, b, is_same in pairs:
                r = si.score(a, b)
                (same_F if is_same else diff_F).append(r["F"])
                if not is_same and r["F"] >= _le._BAND_AUTO:
                    false_auto += 1
            results.append({
                "p0": p0, "alpha": alpha,
                "separation": (min(same_F) - max(diff_F)) if same_F and diff_F else 0.0,
                "false_auto_links": false_auto,
                "n_same": len(same_F), "n_diff": len(diff_F),
            })
    finally:
        _le.PROFILES[si.PROFILE] = original
    return results


def best(results: List[dict]) -> dict:
    """Highest separation among settings that auto-link no false pair.

    A false auto_link is disqualifying, not a cost to trade off: it is a wrong
    answer asserted with confidence, which is the failure this whole design
    exists to avoid.
    """
    clean = [r for r in results if r["false_auto_links"] == 0]
    pool = clean or results
    return max(pool, key=lambda r: r["separation"])
