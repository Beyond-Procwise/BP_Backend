"""Materiality and severity (spec §7). Pure.

Overrides run LAST so nothing before them — and no rule added later — can soften
them. That ordering is the guard; tests break it on purpose (Task 15).
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Optional

from .model import CRITICALITY, NOTE, SCORED, Outcome, Result, Severity

ALWAYS_S1 = frozenset({"cumulative_total", "currency", "duplicate"})
MIN_S2 = frozenset({"payment_terms", "invoice_date"})
MAX_S3 = frozenset({"description", "no_po"})
_MONEY_OR_PARTY = frozenset({"money", "party"})


def threshold_gbp(r: Result, cfg) -> Decimal:
    floor, ceiling = cfg["materiality_floor"], cfg["materiality_ceiling"]
    if r.basis_total is None or r.fx_to_gbp is None:
        return floor
    pct = abs(r.basis_total) * r.fx_to_gbp * cfg["materiality_pct_of_total"] / Decimal(100)
    return min(max(pct, floor), ceiling)


def impact(exposure: Optional[Decimal], threshold: Optional[Decimal]) -> float:
    if exposure is None or exposure <= 0 or threshold is None or threshold <= 0:
        return 0.0
    return max(0.0, min(100.0, 50.0 * (1.0 + math.log10(float(exposure / threshold)))))


def _band(score: float, cfg) -> Severity:
    if score >= cfg["band_s1"]:
        return Severity.S1
    if score >= cfg["band_s2"]:
        return Severity.S2
    return Severity.S3


def apply_overrides(r: Result) -> Result:
    if r.outcome not in SCORED:
        return r
    applied = []
    if r.rule_id in ALWAYS_S1 and r.outcome == Outcome.CONFLICT:
        r.severity = Severity.S1
        applied.append("always_s1")
    if r.rule_id in MIN_S2 and r.outcome == Outcome.CONFLICT:
        r.severity = max(r.severity, Severity.S2)
        applied.append("min_s2")
    if r.outcome == Outcome.UNVERIFIABLE and r.field_class in _MONEY_OR_PARTY:
        r.severity = max(r.severity, Severity.S2)
        applied.append("min_s2_unverifiable")
    if r.rule_id in MAX_S3:
        r.severity = min(r.severity, Severity.S3)
        applied.append("max_s3")
    if applied:
        r.score_inputs["overrides"] = applied
    return r


def score_result(r: Result, cfg) -> Result:
    if r.outcome in (Outcome.MATCH, Outcome.WITHIN_TOL):
        r.severity, r.score = Severity.S0, 0.0
        return r
    if r.outcome in NOTE:
        r.severity, r.score = Severity.S3, 0.0
        return r
    fx_missing = r.fx_to_gbp is None and r.exposure != 0
    if fx_missing:
        # Score in document currency against the percentage part of the threshold,
        # then cap: without a rate we cannot know the amount is large in GBP.
        thr_doc = (abs(r.basis_total) * cfg["materiality_pct_of_total"] / Decimal(100)
                   if r.basis_total else None)
        imp = impact(abs(r.exposure), thr_doc) if thr_doc else 50.0
        thr = None
    else:
        thr = threshold_gbp(r, cfg)
        imp = impact(r.exposure_gbp, thr)
    crit = CRITICALITY[r.field_class]
    score = max(0.0, min(100.0, imp * crit * r.confidence))
    sev = _band(score, cfg)
    if r.outcome == Outcome.UNVERIFIABLE or fx_missing:
        sev = min(sev, Severity.S2)
    r.score, r.severity = round(score, 2), sev
    r.score_inputs = {"exposure_gbp": None if r.exposure_gbp is None else str(r.exposure_gbp),
                      "threshold_gbp": None if thr is None else str(thr),
                      "impact": round(imp, 2), "criticality": crit,
                      "confidence": round(r.confidence, 4), "fx_missing": fx_missing,
                      "fx_rate_date": None if r.fx_rate_date is None else str(r.fx_rate_date)}
    return apply_overrides(r)
