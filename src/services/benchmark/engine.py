"""Deterministic benchmark pricing engine — pure calculation, no I/O, no LLM.

Reproduces the calculation contract of the Benchmark Calculations.xlsx
prototype. Rounding is part of the contract: every intermediate is rounded
exactly where the prototype rounds (stepwise), Excel-style (half away from
zero), so results match the workbook to the penny.
"""
from __future__ import annotations

import logging
from decimal import Decimal, ROUND_HALF_UP
from statistics import median as _true_median
from typing import Callable, Mapping, Optional, Sequence

from .models import (
    BenchmarkPoint,
    BenchmarkResult,
    BenchmarkSettings,
    QuoteLine,
)

logger = logging.getLogger(__name__)

# Mirrors the prototype's IFERROR(..., 1): any arithmetic failure -> 1.0.
# TypeError covers None profiles (arithmetic against a missing weighted
# average) and complex values: a negative base with a fractional exponent
# (quantity/ref_quantity < 0) returns complex, and _clamp's min()/max()
# raise TypeError comparing complex to float. If a complex value ever
# reached excel_round directly it would raise decimal.InvalidOperation
# (an ArithmeticError) instead, which is why that class is included too.
_FACTOR_ERRORS = (ArithmeticError, ValueError, TypeError)


def excel_round(value: float, digits: int) -> float:
    """Excel ROUND: half away from zero.

    Python's built-in round() is half-to-even and drifts from the workbook
    on .5 ties (e.g. -172.5 -> -172 instead of Excel's -173). Never use
    round() in this module.
    """
    quantum = Decimal(1).scaleb(-digits)
    return float(Decimal(repr(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _weighted_avg(
    values: Sequence[float], weights: Sequence[float], digits: int
) -> Optional[float]:
    total = sum(weights)
    if total == 0:
        return None
    return excel_round(
        sum(v * w for v, w in zip(values, weights)) / total, digits
    )


def _factor(calc: Callable[[], float]) -> float:
    """Wrap one adjustment-factor computation: arithmetic error -> 1.0."""
    try:
        return calc()
    except _FACTOR_ERRORS:
        return 1.0


def _confidence(n_total: int, min_points: int) -> str:
    if n_total == 0:
        return "No Data"
    if n_total < min_points:
        return "Insufficient"
    if n_total >= 10:
        return "HIGH"
    if n_total >= 6:
        return "MEDIUM"
    return "LOW"


def _echo_fields(quote: QuoteLine) -> dict:
    return dict(
        deal_id=quote.deal_id,
        item_name=quote.item_name,
        supplier_name=quote.supplier_name,
        quote_ref=quote.quote_ref,
        category=quote.category,
        quantity=quote.quantity,
        uom=quote.uom,
        currency=quote.currency,
        location=quote.location,
        requested_spec_score=quote.requested_spec_score,
        service_level=quote.service_level,
        requested_sla_score=quote.requested_sla_score,
        index_id=quote.index_id,
        quoted_unit_price=quote.quoted_unit_price,
        supplier_risk_penalty=quote.supplier_risk_penalty,
        contract_risk_penalty=quote.contract_risk_penalty,
        strategic_supplier_bonus=quote.strategic_supplier_bonus,
    )


def compute_benchmark(
    quote: QuoteLine,
    points: Sequence[BenchmarkPoint],
    location_index_table: Mapping[str, float],
    index_table: Mapping[str, float],
    settings: Optional[BenchmarkSettings] = None,
) -> BenchmarkResult:
    """Compute the adjusted benchmark and variance for one quote line.

    Pure function: same inputs -> identical BenchmarkResult. Fails closed
    (all computed fields None) below the evidence threshold.
    """
    settings = settings if settings is not None else BenchmarkSettings()

    # Match set M: exact item/UOM/currency, include=True; sources pooled.
    matched = [
        p for p in points
        if p.include
        and p.item_name == quote.item_name
        and p.uom == quote.uom
        and p.currency == quote.currency
    ]
    n_internal = sum(1 for p in matched if p.source == "internal")
    n_total = len(matched)
    total_weight = sum(p.source_weight for p in matched)
    confidence = _confidence(n_total, settings.min_data_points)

    echo = _echo_fields(quote)
    evidence = dict(
        matched_point_ids=[p.benchmark_point_id for p in matched],
        n_internal=n_internal,
        n_external=n_total - n_internal,
        n_total=n_total,
        total_weight=total_weight,
    )

    # Step 0 — fail-closed evidence gate (non-negotiable).
    if n_total < settings.min_data_points:
        logger.info(
            "benchmark gated for %s: %d matching points < min %d",
            quote.deal_id, n_total, settings.min_data_points,
        )
        return BenchmarkResult(**echo, **evidence, confidence=confidence, gated=True)

    # Step 1 — raw benchmark candidates.
    prices = [p.raw_unit_price for p in matched]
    weights = [p.source_weight for p in matched]
    simple = excel_round(sum(prices) / n_total, 2)
    weighted = _weighted_avg(prices, weights, 2)
    # Decision 1: TRUE median of the pooled internal+external matched set
    # (the prototype's internal-only, non-array MEDIAN(IF) is a defect).
    median = excel_round(_true_median(prices), 2)
    selected = {"simple": simple, "weighted": weighted, "median": median}[
        settings.method.value
    ]
    if selected is None:
        # Weighted method with zero total weight: no reliable benchmark.
        logger.warning(
            "benchmark gated for %s: total source weight is 0", quote.deal_id
        )
        # Evidence exists but carries zero weight, so a weighted benchmark
        # cannot be trusted — label it Insufficient rather than emit a
        # contradictory gated-but-HIGH record.
        return BenchmarkResult(
            **echo, **evidence, confidence="Insufficient", gated=True
        )

    # Step 2 — weighted profiles (same weights as the price average).
    # Decision 2: location profile uses the STATIC location_cost_index stored
    # on each row, not a live lookup of the row's location text.
    ref_quantity = _weighted_avg([p.historical_quantity for p in matched], weights, 2)
    avg_spec_score = _weighted_avg([p.specification_score for p in matched], weights, 2)
    avg_loc_index = _weighted_avg([p.location_cost_index for p in matched], weights, 4)
    avg_sla_score = _weighted_avg([p.sla_score for p in matched], weights, 2)
    avg_hist_index = _weighted_avg(
        [p.index_value_at_price_date for p in matched], weights, 2
    )

    # Step 3 — lookups. Decision 3: preserve the 1.0-style defaults for
    # parity, but record and log every substitution — never silent.
    fallbacks_used: list[str] = []
    try:
        target_loc_index = float(location_index_table[quote.location])
    except KeyError:
        target_loc_index = settings.default_location_index
        fallbacks_used.append("location_default")
        logger.warning(
            "location %r not in Location Index Table; using default %s (deal %s)",
            quote.location, target_loc_index, quote.deal_id,
        )
    try:
        current_index = float(index_table[quote.index_id])
    except KeyError:
        current_index = settings.default_current_index
        fallbacks_used.append("index_default")
        logger.warning(
            "index_id %r not in Index Table; using default %s (deal %s)",
            quote.index_id, current_index, quote.deal_id,
        )

    # Step 4 — the five adjustment factors, each individually rounded to 4dp
    # and each collapsing to 1.0 on any arithmetic error (prototype parity).
    volume = _factor(lambda: excel_round(_clamp(
        (quote.quantity / ref_quantity) ** (-settings.volume_elasticity),
        settings.volume_min_factor, settings.volume_max_factor), 4))
    spec = _factor(lambda: excel_round(_clamp(
        1 + (quote.requested_spec_score - avg_spec_score) * settings.spec_factor,
        settings.spec_min_factor, settings.spec_max_factor), 4))
    location = _factor(
        lambda: excel_round(target_loc_index / avg_loc_index, 4)  # no clamp
    )
    sla = _factor(lambda: excel_round(_clamp(
        1 + (quote.requested_sla_score - avg_sla_score) * settings.sla_factor,
        settings.sla_min_factor, settings.sla_max_factor), 4))
    inflation = _factor(
        lambda: excel_round(current_index / avg_hist_index, 4)  # no clamp
    )

    # Step 5 — final benchmark = rounded product of the ROUNDED factors.
    final_benchmark = excel_round(
        selected * volume * spec * location * sla * inflation, 2
    )
    combined_factor = excel_round(volume * spec * location * sla * inflation, 4)

    # Step 6 — totals and variance. Adders are added ONCE (not x quantity);
    # discount_rebate is a positive number that subtracts. Totals unrounded,
    # exactly like the prototype.
    adders = (
        quote.delivery_cost + quote.implementation_cost + quote.support_cost
        + quote.risk_premium - quote.discount_rebate
    )
    quoted_total = quote.quoted_unit_price * quote.quantity + adders
    benchmark_total = final_benchmark * quote.quantity + adders
    unit_variance_gbp = excel_round(quote.quoted_unit_price - final_benchmark, 2)
    unit_variance_pct = (
        excel_round(unit_variance_gbp / final_benchmark, 4)
        if final_benchmark != 0 else None
    )
    total_cost_gap = excel_round(quoted_total - benchmark_total, 0)

    return BenchmarkResult(
        **echo, **evidence,
        simple_benchmark=simple,
        weighted_benchmark=weighted,
        median_benchmark=median,
        selected_benchmark=selected,
        method_used=settings.method.value,
        ref_quantity=ref_quantity,
        avg_spec_score=avg_spec_score,
        avg_loc_index=avg_loc_index,
        avg_sla_score=avg_sla_score,
        avg_hist_index=avg_hist_index,
        target_loc_index=target_loc_index,
        current_index=current_index,
        fallbacks_used=fallbacks_used,
        volume_adjustment=volume,
        spec_adjustment=spec,
        location_adjustment=location,
        sla_adjustment=sla,
        inflation_adjustment=inflation,
        combined_factor=combined_factor,
        final_benchmark=final_benchmark,
        quoted_total=quoted_total,
        benchmark_total=benchmark_total,
        unit_variance_gbp=unit_variance_gbp,
        unit_variance_pct=unit_variance_pct,
        total_cost_gap=total_cost_gap,
        confidence=confidence,
        gated=False,
    )
