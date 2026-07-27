"""Unit tests for the benchmark engine: gate, guards, clamps, fallbacks,
median correctness, confidence bands, Excel-style rounding."""
import pytest

from services.benchmark.engine import compute_benchmark, excel_round
from services.benchmark.models import (
    BenchmarkMethod,
    BenchmarkPoint,
    BenchmarkSettings,
    QuoteLine,
)


def _quote(**overrides):
    base = dict(
        deal_id="T1", item_name="Widget", quantity=10.0, uom="Each",
        currency="GBP", location="UK", requested_spec_score=5.0,
        requested_sla_score=5.0, index_id="IDX", quoted_unit_price=25.0,
        supplier_risk_penalty=7.0,
    )
    base.update(overrides)
    return QuoteLine(**base)


def _point(pid, price, weight=1.0, **overrides):
    base = dict(
        benchmark_point_id=pid, source="internal", item_name="Widget",
        uom="Each", currency="GBP", include=True, raw_unit_price=price,
        source_weight=weight, specification_score=5.0,
        location_cost_index=1.0, sla_score=5.0, historical_quantity=10.0,
        index_value_at_price_date=100.0,
    )
    base.update(overrides)
    return BenchmarkPoint(**base)


LOC = {"UK": 1.0, "London": 1.07}
IDX = {"IDX": 100.0}
POINTS = [_point("P1", 10.0), _point("P2", 20.0), _point("P3", 30.0, weight=2.0)]


# ---------------------------------------------------------------- rounding
def test_excel_round_is_half_away_from_zero_not_bankers():
    assert excel_round(-172.5, 0) == -173  # Python round() gives -172
    assert excel_round(2.5, 0) == 3        # Python round() gives 2
    assert excel_round(1.0159499, 4) == 1.0159


# ------------------------------------------------------------------- gate
def test_gate_below_min_points_returns_all_none_no_exception():
    result = compute_benchmark(_quote(), POINTS[:2], LOC, IDX)
    assert result.gated is True
    assert result.n_total == 2
    assert result.confidence == "Insufficient"
    for field in (
        "simple_benchmark", "weighted_benchmark", "median_benchmark",
        "selected_benchmark", "method_used", "ref_quantity", "avg_spec_score",
        "avg_loc_index", "avg_sla_score", "avg_hist_index", "target_loc_index",
        "current_index", "volume_adjustment", "spec_adjustment",
        "location_adjustment", "sla_adjustment", "inflation_adjustment",
        "combined_factor", "final_benchmark", "quoted_total",
        "benchmark_total", "unit_variance_gbp", "unit_variance_pct",
        "total_cost_gap",
    ):
        assert getattr(result, field) is None, field
    assert result.fallbacks_used == []


def test_gate_zero_matches_is_no_data():
    result = compute_benchmark(_quote(item_name="Nonexistent"), POINTS, LOC, IDX)
    assert result.gated is True
    assert result.n_total == 0
    assert result.confidence == "No Data"


def test_exclude_and_mismatched_rows_do_not_count():
    points = POINTS + [
        _point("P4", 99.0, include=False),
        _point("P5", 99.0, uom="Box"),
        _point("P6", 99.0, currency="USD"),
        _point("P7", 99.0, item_name="Gadget"),
    ]
    result = compute_benchmark(_quote(), points, LOC, IDX)
    assert result.n_total == 3
    assert result.matched_point_ids == ["P1", "P2", "P3"]


# ------------------------------------------------------------- candidates
def test_candidates_simple_weighted_median():
    result = compute_benchmark(_quote(), POINTS, LOC, IDX)
    assert result.simple_benchmark == 20.0            # (10+20+30)/3
    assert result.weighted_benchmark == 22.5          # (10+20+60)/4
    assert result.median_benchmark == 20.0            # true pooled median
    assert result.selected_benchmark == 22.5          # default method=weighted
    assert result.method_used == "weighted"
    assert result.total_weight == 4.0


def test_median_method_uses_true_pooled_median_never_zero():
    # Decision 1: pooled internal+external, correct median, non-zero.
    points = POINTS + [_point("E1", 40.0, source="external")]
    settings = BenchmarkSettings(method="Median")  # Excel spelling
    result = compute_benchmark(_quote(), points, LOC, IDX, settings)
    assert result.median_benchmark == 25.0  # median of 10,20,30,40
    assert result.median_benchmark != 0
    assert result.selected_benchmark == 25.0
    assert result.method_used == "median"


def test_zero_total_weight_simple_method_documents_partial_audit_record():
    # All source_weight=0.0 is out-of-range for real data (prototype uses
    # 0.5-1.5) but must not raise: _weighted_avg returns None for every
    # weighted profile, so the weighted candidate is None too. With
    # method="simple" the simple/median candidates still compute (they
    # don't depend on weight), so the engine produces a real, non-gated
    # result with all five factors neutral and the None-driven profiles
    # left None — a partial-but-honest audit record (Excel IFERROR parity).
    points = [
        _point("W1", 10.0, weight=0.0),
        _point("W2", 20.0, weight=0.0),
        _point("W3", 30.0, weight=0.0),
    ]
    settings = BenchmarkSettings(method="simple")
    result = compute_benchmark(_quote(), points, LOC, IDX, settings)
    assert result.gated is False
    assert result.total_weight == 0.0
    assert result.ref_quantity is None
    assert result.avg_spec_score is None
    for factor in (
        result.volume_adjustment, result.spec_adjustment,
        result.location_adjustment, result.sla_adjustment,
        result.inflation_adjustment,
    ):
        assert factor == 1.0
    assert result.final_benchmark == result.simple_benchmark

    # Same zero-weight points but the default (weighted) method: the
    # weighted candidate is None -> selected is None -> gate stays closed.
    gated_result = compute_benchmark(_quote(), points, LOC, IDX)
    assert gated_result.gated is True
    assert gated_result.confidence == "Insufficient"


# ---------------------------------------------------------------- factors
def test_volume_clamps_at_min_factor():
    # qty >> ref_quantity drives the factor below 0.85 -> clamped.
    result = compute_benchmark(_quote(quantity=1e9), POINTS, LOC, IDX)
    assert result.volume_adjustment == 0.85


def test_volume_clamps_at_max_factor():
    result = compute_benchmark(_quote(quantity=1e-9), POINTS, LOC, IDX)
    assert result.volume_adjustment == 1.15


def test_spec_and_sla_factors_move_with_requested_scores():
    result = compute_benchmark(
        _quote(requested_spec_score=7.0, requested_sla_score=3.0), POINTS, LOC, IDX
    )
    assert result.spec_adjustment == 1.06   # 1 + (7-5)*0.03
    assert result.sla_adjustment == 0.95    # 1 + (3-5)*0.025


def test_zero_ref_quantity_yields_neutral_volume_factor():
    points = [_point(f"Z{i}", 10.0, historical_quantity=0.0) for i in range(3)]
    result = compute_benchmark(_quote(), points, LOC, IDX)
    assert result.volume_adjustment == 1.0  # divide-by-zero -> neutral 1.0


def test_zero_avg_loc_index_yields_neutral_location_factor():
    points = [_point(f"Z{i}", 10.0, location_cost_index=0.0) for i in range(3)]
    result = compute_benchmark(_quote(), points, LOC, IDX)
    assert result.location_adjustment == 1.0


def test_zero_hist_index_yields_neutral_inflation_factor():
    points = [_point(f"Z{i}", 10.0, index_value_at_price_date=0.0) for i in range(3)]
    result = compute_benchmark(_quote(), points, LOC, IDX)
    assert result.inflation_adjustment == 1.0


def test_negative_quantity_complex_power_yields_neutral_volume_factor():
    # base = quantity/ref_quantity = -10/10 = -1; (-1) ** (-0.06) is a
    # negative base raised to a fractional exponent, which Python returns
    # as complex rather than raising. _clamp's min()/max() then raise
    # TypeError comparing complex to float, caught by _FACTOR_ERRORS ->
    # neutral 1.0 (prototype IFERROR parity). Must not raise.
    result = compute_benchmark(_quote(quantity=-10.0), POINTS, LOC, IDX)
    assert result.volume_adjustment == 1.0
    assert result.final_benchmark is not None


def test_negative_ref_quantity_complex_power_yields_neutral_volume_factor():
    # Same complex-base path driven from the other side: ref_quantity
    # negative (historical_quantity=-5.0), quantity positive (default
    # 10.0) -> base = 10/-5 = -2, still negative -> complex -> neutral 1.0.
    points = [_point(f"N{i}", 10.0, historical_quantity=-5.0) for i in range(3)]
    result = compute_benchmark(_quote(), points, LOC, IDX)
    assert result.volume_adjustment == 1.0
    assert result.final_benchmark is not None


# --------------------------------------------------------------- fallbacks
def test_missing_location_uses_default_and_is_recorded():
    result = compute_benchmark(_quote(location="Atlantis"), POINTS, LOC, IDX)
    assert result.target_loc_index == 1.0
    assert "location_default" in result.fallbacks_used
    assert result.final_benchmark is not None  # still computes


def test_missing_index_id_uses_default_and_is_recorded():
    result = compute_benchmark(_quote(index_id="NOPE"), POINTS, LOC, IDX)
    assert result.current_index == 1.0
    assert "index_default" in result.fallbacks_used
    assert result.inflation_adjustment == 0.01  # 1.0 / 100.0
    assert result.final_benchmark is not None


def test_no_fallbacks_recorded_when_lookups_hit():
    result = compute_benchmark(_quote(), POINTS, LOC, IDX)
    assert result.fallbacks_used == []


# -------------------------------------------------------------- outputs
def test_totals_add_adders_once_not_multiplied_by_quantity():
    quote = _quote(
        quoted_unit_price=25.0, quantity=10.0, delivery_cost=100.0,
        implementation_cost=50.0, support_cost=30.0, risk_premium=20.0,
        discount_rebate=40.0,
    )
    result = compute_benchmark(quote, POINTS, LOC, IDX)
    assert result.quoted_total == 25.0 * 10 + 100 + 50 + 30 + 20 - 40  # 410
    assert result.benchmark_total == pytest.approx(
        result.final_benchmark * 10 + 160
    )


def test_variance_pct_is_fraction_of_final_benchmark():
    result = compute_benchmark(_quote(), POINTS, LOC, IDX)
    assert result.unit_variance_gbp == excel_round(
        25.0 - result.final_benchmark, 2
    )
    assert result.unit_variance_pct == excel_round(
        result.unit_variance_gbp / result.final_benchmark, 4
    )


def test_zero_final_benchmark_guards_variance_pct_to_none():
    # raw_unit_price=0.0 on every matched point drives simple/weighted
    # benchmark to 0.0 and every adjustment factor stays neutral (1.0), so
    # final_benchmark is exactly 0 -> unit_variance_pct must guard the
    # divide-by-zero and be None, not raise or produce inf/nan.
    points = [_point(f"Z{i}", 0.0) for i in range(3)]
    result = compute_benchmark(_quote(), points, LOC, IDX)
    assert result.final_benchmark == 0.0
    assert result.unit_variance_pct is None
    assert result.unit_variance_gbp == 25.0  # 25.0 (quote) - 0.0 (benchmark)


# ------------------------------------------------------------- confidence
@pytest.mark.parametrize("n,expected", [
    (3, "LOW"), (5, "LOW"), (6, "MEDIUM"), (9, "MEDIUM"), (10, "HIGH"),
])
def test_confidence_bands(n, expected):
    points = [_point(f"C{i}", 10.0 + i) for i in range(n)]
    result = compute_benchmark(_quote(), points, LOC, IDX)
    assert result.confidence == expected
    assert result.gated is False


# ----------------------------------------------------- info-only passthrough
def test_info_only_scores_pass_through_and_do_not_affect_result():
    baseline = compute_benchmark(_quote(supplier_risk_penalty=None), POINTS, LOC, IDX)
    scored = compute_benchmark(
        _quote(supplier_risk_penalty=90.0, contract_risk_penalty=80.0,
               strategic_supplier_bonus=70.0),
        POINTS, LOC, IDX,
    )
    assert scored.supplier_risk_penalty == 90.0
    assert scored.contract_risk_penalty == 80.0
    assert scored.strategic_supplier_bonus == 70.0
    assert scored.final_benchmark == baseline.final_benchmark
    assert scored.total_cost_gap == baseline.total_cost_gap


# ------------------------------------------------------------- determinism
def test_same_inputs_identical_outputs():
    a = compute_benchmark(_quote(), POINTS, LOC, IDX)
    b = compute_benchmark(_quote(), POINTS, LOC, IDX)
    assert a == b


# ------------------------------------------------- missing historical qty
def test_missing_historical_quantity_is_excluded_not_zeroed():
    """A services line with no quantity must not pull the reference quantity
    toward zero — that would fake a volume premium out of missing data."""
    from services.benchmark.engine import compute_benchmark
    from services.benchmark.models import BenchmarkPoint, QuoteLine

    def point(pid, qty):
        return BenchmarkPoint(
            benchmark_point_id=pid, source="internal", item_name="widget",
            uom="each", currency="GBP", include=True, raw_unit_price=100.0,
            source_weight=1.0, specification_score=5.0, location_cost_index=1.0,
            sla_score=5.0, historical_quantity=qty,
            index_value_at_price_date=1.0,
        )

    quote = QuoteLine(
        deal_id="D", item_name="widget", quantity=100, uom="each",
        currency="GBP", location="UK", requested_spec_score=5,
        requested_sla_score=5, index_id="", quoted_unit_price=100.0,
    )
    known = [point("a", 100.0), point("b", 100.0), point("c", 100.0)]
    with_gap = known + [point("d", None)]

    assert compute_benchmark(quote, known, {}, {}).ref_quantity == 100.0
    assert compute_benchmark(quote, with_gap, {}, {}).ref_quantity == 100.0


def test_missing_quantity_drops_its_weight_not_just_its_value():
    """Weight preservation, made visible with non-uniform weights: a
    None-quantity point must remove its weight from the denominator too,
    not just be skipped from the numerator as if it carried zero weight.

    Points: qty=100 @ weight 1.0, qty=200 @ weight 3.0, qty=None @ weight 5.0.
    Correct (weight dropped): (100*1 + 200*3) / (1+3) = 175.0.
    Wrong (weight kept, e.g. treated as a zero contribution over all 9
    weight): (100*1 + 200*3 + 0*5) / 9 = 77.78 — a different, wrong number.
    """
    from services.benchmark.engine import compute_benchmark
    from services.benchmark.models import BenchmarkPoint, QuoteLine

    def point(pid, qty, weight):
        return BenchmarkPoint(
            benchmark_point_id=pid, source="internal", item_name="widget",
            uom="each", currency="GBP", include=True, raw_unit_price=100.0,
            source_weight=weight, specification_score=5.0,
            location_cost_index=1.0, sla_score=5.0, historical_quantity=qty,
            index_value_at_price_date=1.0,
        )

    quote = QuoteLine(
        deal_id="D", item_name="widget", quantity=100, uom="each",
        currency="GBP", location="UK", requested_spec_score=5,
        requested_sla_score=5, index_id="", quoted_unit_price=100.0,
    )
    points = [
        point("a", 100.0, 1.0),
        point("b", 200.0, 3.0),
        point("c", None, 5.0),
    ]
    assert compute_benchmark(quote, points, {}, {}).ref_quantity == 175.0


def test_all_missing_quantity_falls_back_to_neutral_volume_not_zero_or_crash():
    """When every matched point's quantity is unknown, ref_quantity has
    nothing to average and must be None — not a fabricated 0.0. The volume
    factor then divides quantity by None; that must fail SAFE to the
    existing neutral-factor contract (1.0), not raise out of
    compute_benchmark and not silently become some other number. The line
    still benchmarks on price alone: gated stays False and a
    final_benchmark is produced."""
    from services.benchmark.engine import compute_benchmark
    from services.benchmark.models import BenchmarkPoint, QuoteLine

    def point(pid):
        return BenchmarkPoint(
            benchmark_point_id=pid, source="internal", item_name="widget",
            uom="each", currency="GBP", include=True, raw_unit_price=100.0,
            source_weight=1.0, specification_score=5.0, location_cost_index=1.0,
            sla_score=5.0, historical_quantity=None,
            index_value_at_price_date=1.0,
        )

    quote = QuoteLine(
        deal_id="D", item_name="widget", quantity=100, uom="each",
        currency="GBP", location="UK", requested_spec_score=5,
        requested_sla_score=5, index_id="", quoted_unit_price=100.0,
    )
    points = [point("a"), point("b"), point("c")]
    result = compute_benchmark(quote, points, {}, {})

    assert result.ref_quantity is None
    assert result.volume_adjustment == 1.0
    assert result.gated is False
    assert result.final_benchmark is not None
