"""Model contracts for the benchmark pricing engine."""
import pytest
from pydantic import ValidationError

from services.benchmark.models import (
    BenchmarkMethod,
    BenchmarkPoint,
    BenchmarkResult,
    BenchmarkSettings,
    QuoteLine,
)


def test_settings_defaults_match_spec():
    s = BenchmarkSettings()
    assert s.min_data_points == 3
    assert s.method is BenchmarkMethod.WEIGHTED
    assert s.volume_elasticity == 0.06
    assert (s.volume_min_factor, s.volume_max_factor) == (0.85, 1.15)
    assert s.spec_factor == 0.03
    assert (s.spec_min_factor, s.spec_max_factor) == (0.90, 1.20)
    assert s.sla_factor == 0.025
    assert (s.sla_min_factor, s.sla_max_factor) == (0.90, 1.25)
    assert s.default_location_index == 1.0
    assert s.default_current_index == 1.0


@pytest.mark.parametrize("raw,expected", [
    ("Weighted Average", BenchmarkMethod.WEIGHTED),
    ("Simple Average", BenchmarkMethod.SIMPLE),
    ("Median", BenchmarkMethod.MEDIAN),
    ("weighted", BenchmarkMethod.WEIGHTED),
    ("median", BenchmarkMethod.MEDIAN),
])
def test_settings_method_accepts_excel_spellings(raw, expected):
    assert BenchmarkSettings(method=raw).method is expected


def test_settings_rejects_unknown_method():
    with pytest.raises(ValidationError):
        BenchmarkSettings(method="mode")


def _point(**overrides):
    base = dict(
        benchmark_point_id="INT0001", source="internal",
        item_name="Laptop", uom="Each", currency="GBP", include="Yes",
        raw_unit_price=780.0, source_weight=0.8, specification_score=6.0,
        location_cost_index=1.0, sla_score=5.0, historical_quantity=50.0,
        index_value_at_price_date=104.0,
    )
    base.update(overrides)
    return BenchmarkPoint(**base)


def test_point_include_coerces_yes_no_strings():
    assert _point(include="Yes").include is True
    assert _point(include="No").include is False
    assert _point(include=True).include is True


def test_point_rejects_unknown_source():
    with pytest.raises(ValidationError):
        _point(source="market")


def test_quote_line_adders_default_to_zero_and_info_fields_to_none():
    q = QuoteLine(
        deal_id="D001", item_name="Laptop", quantity=100, uom="Each",
        currency="GBP", location="UK", requested_spec_score=7,
        requested_sla_score=5, index_id="IT-HARDWARE", quoted_unit_price=950,
    )
    assert q.delivery_cost == 0.0 and q.discount_rebate == 0.0
    assert q.supplier_risk_penalty is None


def test_result_gated_shape_allows_all_none():
    r = BenchmarkResult(
        deal_id="D001", item_name="Laptop", supplier_name="", quote_ref="",
        category="", quantity=100.0, uom="Each", currency="GBP", location="UK",
        requested_spec_score=7.0, service_level="", requested_sla_score=5.0,
        index_id="IT-HARDWARE", quoted_unit_price=950.0,
        matched_point_ids=[], n_internal=0, n_external=0, n_total=0,
        total_weight=0.0, confidence="No Data", gated=True,
    )
    assert r.final_benchmark is None
    assert r.fallbacks_used == []
