# tests/test_benchmark_parity.py
"""Parity suite: the engine must reproduce the Excel prototype's golden
outputs exactly (default settings, method="weighted").

Fixture: tests/fixtures/benchmark/golden.json — exported once from
'Benchmark Calculations.xlsx' by scripts/export_benchmark_fixtures.py.
"""
import json
from pathlib import Path
from statistics import median

import pytest

from services.benchmark.engine import compute_benchmark, excel_round
from services.benchmark.models import (
    BenchmarkPoint,
    BenchmarkSettings,
    QuoteLine,
)

FIXTURE = Path(__file__).parent / "fixtures" / "benchmark" / "golden.json"


@pytest.fixture(scope="module")
def golden():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def points(golden):
    return [BenchmarkPoint(**p) for p in golden["points"]]


def _run(golden, points, deal_id, settings=None, **quote_overrides):
    raw = next(q for q in golden["quotes"] if q["deal_id"] == deal_id)
    quote = QuoteLine(**{**raw, **quote_overrides})
    return compute_benchmark(
        quote, points, golden["location_index_table"], golden["index_table"],
        settings or BenchmarkSettings(),
    )


# Every intermediate in the D001 trace, mapped result-field -> fixture key.
D001_TRACE = [
    "n_internal", "n_external", "n_total", "total_weight",
    "simple_benchmark", "weighted_benchmark", "selected_benchmark",
    "ref_quantity", "avg_spec_score", "avg_loc_index", "avg_sla_score",
    "target_loc_index", "current_index", "avg_hist_index",
    "volume_adjustment", "spec_adjustment", "location_adjustment",
    "sla_adjustment", "inflation_adjustment", "combined_factor",
    "final_benchmark", "quoted_total", "benchmark_total",
    "unit_variance_gbp", "unit_variance_pct", "total_cost_gap",
]


def test_d001_full_intermediate_trace(golden, points):
    result = _run(golden, points, "D001")
    expected = golden["expected"]["D001"]
    for field in D001_TRACE:
        assert getattr(result, field) == pytest.approx(
            float(expected[field]), abs=1e-9
        ), field
    assert result.confidence == "HIGH"
    assert result.gated is False
    assert result.method_used == "weighted"
    assert result.fallbacks_used == []
    assert len(result.matched_point_ids) == result.n_total == 12
    assert result.n_internal == 7 and result.n_external == 5


FINAL_FIELDS = [
    "n_total", "selected_benchmark", "final_benchmark",
    "unit_variance_gbp", "unit_variance_pct", "total_cost_gap",
]


@pytest.mark.parametrize("deal_id", ["D002", "D003", "D004", "D005"])
def test_remaining_deals_final_outputs(golden, points, deal_id):
    result = _run(golden, points, deal_id)
    expected = golden["expected"][deal_id]
    for field in FINAL_FIELDS:
        assert getattr(result, field) == pytest.approx(
            float(expected[field]), abs=1e-9
        ), field
    assert result.confidence == expected["confidence"]
    assert result.gated is False


# ---------------------------------------------------------------- Decision 1
@pytest.mark.parametrize("deal_id", ["D001", "D002", "D003", "D004", "D005"])
def test_median_is_true_pooled_median_and_nonzero(golden, points, deal_id):
    """The workbook's median column shows 0 for D002-D005 (non-array
    MEDIAN(IF) reading internal data only). Ours must be the true median of
    the pooled matched set — never 0 when matches exist."""
    result = _run(golden, points, deal_id, settings=BenchmarkSettings(method="median"))
    matched_prices = [
        p.raw_unit_price for p in points
        if p.include
        and p.item_name == result.item_name
        and p.uom == result.uom
        and p.currency == result.currency
    ]
    assert result.median_benchmark == excel_round(median(matched_prices), 2)
    assert result.median_benchmark != 0
    assert result.selected_benchmark == result.median_benchmark


# ------------------------------------------------------------------- gate
def test_gate_no_matches_all_outputs_none(golden, points):
    result = _run(golden, points, "D001", item_name="No Such Item")
    assert result.gated is True
    assert result.n_total == 0
    assert result.confidence == "No Data"
    for field in D001_TRACE[4:] + ["median_benchmark", "method_used"]:
        assert getattr(result, field) is None, field


def test_gate_below_min_points_all_outputs_none(golden, points):
    laptop = [
        p for p in points
        if p.item_name == "Laptop" and p.uom == "Each"
        and p.currency == "GBP" and p.include
    ][:2]
    result = _run(golden, laptop, "D001")
    assert result.gated is True
    assert result.n_total == 2
    assert result.confidence == "Insufficient"
    assert result.final_benchmark is None
    assert result.total_cost_gap is None


# --------------------------------------------------------------- Decision 3
def test_missing_location_computes_with_default_and_records_fallback(golden, points):
    # D003's real location is London (1.07); an unknown location must fall
    # back to 1.0, still compute, and leave an audit trail.
    result = _run(golden, points, "D003", location="Nowhereville")
    assert result.fallbacks_used == ["location_default"]
    assert result.target_loc_index == 1.0
    assert result.location_adjustment == excel_round(1.0 / result.avg_loc_index, 4)
    assert result.final_benchmark is not None


def test_missing_index_id_computes_with_default_and_records_fallback(golden, points):
    result = _run(golden, points, "D001", index_id="NO-SUCH-INDEX")
    assert result.fallbacks_used == ["index_default"]
    assert result.current_index == 1.0
    assert result.inflation_adjustment == excel_round(1.0 / result.avg_hist_index, 4)
    assert result.final_benchmark is not None


def test_missing_both_lookups_records_both_fallbacks(golden, points):
    result = _run(
        golden, points, "D001", location="Nowhereville", index_id="NO-SUCH-INDEX"
    )
    assert result.fallbacks_used == ["location_default", "index_default"]
