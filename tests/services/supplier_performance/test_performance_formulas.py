"""PPV and OTIF through the registry: the entry points a model and a report actually use."""
import pytest

from src.services.calculation_tools import _formulas, run_calculation

evaluate = _formulas().evaluate
REGISTRY = _formulas().REGISTRY

LINES = [{"po_unit_price": 10.0, "invoice_unit_price": 11.0, "quantity": 100.0, "currency": "GBP"}]
DELIVERIES = [{"promised_date": "2026-03-10", "received_date": "2026-03-10",
               "ordered_qty": 100.0, "received_qty": 100.0}]


def test_both_are_registered():
    assert "ppv.purchase_price_variance" in REGISTRY and "otif.delivery_performance" in REGISTRY


def test_ppv_runs_as_a_model_would_call_it():
    out = run_calculation("ppv.purchase_price_variance", {"lines": LINES})
    assert out["assessed"] is True, out
    value = out.get("value") or out.get("result")
    assert value["total_ppv"] == pytest.approx(100.0)


def test_otif_runs_with_json_dates():
    out = run_calculation("otif.delivery_performance", {
        "deliveries": DELIVERIES, "late_tolerance_days": 0, "in_full_tolerance": 0.0})
    assert out["assessed"] is True, out
    assert (out.get("value") or out.get("result"))["otif_rate"] == 1.0


def test_otif_with_nothing_assessable_is_a_refusal_not_a_zero():
    res = evaluate("otif.delivery_performance", {
        "deliveries": [{"promised_date": None, "received_date": None, "ordered_qty": None,
                        "received_qty": None}],
        "late_tolerance_days": 0, "in_full_tolerance": 0.0}, audit=False)
    assert res.unassessed


def test_otif_tolerances_are_required_not_defaulted():
    res = evaluate("otif.delivery_performance", {"deliveries": DELIVERIES}, audit=False)
    assert res.unassessed


def test_an_out_of_range_tolerance_is_refused():
    res = evaluate("otif.delivery_performance", {
        "deliveries": DELIVERIES, "late_tolerance_days": 0, "in_full_tolerance": 1.5}, audit=False)
    assert res.unassessed


def test_the_otif_score_feeds_the_existing_composite_score_and_a_gap_is_not_charged():
    """A scorecard is supplier.composite_score over these dimensions."""
    otif = evaluate("otif.delivery_performance", {
        "deliveries": DELIVERIES, "late_tolerance_days": 0, "in_full_tolerance": 0.0},
        audit=False).value
    batch = _formulas().evaluate_many(
        "supplier.composite_score",
        [{"scores": {"otif": otif["otif_score"], "price": 80.0}},      # A: both dimensions held
         {"scores": {"price": 60.0}}],                                # B: no OTIF data at all
        shared={"weights": {"otif": 0.5, "price": 0.5}}, audit=False)
    a, b = [r.value for r in batch]
    assert a == pytest.approx(90.0)          # (100 + 80) / 2
    assert b == pytest.approx(60.0)          # scored on price alone, NOT charged 0 for the missing OTIF
