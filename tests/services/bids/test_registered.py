import pytest

from src.services.calculation_tools import _formulas, list_calculations, run_calculation

IN = {
    "line_master": [{"line_id": "L1", "description": "Bracket", "uom": "EA", "qty": 100}],
    "bids": [{"supplier_id": "A", "currency": "GBP", "payment_days": 60,
              "lines": [{"line_id": "L1", "unit_price": 10.0, "uom": "EA"}]},
             {"supplier_id": "B", "currency": "GBP", "payment_days": 60,
              "lines": [{"line_id": "L1", "unit_price": 9.0, "uom": "EA"}]}],
    "base_currency": "GBP", "fx_rates": {"GBP": 1.0},
}


def test_registered_and_discoverable():
    assert "bids.normalise" in _formulas().REGISTRY
    assert any(c["name"] == "bids.normalise" for c in list_calculations("bids")["calculations"])


def test_runs_through_run_calculation_with_defaults_for_what_is_left_out():
    out = run_calculation("bids.normalise", IN)
    assert out["assessed"] is True, out
    v = out["value"] if "value" in out else out["result"]
    assert v["best_single_supplier"]["supplier_id"] == "B" and v["cherry_pick_total"] == pytest.approx(900.0)


def test_a_bad_input_comes_back_unassessed_with_the_reason():
    out = run_calculation("bids.normalise", {**IN, "missing_line_policy": "fill_with_last_paid_price"})
    assert out["assessed"] is False and "last_paid_prices" in out["why"]
