import pytest

from src.services.calculation_tools import _formulas, run_calculation
from src.services.demand import price_harmonisation as ph

LINES = [{"id": "A", "price": 10, "quantity": 1000}, {"id": "B", "price": 12, "quantity": 2000},
         {"id": "C", "price": 13, "quantity": 500}]


def test_the_reference_is_the_percentile_not_the_lowest_price():
    out = ph.harmonise(LINES, 25)
    assert out["reference_price"] == pytest.approx(11.0)
    assert [l["saving"] for l in out["lines"]] == [0.0, 2000.0, 1000.0]
    assert out["total_saving"] == pytest.approx(3000.0)


def test_a_purchase_below_the_reference_saves_nothing_and_is_never_negative():
    assert all(l["saving"] >= 0 for l in ph.harmonise(LINES, 90)["lines"])


def test_percentile_zero_is_the_lowest_and_a_hundred_the_highest():
    assert ph.harmonise(LINES, 0)["reference_price"] == 10.0
    assert ph.harmonise(LINES, 100)["reference_price"] == 13.0
    assert ph.harmonise(LINES, 100)["total_saving"] == 0.0


@pytest.mark.parametrize("lines,pct,words", [
    ([], 25, "no purchase lines"),
    (LINES[:2], 25, "at least 3"),
    ([*LINES[:2], {"id": "C", "price": 0, "quantity": 5}], 25, "greater than zero"),
    ([*LINES[:2], {"id": "C", "price": 13, "quantity": -1}], 25, "greater than zero"),
    ([*LINES[:2], {"id": "C", "price": "x", "quantity": 1}], 25, "not a number"),
    (LINES, 101, "between 0 and 100"),
    (LINES, "x", "not a number"),
])
def test_bad_inputs_are_refused_with_the_reason(lines, pct, words):
    with pytest.raises(ph.HarmonisationError, match=words):
        ph.harmonise(lines, pct)


def test_the_minimum_number_of_prices_can_be_set():
    assert ph.harmonise(LINES[:2], 25, min_observations=2)["reference_price"] == pytest.approx(10.5)


def test_registered_and_runs_through_run_calculation():
    assert "demand.price_harmonisation" in _formulas().REGISTRY
    out = run_calculation("demand.price_harmonisation", {"lines": LINES, "percentile": 25})
    assert out["assessed"] is True and out["value"]["total_saving"] == pytest.approx(3000.0)


def test_through_run_calculation_a_refusal_says_why():
    out = run_calculation("demand.price_harmonisation", {"lines": LINES[:1], "percentile": 25})
    assert out["assessed"] is False and "at least 3" in out["why"]
