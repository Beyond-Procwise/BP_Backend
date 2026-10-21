from src.services.calculation_tools import run_calculation

BAD = {
    "criteria": [{"name": "quality", "weight": 0.6, "scale": 5}],
    "scale_min": 0, "scale_max": 5,
    "scores": [{"supplier": "A", "criterion": "quality", "score": 4}],
}


def test_a_formula_that_declines_tells_the_caller_why():
    out = run_calculation("scoring.weighted_total", BAD)
    assert out["assessed"] is False
    assert "criterion_id" in out["why"], out


def test_the_reason_does_not_leak_into_the_next_call():
    run_calculation("scoring.weighted_total", BAD)
    other = run_calculation("deal.pct_change", {"new": 110, "base": 100})
    assert other["assessed"] is True
