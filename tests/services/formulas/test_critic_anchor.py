"""Anchor and inflation maths, pinned.

The canonical false positive from the spec's Appendix A is the acceptance
vector: two contracted 4% uplifts against ~3.8% CPI must read as inflation,
not opportunity, forever.
"""
import pytest

from src.services.formulas import ensure_registered, evaluate
from src.services.formulas.definitions import critic  # noqa: F401


@pytest.fixture(autouse=True)
def _registered():
    ensure_registered()


def test_annualised_rate_of_two_four_percent_uplifts():
    # 0.041 -> 0.0447 over 4 years is ~2.18% a year.
    out = evaluate("critic.annualised_rate",
                   {"anchor_value": 0.041, "current_value": 0.0447, "years": 4.0})
    assert out.value == pytest.approx(2.18, abs=0.02)


def test_annualised_rate_refuses_a_zero_anchor():
    # A zero anchor is not a 0% rise, it is an unusable comparator.
    from src.services.formulas.unassessed import UNASSESSED
    out = evaluate("critic.annualised_rate",
                   {"anchor_value": 0.0, "current_value": 0.0447, "years": 4.0})
    assert out.value is UNASSESSED


def test_excess_over_index_is_zero_inside_the_band():
    # 4.0% against a 3.8% index, 2pp band -> inflation, not opportunity.
    out = evaluate("critic.excess_over_index",
                   {"annualised_pct": 4.0, "index_pct": 3.8, "band_pp": 2.0})
    assert out.value == 0.0


def test_excess_over_index_reports_only_the_excess():
    # 9.0% against 3.8% with a 2pp band -> the opportunity is 3.2pp, not 9.
    out = evaluate("critic.excess_over_index",
                   {"annualised_pct": 9.0, "index_pct": 3.8, "band_pp": 2.0})
    assert out.value == pytest.approx(3.2, abs=0.001)


def test_excess_over_index_is_unassessed_without_an_index():
    from src.services.formulas.unassessed import UNASSESSED
    out = evaluate("critic.excess_over_index",
                   {"annualised_pct": 9.0, "index_pct": None, "band_pp": 2.0})
    assert out.value is UNASSESSED


def test_unit_basis_mismatch_is_detected():
    out = evaluate("critic.unit_basis_match",
                   {"anchor_basis": "per_page", "current_basis": "per_month"})
    assert out.value is False


def test_unit_basis_match_is_case_and_space_tolerant():
    out = evaluate("critic.unit_basis_match",
                   {"anchor_basis": "Per Page", "current_basis": "per_page"})
    assert out.value is True


def test_fabricated_anchor_when_unit_price_equals_line_value_at_qty_one():
    # opportunity_miner_agent.py:5545 defaults a missing quantity to 1.0 and a
    # missing unit price to the whole line value. .min() then selects for
    # whichever row is most corrupted downward.
    out = evaluate("critic.fabricated_anchor",
                   {"unit_price": 4540.26, "line_value": 4540.26, "quantity": 1.0})
    assert out.value is True


def test_a_genuine_single_unit_line_is_not_fabricated():
    # Real qty-1 lines exist. The signal is the coincidence, so a line whose
    # value does not equal its unit price is clean even at quantity 1.
    out = evaluate("critic.fabricated_anchor",
                   {"unit_price": 100.0, "line_value": 250.0, "quantity": 1.0})
    assert out.value is False


def test_anchor_age_in_days():
    from datetime import date
    out = evaluate("critic.anchor_age_days",
                   {"anchor_date": date(2022, 3, 1), "current_date": date(2026, 3, 1)})
    assert out.value == 1461
