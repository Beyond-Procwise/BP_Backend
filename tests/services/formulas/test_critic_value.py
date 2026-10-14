"""Value, scope and confidence maths for the critic.

The rule that matters most here: a claim inherits the WEAKEST confidence of
the evidence it depends on, and nothing can upgrade evidence.
"""
import pytest

from src.services.formulas import ensure_registered, evaluate
from src.services.formulas.definitions import critic  # noqa: F401
from src.services.formulas.unassessed import UNASSESSED


@pytest.fixture(autouse=True)
def _registered():
    ensure_registered()


def test_normalise_unit_rate():
    out = evaluate("critic.normalise_unit_rate",
                   {"total_value": 500.0, "quantity": 10.0})
    assert out.value == 50.0


def test_normalise_unit_rate_refuses_zero_quantity():
    out = evaluate("critic.normalise_unit_rate",
                   {"total_value": 500.0, "quantity": 0.0})
    assert out.value is UNASSESSED


def test_volume_delta_is_a_fraction_of_the_anchor():
    out = evaluate("critic.volume_delta",
                   {"anchor_qty": 100.0, "current_qty": 130.0})
    assert out.value == pytest.approx(0.30, abs=0.001)


def test_addressable_value_subtracts_every_haircut():
    out = evaluate("critic.addressable_value",
                   {"detector_proposed": 48000.0,
                    "haircuts": [{"reason": "inflation", "amount": 30000.0},
                                 {"reason": "friction", "amount": 8000.0}]})
    assert out.value == 10000.0


def test_addressable_value_never_goes_negative():
    # A finding haircut below zero is worth nothing, not worth minus something.
    out = evaluate("critic.addressable_value",
                   {"detector_proposed": 1000.0,
                    "haircuts": [{"reason": "inflation", "amount": 5000.0}]})
    assert out.value == 0.0


def test_addressable_value_can_never_exceed_what_the_detector_proposed():
    # A negative haircut would raise the value. The prompt forbids VALID with a
    # value above the detector's, so the formula cannot produce one.
    out = evaluate("critic.addressable_value",
                   {"detector_proposed": 1000.0,
                    "haircuts": [{"reason": "friction", "amount": -5000.0}]})
    assert out.value == 1000.0


def test_relative_gap_reports_proportion():
    out = evaluate("critic.relative_gap", {"gap_value": 1200.0, "base_value": 3000.0})
    assert out.value == pytest.approx(0.40, abs=0.001)


def test_relative_gap_on_a_negligible_base_is_unassessed():
    # Live, one supplier's +345,261% was GBP 26.72 the year before.
    out = evaluate("critic.relative_gap", {"gap_value": 1200.0, "base_value": 0.0})
    assert out.value is UNASSESSED


def test_friction_haircut():
    out = evaluate("critic.friction_haircut",
                   {"gross_value": 10000.0, "friction_pct": 20.0})
    assert out.value == 2000.0


def test_min_confidence_takes_the_weakest():
    out = evaluate("critic.min_confidence",
                   {"confidences": ["CORROBORATED", "ASSERTED"]})
    assert out.value == "ASSERTED"


def test_any_unassessed_input_forces_unassessed():
    out = evaluate("critic.min_confidence",
                   {"confidences": ["CORROBORATED", "UNASSESSED", "CORROBORATED"]})
    assert out.value == "UNASSESSED"


def test_no_evidence_at_all_is_unassessed_not_corroborated():
    out = evaluate("critic.min_confidence", {"confidences": []})
    assert out.value == "UNASSESSED"
