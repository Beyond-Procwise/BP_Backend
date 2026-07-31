"""If a person keeps telling us this supplier bills in CAD, stop asking.

This is the concrete payoff of the verdict loop: the fourth "$" invoice from a Canadian
supplier resolves itself instead of stopping for review, because three people already
answered the question.
"""
from src.services.extraction_feedback.supplier_currency import (
    learned_currency, MIN_AGREEMENTS,
)


def _row(supplier="SUP-A", value="CAD", verdict="corrected"):
    return {"supplier_id": supplier, "corrected_value": value, "verdict": verdict}


def test_three_people_saying_CAD_settles_it():
    assert learned_currency([_row()] * MIN_AGREEMENTS) == {"SUP-A": "CAD"}


def test_two_is_not_enough():
    assert learned_currency([_row()] * (MIN_AGREEMENTS - 1)) == {}


def test_a_supplier_people_disagree_about_is_left_alone():
    # Genuinely ambiguous, or the supplier really does bill in both. Either way, guessing
    # is exactly what this whole feature exists to stop.
    rows = [_row(value="CAD")] * MIN_AGREEMENTS + [_row(value="USD")] * MIN_AGREEMENTS
    assert learned_currency(rows) == {}


def test_a_clear_majority_settles_it_even_with_one_dissenter():
    rows = [_row(value="CAD")] * 5 + [_row(value="USD")]
    assert learned_currency(rows) == {"SUP-A": "CAD"}


def test_only_corrections_teach_it():
    # A confirmation says the value we already had was right; it does not tell us what this
    # supplier's currency IS when we had nothing.
    assert learned_currency([_row(verdict="confirmed")] * MIN_AGREEMENTS) == {}


def test_suppliers_are_learned_independently():
    rows = [_row(supplier="SUP-A", value="CAD")] * MIN_AGREEMENTS + \
           [_row(supplier="SUP-B", value="SGD")] * MIN_AGREEMENTS
    assert learned_currency(rows) == {"SUP-A": "CAD", "SUP-B": "SGD"}


def test_a_blank_correction_teaches_nothing():
    assert learned_currency([_row(value=None)] * MIN_AGREEMENTS) == {}
