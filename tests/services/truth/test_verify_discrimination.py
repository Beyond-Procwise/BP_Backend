"""Findings from the fresh review of this branch, each pinned as a test.

The review ran a negative control: substitute an arbitrary WRONG quantity into a
real document and ask whether it verifies. A fabricated quantity of 5 verified
in 95.5% of the 175 documents, because `_candidate_numbers` collects every
number on the page -- dates, page numbers, line numbers -- and asks only "is it
in the bag". 55% of all verified fields came from that rule, so the headline
accuracy was substantially inflated.

The answer is the spec's own principle turned on the checker: if a match carries
almost no evidence, the honest verdict is `unverifiable`, not `verified`.
"""
import pytest

from src.services.truth.verify import verify_field


# --- C1: a match that carries no evidence is not a verification ---------------

def test_a_bare_small_integer_is_not_discriminating():
    # The reviewer's control: quantity 5 against a document that merely contains
    # a 5 somewhere. Verifying this is how a fabricated quantity passed.
    v = verify_field("quantity", 5, "Invoice 2024-03-15, page 5 of 9")
    assert v.outcome == "unverifiable"
    assert v.rule == "low-discrimination"


def test_a_small_integer_written_with_decimals_is_discriminating():
    # "490.00" on the page is a real money figure, not a coincidental digit.
    v = verify_field("total_amount", 490.0, "Total: 490.00")
    assert v.outcome == "verified"


def test_a_value_with_a_fractional_part_is_discriminating():
    v = verify_field("line_total", 49.59, "Line total 49.59")
    assert v.outcome == "verified"


def test_a_large_number_is_discriminating_even_when_whole():
    v = verify_field("total_amount", 25000.0, "Contract value 25000")
    assert v.outcome == "verified"


def test_a_bare_small_integer_absent_from_the_page_is_still_unsupported():
    # Absence is still information; only a coincidental PRESENCE is worthless.
    v = verify_field("quantity", 5, "Quantity: 12 units at 3.00 each")
    assert v.outcome == "unsupported"


# --- C2: comma-grouped thousands and negatives --------------------------------

def test_comma_grouped_thousands_verify():
    # rfind(".") returns -1 when there is no dot, so "1,234" was read as 1.234.
    v = verify_field("total_amount", 1234.0, "Total: 1,234")
    assert v.outcome == "verified"


def test_millions_are_not_silently_dropped():
    # float("1.234.567") raises, and the exception was swallowed, so the number
    # never entered the bag and every match against it failed.
    v = verify_field("total_amount", 1234567.0, "Contract value 1,234,567")
    assert v.outcome == "verified"


def test_a_real_money_figure_with_grouping_verifies():
    # The reviewer's concrete case: total_amount 25000.0, source prints 25,000.
    v = verify_field("total_amount", 25000.0, "Grand Total: 25,000")
    assert v.outcome == "verified"


def test_a_negative_value_verifies():
    v = verify_field("line_total", -100.0, "Adjustment -100.00")
    assert v.outcome == "verified"


# --- I5: a currency code against a document that prints the symbol ------------

def test_a_currency_code_verifies_against_its_symbol():
    # 97 of 98 unsupported currency verdicts had the matching symbol on the page.
    assert verify_field("currency", "GBP", "Total: £2,400.00").outcome == "verified"
    assert verify_field("currency", "EUR", "Gesamt: €1.234,56").outcome == "verified"
    assert verify_field("currency", "USD", "Total: $99.00").outcome == "verified"


def test_a_currency_code_still_verifies_when_written_out():
    assert verify_field("currency", "GBP", "Total: GBP 2400.00").outcome == "verified"


def test_the_wrong_currency_is_still_unsupported():
    assert verify_field("currency", "JPY", "Total: £2,400.00").outcome == "unsupported"


# --- I7: dates the checker could not read -------------------------------------

@pytest.mark.parametrize("value,source", [
    ("2024-03-15 00:00:00", "Date: 15/03/2024"),   # datetime stringified
    ("15/03/2024", "Invoice date 2024-03-15"),      # already non-ISO
    ("15.03.2024", "Dated 15 March 2024"),          # European punctuation
])
def test_dates_in_other_shapes_still_verify(value, source):
    assert verify_field("invoice_date", value, source).outcome == "verified"


def test_an_unpadded_day_verifies():
    # "5 March 2024" -- %d renders "05" and never matched.
    assert verify_field("invoice_date", "2024-03-05", "Dated 5 March 2024").outcome == "verified"


# --- M14: source text that is non-whitespace but meaningless ------------------

def test_a_zero_width_source_is_unverifiable_not_unsupported():
    v = verify_field("invoice_id", "INV-1", "​")
    assert v.outcome == "unverifiable"


# --- M13: typographic apostrophes and ligatures -------------------------------

def test_a_typographic_apostrophe_matches_a_straight_one():
    v = verify_field("supplier_name", "O'Brien Ltd", "From: O’Brien Ltd")
    assert v.outcome == "verified"


# --- M16: a boolean is not a string to be hunted for --------------------------

def test_a_boolean_is_unverifiable_rather_than_unsupported():
    v = verify_field("is_paid", True, "Paid: yes")
    assert v.outcome == "unverifiable"
