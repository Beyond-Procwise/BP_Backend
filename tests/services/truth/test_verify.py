import pytest

from src.services.truth.verify import verify_field


def test_a_value_present_verbatim_is_verified():
    v = verify_field("invoice_id", "INV-2024-001", "Invoice No: INV-2024-001\nDate: 01/02/2024")
    assert v.outcome == "verified"


def test_a_value_absent_from_the_source_is_unsupported():
    v = verify_field("invoice_id", "INV-9999", "Invoice No: INV-2024-001")
    assert v.outcome == "unsupported"


def test_no_source_text_is_unverifiable_never_verified():
    for empty in (None, "", "   "):
        v = verify_field("invoice_id", "INV-2024-001", empty)
        assert v.outcome == "unverifiable", empty


def test_a_number_inside_a_longer_number_is_not_verified():
    # Review Focus 1. Substring matching calls this verified; it is not.
    v = verify_field("invoice_amount", 1234.56, "Total due 91234.567 after adjustment")
    assert v.outcome == "unsupported"


def test_european_decimal_notation_verifies():
    # Review Focus 2.
    v = verify_field("invoice_amount", 1234.56, "Gesamtbetrag: 1.234,56 EUR")
    assert v.outcome == "verified"


def test_thousand_separators_and_currency_symbols_verify():
    v = verify_field("invoice_amount", 2400.0, "Subtotal: £2,400.00")
    assert v.outcome == "verified"


def test_zero_is_a_value_and_is_checked():
    # Review Focus 3.
    present = verify_field("tax_amount", 0, "VAT (0%): 0.00")
    assert present.outcome == "verified"
    absent = verify_field("tax_amount", 0, "VAT (20%): 480.00")
    assert absent.outcome == "unsupported"


def test_a_date_in_another_rendering_verifies():
    # Review Focus 4.
    v = verify_field("invoice_date", "2024-03-15", "Date: 15/03/2024")
    assert v.outcome == "verified"


def test_the_verdict_says_which_rule_decided_it():
    v = verify_field("invoice_id", "INV-2024-001", "Invoice No: INV-2024-001")
    assert v.rule and isinstance(v.rule, str)
    assert v.span == "INV-2024-001"
