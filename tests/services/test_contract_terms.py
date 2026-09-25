"""A contract line says what it allows in its own words; the basis is read from them."""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest

from src.services.extraction.contract_terms import (
    classify_contract_lines,
    classify_term_basis,
    qualifier_for,
)


@pytest.mark.parametrize("desc, price, expected", [
    ("Installation — day rate", "1,485.00", "rate"),
    ("Access switch, per unit", 2730, "rate"),
    ("Core switching, not to exceed", "76,960", "cap"),
    ("Core switching — capped at", 76960, "cap"),
    ("Support hours (maximum)", 1200, "cap"),
    ("Freight — included", None, "included"),
    ("Delivery at no additional charge", "", "included"),
    ("Onboarding (FOC)", None, "included"),
    ("Service description only", None, None),       # no price, no inclusion wording: not guessed
    ("Up to 5 site visits", None, None),            # a cap with nothing to cap is not a term
])
def test_classify_term_basis(desc, price, expected):
    assert classify_term_basis(desc, price) == expected


def test_qualifier_text_counts():
    assert classify_term_basis("Core switching", "76960", "Not to exceed") == "cap"


def test_classify_contract_lines_sets_basis_qualifier_and_zero_price_for_included():
    rows = classify_contract_lines([
        {"item_description": "Installation day rate", "unit_price": "1485"},
        {"item_description": "Freight included", "unit_price": None},
        {"item_description": "Core switching NTE", "unit_price": "76960", "qualifier": "2026 cap"},
    ])
    assert [(r["term_basis"], r["qualifier"]) for r in rows] == [
        ("rate", None), ("included", "included"), ("cap", "2026 cap")]
    assert rows[1]["unit_price"] == 0


def test_qualifier_for():
    assert [qualifier_for(b) for b in ("rate", "cap", "included", None)] == [None, "price cap", "included", None]
