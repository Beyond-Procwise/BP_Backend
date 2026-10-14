"""What a contract is worth, and in what currency, read from its own words.

Audited 2026-10-04: `total_contract_value` and `currency` are pattern-less (nine
and four `canonical_labels`, no `patterns`), and NULL on all 7 live contract rows
-- while **every one of those documents states a value**:

    The total estimated value of this Framework Agreement over its term is GBP 750,000.
    The total cost of the Services will be £25,000.
    The total charges for this Order Form are GBP 48,000, invoiced monthly in arrears
        at GBP 4,000 per month.

This is the fourth instance of the same structural hole (parties, signatories,
term, value) and the one that costs money: a contract whose value is NULL
contributes nothing to any spend or savings figure.

THE TRAP IN THIS ONE is the second number. "GBP 48,000, invoiced monthly ... at
GBP 4,000 per month" states a total and a rate in one sentence, and the real
Marketing Agreement states two instalments ("£10,000 will be paid at the signing
... £15,000 will be paid at completion") alongside its total. Taking the wrong
number is worse than taking none: it is a plausible figure that silently
misreports the contract. So the value must be the FIRST money token after a phrase
that actually says "total", inside one sentence.

A currency marker is REQUIRED. "The total charges are 48,000" yields nothing: a
bare number after the word "total" could be a headcount, and a money figure whose
currency nobody stated is the thing this product has been burned by before
(`project_spendiq_spend_canonical_figure`).

Offline and model-free -- the amount and currency parsers it reuses need no
optional dependency, which was checked before relying on them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.engineered.contract_value import (  # noqa: E402
    read_value, value_candidates,
)

#: The real Marketing Agreement's payment clause, as the parser renders it: a
#: total AND two instalments, which is exactly how the wrong number gets taken.
REAL = (
    "PAYMENT AND FEES\n\nThe Parties agree that the total cost of the Services will "
    "be £25,000.\n\nMore specifically, £10,000 will be paid at the signing of "
    "this Agreement, and £15,000 will be paid at completion.\n\nThe Marketer agrees "
    "to obtain consent from the Client prior to making the purchase if an expense is "
    "over £500.\n"
)


def test_the_real_contracts_total_is_read_and_not_an_instalment():
    v = read_value(REAL)
    assert v.value == "25000.00", v
    assert v.currency == "GBP", v


@pytest.mark.parametrize("sentence,amount,ccy", [
    ("The total estimated value of this Framework Agreement over its term is GBP 750,000.",
     "750000.00", "GBP"),
    ("The total charges for this Order Form are GBP 12,500.", "12500.00", "GBP"),
    ("The total cost of the Services will be £25,000.", "25000.00", "GBP"),
    ("The total consideration shall be EUR 40,000.", "40000.00", "EUR"),
    ("The total price is USD 100,000.00.", "100000.00", "USD"),
    ("The total fees are €18,500.", "18500.00", "EUR"),
    ("Total Contract Value: GBP 750,000", "750000.00", "GBP"),
    ("Contract Value: £25,000", "25000.00", "GBP"),
    ("Total Value: GBP 60,000", "60000.00", "GBP"),
    ("Maximum Contract Value: GBP 90,000", "90000.00", "GBP"),
    ("Not to Exceed: USD 100,000", "100000.00", "USD"),
])
def test_each_shape_is_read(sentence, amount, ccy):
    v = read_value(sentence)
    assert (v.value, v.currency) == (amount, ccy), sentence


def test_the_monthly_rate_in_the_same_sentence_is_not_the_value():
    """THE trap: a total and a rate in one sentence. 48,000 is the contract's
    worth; 4,000 is what it bills each month."""
    v = read_value("The total charges for this Order Form are GBP 48,000, invoiced "
                   "monthly in arrears at GBP 4,000 per month.")
    assert v.value == "48000.00", v


def test_a_currency_after_the_number_is_read():
    v = read_value("The total charges are 60,000 GBP.")
    assert (v.value, v.currency) == ("60000.00", "GBP"), v


def test_a_total_with_no_currency_yields_nothing():
    """A bare number after 'total' could be a headcount, and a money figure whose
    currency nobody stated is how a spend total silently becomes fiction."""
    assert read_value("The total charges for this Order Form are 48,000.").value is None


def test_instalments_alone_are_not_a_contract_value():
    text = ("More specifically, £10,000 will be paid at the signing of this "
            "Agreement, and £15,000 will be paid at completion.\n")
    assert read_value(text).value is None


def test_a_spending_threshold_is_not_a_contract_value():
    text = ("The Marketer agrees to obtain consent from the Client prior to making "
            "the purchase if an expense is over £500.\n")
    assert read_value(text).value is None


def test_a_monthly_rate_alone_is_not_a_contract_value():
    assert read_value("Invoiced monthly in arrears at GBP 5,000 per month.").value is None


def test_a_document_with_no_money_yields_nothing():
    v = read_value("FRAMEWORK AGREEMENT\nThis Agreement is governed by English law.\n")
    assert v.value is None and v.currency is None, v


def test_the_value_does_not_reach_into_the_next_sentence():
    """A window that crosses a full stop reads the next clause's number."""
    v = read_value("The total value is stated in Schedule 1. The deposit is GBP 5,000.")
    assert v.value is None, v


def test_a_label_wins_over_prose():
    text = ("Total Contract Value: GBP 750,000\n"
            "The total charges for the first year are GBP 60,000.\n")
    assert read_value(text).value == "750000.00"


def test_two_labelled_totals_that_disagree_yield_neither():
    """One contract, one value. Two different labelled totals is a misread."""
    v = read_value("Total Contract Value: GBP 750,000\nContract Value: GBP 60,000\n")
    assert v.value is None and v.currency is None, v


def test_the_same_total_stated_twice_is_not_a_conflict():
    v = read_value("Total Contract Value: GBP 750,000\nContract Value: GBP 750,000\n")
    assert v.value == "750000.00", v


def test_candidates_carry_the_schemas_field_names():
    by_field = {c.field: c.value for c in value_candidates(REAL)}
    assert by_field == {"total_contract_value": "25000.00", "currency": "GBP"}, by_field


def test_the_candidate_keeps_the_literal_money_text_as_evidence():
    for c in value_candidates(REAL):
        assert c.span.text in REAL, c


def test_nothing_is_emitted_without_a_stated_total():
    assert value_candidates("AGREEMENT\nGoverned by English law.\n") == []


def test_the_value_reaches_dispatchs_candidate_set():
    from src.services.extraction.dispatch import _contract_party_candidates
    cands, _barred = _contract_party_candidates("contract", REAL)
    by_field = {c.field: c.value for c in cands}
    assert by_field.get("total_contract_value") == "25000.00", by_field
    assert by_field.get("currency") == "GBP", by_field


def test_the_parsers_it_reuses_need_no_optional_dependency():
    """The date reader's first version called dateparser, which exists in .venv and
    not in venv, so it produced nothing under pytest while working in production.
    These two parsers were checked for the same trap before being relied on, and
    this test is what keeps that true."""
    from src.services.extraction_v2.parsers.amounts import parse_amount
    from src.services.extraction_v2.parsers.currency import parse_currency
    assert str(parse_amount("25,000")) == "25000.00"
    assert str(parse_currency("£")) == "GBP"


# ---------------------------------------------------------------------------
# The backfill decision. Three rules, like the term's: the sweep never produced a
# contract value either, so there is no wrong value of its to clear.
# ---------------------------------------------------------------------------

def _decide(**kw):
    from src.services.extraction.engineered.contract_value import decide_value_correction
    return decide_value_correction(**kw)


def test_a_stored_row_with_no_value_is_filled():
    d = _decide(full_text=REAL, stored_value=None, stored_currency=None,
                provenance_source=None)
    assert (d.value, d.currency) == ("25000.00", "GBP"), d
    assert d.changed is True


def test_a_row_already_holding_the_documents_value_is_left_alone():
    d = _decide(full_text=REAL, stored_value="25000.00", stored_currency="GBP",
                provenance_source="regex")
    assert d.changed is False, d


def test_a_human_confirmed_value_is_never_touched():
    d = _decide(full_text=REAL, stored_value="1.00", stored_currency="USD",
                provenance_source="hitl")
    assert d.changed is False and d.value == "1.00", d


def test_a_value_the_reader_cannot_find_is_not_cleared():
    d = _decide(full_text="AGREEMENT\nGoverned by English law.\n",
                stored_value="500.00", stored_currency="GBP",
                provenance_source="context_layer")
    assert d.changed is False and d.value == "500.00", d


def test_a_stored_value_that_disagrees_with_the_document_is_corrected():
    """The document is the authority. 4,000 is the monthly rate that an earlier
    read could plausibly have taken."""
    d = _decide(full_text="The total charges are GBP 48,000, invoiced at GBP 4,000 per month.",
                stored_value="4000.00", stored_currency="GBP", provenance_source="regex")
    assert d.value == "48000.00" and d.changed is True, d
