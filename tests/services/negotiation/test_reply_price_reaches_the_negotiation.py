"""The price stored with a supplier's reply is the offer the negotiation works from.

The supplier agent hands each reply over with its figures nested under
`supplier_output`. The negotiation only ever looked for a top-level `price`, so
a reply stored with 470.00 GBP still produced "Supplier offer missing or
invalid" and no decision about the price could be made.
"""
from decimal import Decimal

import pytest

from agents.negotiation_agent import NegotiationAgent
from agents.supplier_interaction_agent import SupplierInteractionAgent
from tests.test_negotiation_agent import DummyNick


@pytest.fixture
def agent():
    return NegotiationAgent(DummyNick())


def _handed_over(**row):
    """A reply exactly as the supplier agent hands it to the negotiation."""
    base = {"workflow_id": "wf-1", "unique_id": "PROC-WF-1", "supplier_id": "SUP-1",
            "response_text": "Final price is £470 total."}
    return SupplierInteractionAgent._response_from_row({**base, **row})


def _entry(agent, response):
    entries, _ = agent._extract_batch_inputs({"supplier_responses": [response]})
    (entry,) = entries
    return entry


def test_the_stored_price_becomes_the_current_offer(agent):
    """The database returns NUMERIC as Decimal, and that is what must be read."""
    entry = _entry(agent, _handed_over(price=Decimal("470.0000"), currency="GBP"))
    normalised, issues = agent._normalise_negotiation_inputs(entry)
    assert normalised["current_offer"] == 470.0
    assert normalised["currency"] == "GBP"
    assert "Supplier offer missing or invalid" not in issues


def test_the_handover_carries_the_currency(agent):
    assert _handed_over(price=470.0, currency="GBP")["supplier_output"]["currency"] == "GBP"


def test_a_reply_with_no_stored_price_gains_none(agent):
    entry = _entry(agent, _handed_over(price=None))
    assert entry.get("price") is None
    normalised, issues = agent._normalise_negotiation_inputs(entry)
    assert normalised["current_offer"] is None
    assert "Supplier offer missing or invalid" in issues


def test_a_price_already_on_the_entry_is_kept(agent):
    response = _handed_over(price=470.0, currency="GBP")
    response["price"] = 480.0
    assert _entry(agent, response)["price"] == 480.0


def test_the_handed_over_reply_is_not_changed_in_place(agent):
    response = _handed_over(price=470.0, currency="GBP")
    _entry(agent, response)
    assert "price" not in response


@pytest.mark.parametrize("stored,read", [
    (Decimal("470.0000"), 470.0),
    (Decimal("1250.50"), 1250.5),
    (Decimal("0"), 0.0),
])
def test_a_decimal_from_the_database_is_money(agent, stored, read):
    assert agent._parse_money(stored) == read


@pytest.mark.parametrize("not_money", [Decimal("NaN"), Decimal("Infinity"), True, None, ""])
def test_what_is_not_an_amount_is_still_not_money(agent, not_money):
    assert agent._parse_money(not_money) is None
