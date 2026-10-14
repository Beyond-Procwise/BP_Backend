"""The price a supplier states in a reply is read from their words, and only theirs.

The live case (2026-10-12): the supplier answered a 455.00 GBP counter-offer with
"a final price and no further of £470 total". The reply was stored with no
price, and the negotiation ran with "Supplier offer missing or invalid".

The quoted copy of our own email sits under every reply and carries OUR figures
(490.00 and 455.00 here). Reading those as the supplier's offer would be worse
than reading nothing, so most of this file is about what must NOT be read.
"""
from decimal import Decimal

import pytest

from src.services.negotiation.reply_offer import METHOD, own_words, read_offer

LIVE_REPLY = (
    "Hi Procurement Team,\r\n\r\n"
    "When I have looked at the quote, I have provided the best pricing I can.\r\n"
    "However, I will provide a final price and no further of £470 total.  I need\r\n"
    "this order placed by the 20th of October in order to hold that price for\r\n"
    "you.\r\nThanks\r\nNick\r\n\r\n"
    "On Mon, Oct 12, 2026 at 8:28 AM <supplierconnect@procwise.co.uk> wrote:\r\n\r\n"
    "> Dear Nicholas Geelen,\r\n>\r\n"
    "> We acknowledge your current offer of 490.00 GBP for the supply of 10 A4\r\n"
    "> paper boxes and 4 toner cartridges. We propose a revised total of 455.00\r\n"
    "> GBP, which reflects our targeted commercial position for this order.\r\n"
)


def test_the_live_reply_reads_as_470_pounds():
    offer = read_offer(LIVE_REPLY)
    assert offer is not None
    assert (offer.price, offer.currency, offer.method) == (Decimal("470"), "GBP", METHOD)


def test_our_own_quoted_figures_are_not_the_suppliers_offer():
    """Guards the premise: the quoted text does hold amounts, and they are dropped."""
    assert "490.00" in LIVE_REPLY and "455.00" in LIVE_REPLY
    words = own_words(LIVE_REPLY)
    assert "490.00" not in words and "455.00" not in words
    assert "£470" in words


def test_a_reply_that_states_no_price_reads_as_nothing():
    reply = "Thanks, we will come back to you on Friday.\n\nOn Mon, 12 Oct 2026, Buyer wrote:\n> We propose 455.00 GBP."
    assert read_offer(reply) is None


def test_dates_times_and_quantities_are_not_prices():
    """The old reader took the first number in the email as the price."""
    reply = "We can deliver 10 boxes by the 20th of October, call me on 020 7946 0000 after 8:28."
    assert read_offer(reply) is None


def test_two_different_amounts_are_left_for_a_person():
    reply = "Unit price is £25.00 and the total would be £470.00."
    assert read_offer(reply) is None


def test_the_same_amount_said_twice_is_one_amount():
    offer = read_offer("Our price is £470. To confirm: 470.00 GBP, final.")
    assert offer is not None and offer.price == Decimal("470")


@pytest.mark.parametrize("text,price,currency", [
    ("We can do £1,250.50 all in.", "1250.50", "GBP"),
    ("Best we can offer is 470 GBP.", "470", "GBP"),
    ("Revised total: GBP 470.00", "470.00", "GBP"),
    ("Final: $ 99.9", "99.9", "USD"),
    ("Total 12,000 EUR.", "12000", "EUR"),
    ("Our offer is €470", "470", "EUR"),
])
def test_the_ways_an_amount_is_written(text, price, currency):
    offer = read_offer(text)
    assert offer is not None
    assert (offer.price, offer.currency) == (Decimal(price), currency)


def test_an_amount_wrapped_onto_the_wrapped_reply_header_is_still_quoted():
    """Gmail wraps "On ... wrote:" across two lines; what follows is still the quote."""
    reply = (
        "No change from us.\n\n"
        "On Mon, Oct 12, 2026 at 8:28 AM Procurement <supplierconnect@procwise.co.uk>\n"
        "wrote:\n\n"
        "We propose a revised total of 455.00 GBP.\n"
    )
    assert read_offer(reply) is None


def test_an_outlook_quoted_block_is_not_read():
    reply = (
        "We hold at £470.\n\n"
        "From: Procurement <supplierconnect@procwise.co.uk>\n"
        "Sent: 12 October 2026 08:28\n"
        "To: Nick\nSubject: Re: Pricing Discussion\n\n"
        "We propose a revised total of 455.00 GBP.\n"
    )
    offer = read_offer(reply)
    assert offer is not None and offer.price == Decimal("470")


def test_an_original_message_divider_ends_the_suppliers_words():
    reply = "Agreed at £470.\n-----Original Message-----\nWe propose 455.00 GBP."
    assert read_offer(reply).price == Decimal("470")


def test_a_sentence_starting_with_on_is_not_mistaken_for_a_quote_header():
    reply = "On reflection we can do £470.\nOn delivery, nothing changes."
    assert read_offer(reply).price == Decimal("470")


@pytest.mark.parametrize("empty", [None, "", "   \n"])
def test_nothing_in_reads_as_nothing(empty):
    assert read_offer(empty) is None


def test_zero_is_not_an_offer():
    assert read_offer("Delivery is £0.00 on this order.") is None
