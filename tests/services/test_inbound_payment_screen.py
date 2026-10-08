"""The inbound payment-detail-change detector: deterministic, no model.

A reply that asks for NEW or CHANGED bank/payment details is the classic invoice-redirection fraud. The cost of missing one is
a payment to a criminal; the cost of a false alarm is a person glancing at a reply. So the bias is toward flagging, and the
fixtures below pin both lists: what MUST be flagged (including disguised) and what must NOT be (ordinary supplier mail that
shares words with the fraud). Phrases that are flagged but are innocent are listed separately as ACCEPTED false positives.
"""

import pytest

from src.services.draft_assurance import inbound

MUST_FLAG = [
    "Please note our bank details have changed. The new account number is 31926819, sort code 60-16-13.",
    "We have changed banks. Please use the updated bank account for all future payments.",
    "Kindly update your records with our new banking details effective immediately.",
    "Our IBAN has changed to GB29NWBK60161331926819, please remit to this account from now on.",
    "URGENT: replace the payment details you hold for us with the ones below. Do not call to verify, our phones are down.",
    "Going forward, payments should be sent to a different account. Details attached.",
    "We are migrating to a new bank and our sort code and account number will change next week.",
    "Please disregard the bank account on previous invoices and pay to the account in this email instead.",
    "New remittance details: beneficiary Acme Ltd, SWIFT NWBKGB2L. Please amend your supplier file.",
    "I'm writing to inform you of an update to our bank information. Use the following for the next payment.",
    "Our account has been closed. Send the outstanding invoice payment to our new account.",
    "Change of bank: please wire the next payment to the account below.",
    "Please be advised that our payment details were updated, and previous details are no longer valid.",
    "We've switched our bank account, please confirm once you have updated it on your side.",
    "The beneficiary account for settlement has been revised. Pay to the revised account number.",
    "Please transfer the invoice amount to our alternative account, the old one is under audit.",
    "Re: Invoice 4471 - our banking details are different now, new IBAN: DE89370400440532013000",
    # disguised
    "Our B A N K details have changed, please update.",
    "Our b​a​n​k account has changed - new details below.",
    "Our <b>bank</b> <i>account</i> has <u>changed</u>, please use the new one.",
    "OUR NEW BANK ACCOUNT DETAILS ARE BELOW. PLEASE UPDATE YOUR RECORDS.",
    "Our ban<b></b>k det<span>ails</span> have chan<i>ged</i>, please upd<u>ate</u> them.",
    "Our &#98;ank details have &#99;hanged, please update.",
    "please   update\n\nthe\tpayment    details   we   have   on   file",
]

MUST_NOT_FLAG = [
    "Thanks for the order. We can offer 12.50 per unit with delivery in 14 days.",
    "Our new account manager, Sam, will be in touch about your renewal.",
    "The warehouse is closed for the bank holiday on Monday, so delivery moves to Tuesday.",
    "Please change the delivery date to the 30th and confirm the quantity of 500 units.",
    "We have updated our price list; the new prices apply from 1 November.",
    "The payment terms are 30 days from the invoice date; the invoice is attached.",
    "Remittance advice for your payment of 4,200.00 is attached for your records.",
    "Our accounts payable team will process the credit note this week.",
    "The new wire harness assembly is ready for inspection.",
    "Can you send a revised quote with the extended warranty option?",
    "We are happy to update the specification to match your drawing.",
    "Thank you for your payment, which we received on the 3rd.",
    "Our sales office has moved to a new address; the registered address is unchanged.",
    "Please find the amended purchase order attached.",
    "I'll be on leave next week; my colleague will cover the account.",
    "",
]

# Innocent text that the detector will flag: a person looks, which costs seconds. Listed so nobody mistakes it for a bug.
ACCEPTED_FALSE_POSITIVES = [
    "The bank transfer fee has changed, so the total on the invoice is now 5.00 higher.",
]


@pytest.mark.parametrize("text", MUST_FLAG)
def test_a_request_for_new_or_changed_payment_details_is_flagged(text):
    r = inbound.screen_payment_change(subject="", body=text)
    assert r["suspected"] is True and "payment_detail_change" in r["kinds"], text


@pytest.mark.parametrize("text", MUST_NOT_FLAG)
def test_ordinary_supplier_mail_is_not_flagged(text):
    assert inbound.screen_payment_change(subject="", body=text)["suspected"] is False, text


@pytest.mark.parametrize("text", ACCEPTED_FALSE_POSITIVES)
def test_known_false_positives_stay_flagged_because_a_human_look_is_cheap(text):
    assert inbound.screen_payment_change(subject="", body=text)["suspected"] is True


def test_the_subject_alone_can_raise_the_flag():
    assert inbound.screen_payment_change(subject="Our bank details have changed", body="See below.")["suspected"] is True
    assert inbound.screen_payment_change(subject="Re: PO-77123 price", body="Thanks, confirmed at 12.50.")["suspected"] is False


def test_bank_details_alone_are_noted_but_do_not_by_themselves_block():
    r = inbound.screen_payment_change(subject="Invoice 9", body="Remit to IBAN GB29NWBK60161331926819 as always. Thanks.")
    assert r["bank_details_present"] is True and r["suspected"] is False


def test_bank_details_plus_pressure_is_suspected():
    r = inbound.screen_payment_change(subject="Invoice 9", body="URGENT pay today to IBAN GB29NWBK60161331926819 and do not call.")
    assert r["suspected"] is True and r["pressure"] is True and "bank_details_with_pressure" in r["kinds"]


def test_the_result_names_kinds_and_keywords_but_never_carries_text_from_the_email():
    r = inbound.screen_payment_change(subject="", body="Our bank details have changed. New IBAN GB29NWBK60161331926819, sort code 60-16-13.")
    blob = repr(r)
    assert "GB29NWBK" not in blob and "60-16-13" not in blob and "31926819" not in blob
    assert set(r) == {"suspected", "kinds", "terms", "bank_details_present", "pressure"} and all(isinstance(t, str) and len(t) <= 30 for t in r["terms"])


@pytest.mark.parametrize("bad", [None, 0, 5.5, [], {}, b"bytes", object()])
def test_non_text_input_is_not_flagged_and_never_raises(bad):
    assert inbound.screen_payment_change(subject=bad, body=bad)["suspected"] is False


def test_a_huge_body_is_screened_in_bounded_time_and_a_late_phrase_still_counts():
    pad = "Thanks for your continued support and the order. " * 4000           # ~190k chars
    r = inbound.screen_payment_change(subject="", body="Our bank details have changed, please update. " + pad)
    assert r["suspected"] is True
    late = inbound.screen_payment_change(subject="", body=pad + " Our bank details have changed, please update.")
    assert late["suspected"] is True                                          # not only the first part of a long mail is read


def test_the_same_input_always_gives_the_same_answer():
    t = MUST_FLAG[0]
    assert inbound.screen_payment_change("", t) == inbound.screen_payment_change("", t)


@pytest.mark.parametrize("bad", [None, 0, 5.5, [], {}, b"bytes", object()])
def test_the_normaliser_itself_is_safe_on_non_text(bad):
    assert inbound._normalise(bad) == ""


def test_markup_inside_a_word_is_removed_but_a_block_tag_still_separates_words():
    assert inbound._normalise("ban<b></b>k") == "bank"
    assert inbound._normalise("one<br>two<p>three") == "one two three"
