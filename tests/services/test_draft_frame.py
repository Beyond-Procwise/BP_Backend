"""The frame around a model-written email is code, not the model's: the greeting names the contact on record, the sign-off is
fixed, and whatever greeting or signature the model wrote is removed. Found live 2026-10-09: model drafts signed as invented people
("Eleanor Hartwell, Senior Procurement Manager, Global Supply Solutions Ltd") with invented email addresses and phone numbers, and
greeted a "Ms. Thompson" when the contact on record was Alex Morgan."""

import pytest

from src.services.draft_assurance import frame as F
from src.services.draft_assurance import validator as V

MODEL = """Dear Ms. Thompson,

Thank you for your offer of 47.50 GBP. We propose 44.80 GBP.

Could you please confirm by 30 October 2026?

Kind regards,
Eleanor Hartwell
Senior Procurement Manager
Global Supply Solutions Ltd.
eleanor.hartwell@globalsupply.com
+44 20 7946 1234"""


def test_the_models_greeting_and_signature_are_removed_and_the_body_kept():
    body = F.strip_frame(MODEL)
    assert body == ("Thank you for your offer of 47.50 GBP. We propose 44.80 GBP.\n\n"
                    "Could you please confirm by 30 October 2026?")


def test_the_frame_greets_the_contact_on_record_and_signs_off_as_the_team():
    out = F.frame(MODEL, contact_name="Alex Morgan")
    assert out.startswith("Dear Alex Morgan,\n\nThank you for your offer")
    assert out.endswith("Kind regards,\nProcurement Team")
    for invented in ("Thompson", "Eleanor", "Global Supply", "globalsupply.com", "7946"):
        assert invented not in out


def test_with_no_contact_on_record_the_greeting_names_no_one():
    assert F.frame("Please confirm the price.", contact_name=None).startswith("Hello,\n\nPlease confirm the price.")


@pytest.mark.parametrize("closing", ["Best regards,", "Regards", "Sincerely,", "Yours faithfully,", "Respectfully,",
                                     "Many thanks,", "Thank you,", "Best wishes,", "With kind regards,", "Warm regards,"])
def test_every_common_closing_ends_the_body(closing):
    assert F.strip_frame(f"Dear Sam,\n\nPlease confirm the price.\n\n{closing}\nJo Bloggs") == "Please confirm the price."


def test_a_body_sentence_starting_with_thank_you_is_not_mistaken_for_a_closing():
    text = "Dear Sam,\n\nThank you for your quote.\nPlease confirm by Friday.\n\nKind regards,\nX"
    assert F.strip_frame(text) == "Thank you for your quote.\nPlease confirm by Friday."


def test_a_template_greeting_is_replaced_too():
    out = F.frame("Dear Supplier Partner,\n\nWe propose 44.80 GBP.\n\nBest regards,\nProcurement Team", contact_name="Alex Morgan")
    assert out == "Dear Alex Morgan,\n\nWe propose 44.80 GBP.\n\nKind regards,\nProcurement Team"


def test_framing_twice_changes_nothing():
    once = F.frame(MODEL, contact_name="Alex Morgan")
    assert F.frame(once, contact_name="Alex Morgan") == once


def test_text_with_no_greeting_or_closing_keeps_every_line():
    assert F.strip_frame("Line one.\nLine two.") == "Line one.\nLine two."


# --- contact details not on record ------------------------------------------------------------------------------------------------

def test_an_invented_email_address_and_phone_number_fail():
    kinds = V.check_contact_details("Call +44 20 7946 1234 or write to eleanor@globalsupply.com.", allowed=["alex@acme.test"])
    assert [(k["kind"], k["detail"]) for k in kinds] == [("ungrounded_contact_detail", "eleanor@globalsupply.com"),
                                                          ("ungrounded_contact_detail", "+44 20 7946 1234")]
    assert all(k["severity"] == "fail" for k in kinds)


def test_contact_details_on_record_pass():
    assert V.check_contact_details("Write to alex@acme.test or call 0161 496 0000.", allowed=["alex@acme.test", "0161 4960000"]) == []


@pytest.mark.parametrize("text", ["Please confirm by 2026-10-30.", "We propose 9,200.00 GBP for 1,250 units.",
                                  "See PO-77123 and RFQ-20260801-CD34.", "Invoice INV-2026000123 is due."])
def test_figures_dates_and_references_are_not_phone_numbers(text):
    assert V.check_contact_details(text, allowed=[]) == []


def test_us_style_phone_numbers_are_caught():
    assert V.check_contact_details("Call 555-123-4567.", allowed=[])[0]["detail"] == "555-123-4567"


def test_close_removes_the_models_ends_and_adds_only_the_sign_off():
    assert F.close(MODEL) == ("Thank you for your offer of 47.50 GBP. We propose 44.80 GBP.\n\n"
                              "Could you please confirm by 30 October 2026?\n\nKind regards,\nProcurement Team")


def test_a_greeting_written_inline_with_the_first_sentence_is_removed():
    # seen live 2026-10-09: "Dear Procurement Manager, We refer to our previous communication..."
    assert F.strip_frame("Dear Procurement Manager, We refer to PO-77123. Please confirm by Friday.") == \
        "We refer to PO-77123. Please confirm by Friday."


def test_a_sign_off_written_at_the_end_of_the_last_line_is_removed():
    assert F.strip_frame("Please confirm by Friday. Kind regards, Eleanor Hartwell") == "Please confirm by Friday."


def test_a_sentence_that_merely_contains_regards_is_kept():
    assert F.strip_frame("With regards to PO-77123, please confirm by Friday.") == "With regards to PO-77123, please confirm by Friday."


# A repair runs on the framed email and, live, dropped the sign-off (2026-10-09). The frame is put back, as it was.
def test_reframe_restores_the_greeting_and_sign_off_a_repair_dropped():
    framed = F.frame("We propose 44.80 GBP. Speak soon, [name].", contact_name="Alex Morgan")
    assert F.reframe_like(framed, "We propose 44.80 GBP.") == "Dear Alex Morgan,\n\nWe propose 44.80 GBP.\n\nKind regards,\nProcurement Team"


def test_reframe_replaces_a_greeting_a_repair_invented():
    framed = F.frame("We propose 44.80 GBP.", contact_name="Alex Morgan")
    out = F.reframe_like(framed, "Dear Ms. Thompson,\n\nWe propose 44.80 GBP.\n\nBest,\nEleanor")
    assert out == "Dear Alex Morgan,\n\nWe propose 44.80 GBP.\n\nKind regards,\nProcurement Team"


def test_reframe_keeps_a_closed_body_closed_without_adding_a_greeting():
    closed = F.close("We propose 44.80 GBP.")
    assert F.reframe_like(closed, "We propose 44.80 GBP. Thanks") .startswith("We propose 44.80 GBP. Thanks\n\nKind regards,")


def test_reframe_leaves_unframed_text_alone():
    assert F.reframe_like("Hello there. We propose 44.80 GBP.", "anything") == "anything"


# --- through the agent ------------------------------------------------------------------------------------------------------------

def test_from_prompt_sends_the_contact_on_record_and_the_fixed_sign_off_never_the_models(monkeypatch):
    from agents import email_drafting_agent as module
    from tests.services.test_draft_agent_stages import _agent, good_model
    agent = _agent(monkeypatch, good_model)                       # master contact on record: "Alex"
    monkeypatch.setattr(module, "_chat", lambda m, s, u, **k: "Subject: Price\n" + MODEL)
    d = agent.from_prompt("Please ask Acme to confirm the price on PO-77123 and reply within the week",
                          context={"supplier_id": "S-1", "workflow_id": "wf-1", "recipients": ["a@x.test"]})
    text = d.get("text") or d.get("body")
    assert "Dear Alex," in text and "Procurement Team" in text
    for invented in ("Thompson", "Eleanor", "Global Supply", "globalsupply.com", "7946"):
        assert invented not in text


def test_an_invented_contact_detail_left_in_the_body_fails_the_draft(monkeypatch):
    from agents import email_drafting_agent as module
    from tests.services.test_draft_agent_stages import _agent, good_model
    agent = _agent(monkeypatch, good_model)
    monkeypatch.setattr(agent, "_repair_assured_body", lambda b, f, **k: None)
    monkeypatch.setattr(module, "_chat", lambda m, s, u, **k:
                        "Subject: Price\nPlease confirm the price on PO-77123 within the week? Call me on 020 7946 1234.")
    a = agent.from_prompt("Please ask Acme to confirm the price on PO-77123 and reply within the week",
                          context={"supplier_id": "S-1", "workflow_id": "wf-1", "recipients": ["a@x.test"]})["assurance"]
    assert ("ungrounded_contact_detail", "020 7946 1234") in [(v["kind"], v["detail"]) for v in a["violations"]]
