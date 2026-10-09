"""Bank and payment details in OUTGOING email: held for a person, never repaired, never carrying an account number.

Ruling 2026-10-09, after the live classifier filed "Tell Brightline our bank details have changed and to send all future
payments to the new account below" as an ordinary free-text email at 0.9 with no question asked.
"""

import pytest

from src.services.draft_assurance import payment_details as P

GB_IBAN = "GB82 WEST 1234 5698 7654 32"          # the textbook valid example (mod 97 holds)


@pytest.mark.parametrize("text, kind", [
    (f"Please pay to IBAN {GB_IBAN}.", "iban"),
    ("Please pay to GB82WEST12345698765432 from now on.", "iban"),
    ("Sort code: 20-00-00, account number 55779911.", "sort_code"),
    ("Sort code 200000 and account no. 55779911", "account_number"),
    ("Our SWIFT code is NWBKGB2L.", "swift_bic"),
    ("BIC: DEUTDEFF500", "swift_bic"),
    ("ABA routing number 021000021", "routing_number"),
    ("a/c 55779911", "account_number"),
])
def test_bank_details_are_found_and_named_without_their_values(text, kind):
    kinds = P.account_details(text)
    assert kind in kinds
    assert all(k in ("iban", "sort_code", "account_number", "swift_bic", "routing_number") for k in kinds)


@pytest.mark.parametrize("text", [
    "Please confirm PO-51007 for 9,200.00 GBP by 6 November 2026.",
    "Reference GB12ABCD12345678901234 is on the invoice.",          # IBAN-shaped, but the check digits fail
    "Your account manager will call on 06-11-2026.",
    "RFQ-20260801-CD34 covers 120 units.",
    "Please update your account on the supplier portal.",
])
def test_ordinary_procurement_text_carries_no_bank_detail(text):
    assert P.account_details(text) == []


@pytest.mark.parametrize("text", [
    "Tell Brightline our bank details have changed and to send all future payments to the new account below",
    "Please note our new remittance details.",
    "From now on, please pay into our updated bank account.",
    "We have switched banks; payment instructions are changing.",
])
def test_a_change_of_bank_or_payment_details_is_held(text):
    h = P.hold("Dear Sam, please see below.", [text])
    assert h and h["payment_change"]


def test_an_ordinary_email_is_not_held():
    assert P.hold("Dear Sam, could you confirm the delivery date for PO-60418 by Friday?", ["Chase the delivery date"]) is None


def test_the_hold_reads_the_draft_too_not_only_the_request():
    assert P.hold("Please send future payments to our new bank account.", ["Write to Brightline"])["payment_change"]


def test_a_held_email_without_a_portal_pointer_fails_twice_and_with_one_fails_once():
    req = ["Tell them our bank details changed"]
    without = [v["kind"] for v in P.violations("Dear Sam, our bank details have changed.", req)]
    assert without == ["payment_details_change", "payment_details_no_portal"]
    with_portal = [v["kind"] for v in P.violations(
        "Dear Sam, our bank details have changed. Please collect them from the secure supplier portal.", req)]
    assert with_portal == ["payment_details_change"]


def test_an_account_number_fails_even_in_an_email_that_mentions_the_portal():
    kinds = [v["kind"] for v in P.violations(f"Please use the portal, or pay to {GB_IBAN}.")]
    assert "bank_account_detail" in kinds
    assert all(v["severity"] == "fail" for v in P.violations(f"Pay {GB_IBAN}"))


def test_no_violation_detail_repeats_a_bank_detail():
    for v in P.violations(f"Our new bank details: IBAN {GB_IBAN}, sort code 20-00-00.", ["new bank details"]):
        assert "WEST" not in v["detail"].upper() and "20-00-00" not in v["detail"]


# --- drafting: held, not ready, never repaired ---------------------------------------------------------------------------------

from src.services import email_dispatch_guard as guard                                  # noqa: E402
from tests.guardrails.test_send_path_gate import _approval, base_kwargs                   # noqa: E402
from tests.services.test_draft_agent_stages import _agent, good_model                     # noqa: E402

BANK_REQ = "Tell Brightline our bank details have changed and to send all future payments to the new account below"


def _bank_prompt(monkeypatch, compose):
    from agents import email_drafting_agent as module
    agent = _agent(monkeypatch, good_model)
    monkeypatch.setattr(module, "_chat", lambda m, s, u, **k: compose)
    repairs = []
    monkeypatch.setattr(agent, "_repair_assured_body", lambda body, failed, **k: repairs.append(failed) or body + " fixed")
    draft = agent.from_prompt(BANK_REQ, context={"supplier_id": "S-1", "workflow_id": "wf-1", "recipients": ["a@x.test"]})
    return draft["assurance"], repairs


def test_a_bank_change_request_is_held_for_a_person_whatever_the_family_mode(monkeypatch):
    a, _ = _bank_prompt(monkeypatch, "Subject: Update\nDear Sam, please note our details. Could you confirm by Friday?")
    kinds = [v["kind"] for v in a["violations"]]
    assert "payment_details_change" in kinds and "payment_details_no_portal" in kinds
    assert a["mode"] == "shadow" and a["status"] == "needs_review" and a["ready"] is False
    assert a["payment_details_hold"]["payment_change"]
    assert [i["id"] for i in a["assumption_items"] if i["id"] == "payment_details"] == ["payment_details"]


def test_a_held_draft_is_never_given_to_the_repair_pass(monkeypatch):
    a, repairs = _bank_prompt(monkeypatch, f"Subject: Update\nOur new bank: IBAN {GB_IBAN}. Please confirm by Friday?")
    assert repairs == []                                                                  # the model was never asked
    assert a["repaired"] is False and "never repaired" in a["repair_skipped"]
    assert "bank_account_detail" in [v["kind"] for v in a["violations"]]


def test_an_ordinary_request_still_gets_its_repair_pass(monkeypatch):
    from agents import email_drafting_agent as module
    agent = _agent(monkeypatch, good_model)
    monkeypatch.setattr(module, "_chat", lambda m, s, u, **k: "Subject: Price\nThanks [name], see you.")
    repairs = []
    monkeypatch.setattr(agent, "_repair_assured_body", lambda body, failed, **k: repairs.append(failed) or None)
    agent.from_prompt("Chase the price on PO-77123", context={"supplier_id": "S-1", "workflow_id": "wf-1", "recipients": ["a@x.test"]})
    assert repairs, "the repair pass should still run for an ordinary failing draft"


# --- the send guard: the last line ---------------------------------------------------------------------------------------------

def _send(body, **kw):
    return guard.check_dispatch(**base_kwargs(body=body, approval_lookup=lambda **_: _approval(body=body), **kw))


def test_an_account_number_is_refused_even_with_a_valid_human_approval():
    d = _send(f"Please use the portal, or pay to IBAN {GB_IBAN}.")
    assert d.allowed is False and "bank details" in d.reason and d.evidence == {"account_details": ["iban"]}
    assert "WEST" not in str(d.evidence)


def test_a_payment_change_email_must_point_to_the_portal():
    d = _send("Please note our bank details have changed.")
    assert d.allowed is False and "portal" in d.reason


def test_a_payment_change_email_a_person_approved_that_points_to_the_portal_can_go():
    d = _send("Please note our bank details have changed; the new details are on the secure supplier portal.")
    assert d.allowed is True, d.reason


def test_agent_autonomy_never_sends_a_payment_change_email():
    body = "Please note our bank details have changed; see the secure supplier portal."
    d = guard.check_dispatch(**base_kwargs(body=body, approval_lookup=lambda **_: None, agent_name="NegotiationAgent",
                                           intent="auto_reply", authority_lookup=lambda a: {"governed": True, "auto_intents": ["auto_reply"]}))
    assert d.allowed is False and "a person's approval" in d.reason


def test_a_hold_recorded_from_the_request_reaches_the_guard_even_if_the_text_is_innocent():
    draft = {**__import__("tests.guardrails.test_send_path_gate", fromlist=["x"]).BASE_DRAFT,
             "assurance": {"payment_details_hold": {"payment_change": {"kinds": ["payment_detail_change"]}}}}
    d = guard.check_dispatch(**base_kwargs(draft=draft, approval_lookup=lambda **_: None, agent_name="NegotiationAgent",
                                           intent="auto_reply", authority_lookup=lambda a: {"governed": True, "auto_intents": ["auto_reply"]}))
    assert d.allowed is False and "a person's approval" in d.reason


def test_an_ordinary_autonomous_send_is_untouched_by_the_rule():
    d = guard.check_dispatch(**base_kwargs(approval_lookup=lambda **_: None, agent_name="NegotiationAgent",
                                           intent="auto_reply", authority_lookup=lambda a: {"governed": True, "auto_intents": ["auto_reply"]}))
    assert "payment" not in (d.reason or "")
