"""Every escalate trigger, in isolation, and the one path that auto-sends.

The default policy ships with auto_reply_intents empty, so the auto-send case here
uses a widened policy on purpose: it proves the mechanism works and that widening it
is the ONLY thing that enables a send.

No database is touched: `_fetch_email_reply` and `_classify` are replaced, the same
way tests/test_decision_variance.py replaces `_fetch_finding`.
"""
import os
import sys
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

import pytest

from engines.decision_engine import DecisionEngine, ESCALATED, RESOLVED
from src.services.email_intent import ReplyIntent

GOVERNED = {
    "agent": "email_drafting_agent",
    "governed": True,
    "slug": "email_reply_autonomy",
    "policy_id": 11,
    "policy_name": "EmailReplyAutonomyPolicy",
    "auto_intents": [],
    "escalate_intents": ["price_change", "terms_change"],
    "limit_gbp": "10000",
    "limit_currency": "GBP",
    "max_auto_replies_per_thread": 2,
    "min_intent_confidence": 0.8,
    "reason": "resolved from governed policy",
}

# Shaped like a real proc.supplier_response row joined to its draft. `deal_id` is
# deliberately absent: neither table has such a column (verified against bp_sqldb
# 2026-07-28), so a test row carrying one would prove nothing about production.
REPLY_ROW = {
    "id": 1,
    "workflow_id": "wf-1",
    "unique_id": "wf-1-PeopleFirst",
    "supplier_id": "PeopleFirst HR Solutions Ltd",
    "response_subject": "RE: Negotiation",
    "response_text": "Thank you. We can offer 94,000.00 GBP with 45 day payment terms.",
    "response_from": "billing@peoplefirst.invalid",
    "round_number": 1,
    "match_confidence": 1.0,
    "price": 94000,
    "currency": "GBP",
    "payment_terms": "45 Days",
    "lead_time": 14,
    # from the matched draft, via payload->metadata->counter_price
    "prior_price": 90000,
    "prior_price_source": "proc.draft_rfq_emails.payload.metadata.counter_price",
    "auto_replies_on_thread": 0,
}


def _nick(**over):
    base = dict(
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
        get_db_connection=MagicMock(side_effect=RuntimeError("db off in test")),
        chat=lambda **k: "{}",
    )
    base.update(over)
    return SimpleNamespace(**base)


def _engine(*, row=None, intent=None, nick=None):
    eng = DecisionEngine(nick or _nick())
    eng._fetch_email_reply = lambda _id: (REPLY_ROW if row is None else row)  # type: ignore
    eng._classify = lambda body: (  # type: ignore
        intent or ReplyIntent("price_change", 0.95, "We can offer 94,000.00 GBP", True)
    )
    return eng


def test_a_consequential_intent_escalates():
    d = _engine().decide_email_reply("1", authority=GOVERNED)
    assert d.resolution == ESCALATED
    assert d.subject_type == "email_reply"
    assert d.subject_id == "wf-1-PeopleFirst"
    assert "price_change" in d.rationale


def test_value_over_the_governed_limit_escalates_even_for_a_routine_intent():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": "1000"}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    # 94,000 vs a prior 90,000 is 4,000 at stake, over a 1,000 limit.
    assert "4000" in d.rationale.replace(",", "") or "4,000" in d.rationale


def test_low_confidence_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.4, "Thank you.", True))
    assert eng.decide_email_reply("1", authority=auth).resolution == ESCALATED


def test_an_ungrounded_classification_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "invented sentence", False))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "ground" in d.rationale.lower()


def test_missing_authority_escalates_and_names_the_policy():
    d = _engine().decide_email_reply("1", authority=None)
    assert d.resolution == ESCALATED
    assert "email_reply_autonomy" in d.rationale


def test_ungoverned_authority_escalates():
    auth = {"agent": "email_drafting_agent", "governed": False,
            "auto_intents": [], "escalate_intents": [],
            "reason": "no usable governed policy 'email_reply_autonomy'"}
    d = _engine().decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "no usable governed policy" in d.rationale


def test_thread_cap_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "auto_replies_on_thread": 2}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "2" in d.rationale


def test_an_uncounted_thread_history_escalates():
    """An unknown number of prior unattended replies is not zero of them.

    `_fetch_email_reply` reports None when the count query could not be run, and a
    cap can only be enforced against a number we actually have.
    """
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "auto_replies_on_thread": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "could not be counted" in d.rationale


def test_a_missing_reply_escalates_rather_than_inventing_a_subject():
    eng = _engine(row=None)
    eng._fetch_email_reply = lambda _id: None  # type: ignore
    d = eng.decide_email_reply("999", authority=GOVERNED)
    assert d.resolution == ESCALATED
    assert "999" in d.rationale


def test_a_widened_policy_permits_an_auto_send():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "price": None, "prior_price": None}   # nothing at stake
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert d.decision == "send"


def test_every_fact_carries_a_source():
    d = _engine().decide_email_reply("1", authority=GOVERNED)
    assert d.evidence, "a decision with no evidence is a guess"
    for item in d.evidence:
        assert item.source, f"fact {item.fact} has no source"


# ----------------------------------------------------------------------------
# A priced reply with no resolved limit. resolve_authority() returns
# governed=True with limit_gbp=None whenever the autonomy policy omits
# `defer_value_limit_to` (src/services/governance_tools/authority.py:107-125),
# so this is a reachable state, not a hypothetical -- and under a
# `limit is not None and at_stake > limit` test it would SEND.
# ----------------------------------------------------------------------------
def test_a_priced_reply_with_no_resolved_limit_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": None}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED, "an absent limit is not an unlimited one"
    assert "no governed value limit" in d.rationale
    # The absence is itself recorded, with the source that should have carried it --
    # otherwise the escalation could not be re-derived from the evidence.
    assert d.facts["value_limit_gbp"] is None
    limit_ev = next(e for e in d.evidence if e.fact == "value_limit_gbp")
    assert limit_ev.value is None
    assert "defer_value_limit_to" in limit_ev.source
    assert "value_at_stake" not in d.facts, "nothing was computed against a missing limit"


def test_a_priced_reply_with_no_resolved_limit_escalates_even_within_any_amount():
    """Not rescued by the amount being small: there is nothing to compare it to."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": None}
    row = {**REPLY_ROW, "price": 1, "prior_price": 1}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    assert eng.decide_email_reply("1", authority=auth).resolution == ESCALATED


def test_an_unpriced_reply_does_not_need_a_limit():
    """The gate is about money moving. No price, no limit required."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": None}
    row = {**REPLY_ROW, "price": None, "prior_price": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert d.decision == "send"


def test_a_price_with_no_prior_offer_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "prior_price": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "no prior offer" in d.rationale


def test_the_governed_limit_is_cited_as_evidence():
    """The arithmetic is only checkable if both numbers are sourced."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": "1000"}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    sources = {e.fact: e.source for e in d.evidence}
    assert "email_reply_autonomy" in sources["value_limit_gbp"]
    assert "draft" in sources["value_at_stake"]
    assert sources["price"] == "proc.supplier_response.price"
    # The prior offer is cited at its real path, not as "the draft" in general.
    assert sources["prior_offer"] == (
        "proc.draft_rfq_emails.payload.metadata.counter_price"
    )
    assert "proc.supplier_response.price" in sources["value_at_stake"]
    assert "counter_price" in sources["value_at_stake"]


# ----------------------------------------------------------------------------
# The prior offer, read from the one structured location that really carries it:
# payload->'metadata'->>'counter_price'. Verified live on all 18 draft rows.
# ----------------------------------------------------------------------------
def test_the_prior_offer_is_read_from_the_draft_payload():
    eng = _engine()
    value, source = eng._prior_offer(
        {"metadata": {"counter_price": 96000.0, "strategy": "counter"},
         "body": "<li>Our target positioning: &#163;96,000.00</li>"}
    )
    assert value == Decimal("96000.0")
    assert source == "proc.draft_rfq_emails.payload.metadata.counter_price"


def test_a_wired_prior_offer_computes_the_amount_at_stake():
    """94,000 against our 96,000 is 2,000 at stake -- inside a 10,000 limit."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "price": 94000, "prior_price": 96000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.facts["prior_offer"] == "96000"
    assert d.facts["value_at_stake"] == "2000"
    assert d.resolution == RESOLVED


@pytest.mark.parametrize(
    "payload",
    [
        None,                                       # no draft joined
        {},                                         # payload with no metadata
        {"metadata": None},                         # metadata explicitly null
        {"metadata": {}},                           # metadata without the key
        {"metadata": {"counter_price": None}},      # key present, null
        {"metadata": {"counter_price": ""}},        # key present, empty
        {"metadata": {"counter_price": "about 96k"}},  # unparseable
        {"metadata": {"counter_price": {"gbp": 96000}}},  # wrong shape
        {"metadata": "counter_price=96000"},        # metadata not an object
        "not json at all",                          # payload not an object
        # The supplier's OWN number must never become "our" prior offer.
        {"price": 94000, "target_price": 94000, "offer_price": 94000},
    ],
)
def test_an_unusable_prior_offer_is_absent_not_zero(payload):
    assert _engine()._prior_offer(payload) == (None, None)


def test_an_absent_prior_offer_still_escalates_after_wiring():
    """The wiring must not have turned the missing-prior gate off."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "prior_price": None, "prior_price_source": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "no prior offer" in d.rationale
    assert "value_at_stake" not in d.facts


# ----------------------------------------------------------------------------
# All three governed gate inputs are None-able from resolve_authority while
# governed=True. All three must fail closed, for the same reason.
# ----------------------------------------------------------------------------
def test_a_missing_governed_confidence_minimum_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "min_intent_confidence": None}
    eng = _engine(intent=ReplyIntent("acknowledge", 1.0, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED, "an absent minimum is not a minimum of zero"
    assert "min_intent_confidence" in d.rationale
    assert d.facts["min_intent_confidence"] is None


def test_a_missing_governed_thread_cap_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"],
            "max_auto_replies_per_thread": None}
    row = {**REPLY_ROW, "price": None, "prior_price": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED, "an absent cap is not an unlimited one"
    assert "max_auto_replies_per_thread" in d.rationale
    assert d.facts["max_auto_replies_per_thread"] is None


def test_deal_id_is_not_invented():
    """Neither proc.supplier_response nor proc.draft_rfq_emails has a deal_id
    column, and deal_id is owned by a DB stored procedure. It stays None."""
    d = _engine().decide_email_reply("1", authority=GOVERNED)
    assert d.deal_id is None
    assert d.supplier_id == "PeopleFirst HR Solutions Ltd"


# ----------------------------------------------------------------------------
# The classifier caller. classify_reply(body, *, caller=...) needs an object with
# `call_ollama`, which lives on BaseAgent -- AgentNick itself does not have it.
# ----------------------------------------------------------------------------
def test_the_classifier_caller_is_a_registered_agent_not_agentnick():
    registered = SimpleNamespace(call_ollama=lambda **k: {"response": "{}"})
    nick = _nick(agents={"email_drafting_agent": registered})
    assert not hasattr(nick, "call_ollama"), "AgentNick has no call_ollama; that is the point"
    assert DecisionEngine(nick)._reply_caller() is registered


def test_an_agent_without_call_ollama_is_not_used_as_the_caller():
    nick = _nick(agents={"nope": SimpleNamespace(reason=lambda t: t)})
    # No usable registered agent -> falls back to constructing one, which cannot
    # succeed on this stub nick, so no caller is obtained. It must not raise.
    assert DecisionEngine(nick)._reply_caller() is None


def test_an_unobtainable_classifier_escalates_instead_of_raising():
    nick = _nick(agents={})
    eng = DecisionEngine(nick)
    eng._fetch_email_reply = lambda _id: REPLY_ROW  # type: ignore
    intent = eng._classify("We can offer 94,000.00 GBP.")
    assert intent.grounded is False
    assert intent.confidence == 0.0
    d = eng.decide_email_reply("1", authority=GOVERNED)
    assert d.resolution == ESCALATED
    assert "ground" in d.rationale.lower()
