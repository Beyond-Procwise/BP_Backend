"""Every escalate trigger, in isolation, and the one path that auto-sends.

The default policy ships with auto_reply_intents empty, so the auto-send case here
uses a widened policy on purpose: it proves the mechanism works and that widening it
is the ONLY thing that enables a send.

No database is touched: `_fetch_email_reply` and `_classify` are replaced, the same
way tests/test_decision_variance.py replaces `_fetch_finding`.
"""
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

# tests/conftest.py owns sys.path (repo root + src/). See the note in
# tests/api/test_decisions_email_endpoints.py: re-inserting src/ here reordered
# precedence for everything imported after collection and made the suite
# order-dependent.
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
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    # Not just "escalated" -- escalated FOR THIS REASON. An always-escalate
    # implementation satisfies the resolution assertion on its own.
    assert "0.40" in d.rationale
    assert "below the minimum of 0.80" in d.rationale
    # Which governed setting drove it stays visible, in words.
    assert "EmailReplyAutonomyPolicy" in d.rationale


# ----------------------------------------------------------------------------
# The two intent gates. Neither was pinned before: every other test reaches these
# with an intent that is on the escalate list OR on a widened auto list, so either
# gate could be deleted and the suite stayed green. These two tests are the ones
# that fail when a gate is removed, and they assert text distinctive enough to say
# WHICH gate answered.
# ----------------------------------------------------------------------------
def test_an_intent_on_neither_governed_list_is_denied_by_default():
    """GOVERNED ships auto_intents empty. 'acknowledge' is on neither list."""
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=GOVERNED)
    assert d.resolution == ESCALATED
    assert "is not one of the kinds of reply the agent is allowed to answer on its own" in d.rationale
    # Says which governed setting to change, in words rather than as a config key -- the
    # person reading this screen is not the person who edits the policy.
    assert "list of replies the agent may answer unattended" in d.rationale
    assert "auto_reply_intents" not in d.rationale
    # Distinctively NOT the escalate-list gate's wording.
    assert "always puts in front of a person" not in d.rationale


def test_the_escalate_list_wins_when_an_intent_is_on_both_lists():
    """Precedence: a consequential intent is not rescued by also being auto-listed."""
    auth = {**GOVERNED,
            "auto_intents": ["price_change", "acknowledge"],
            "escalate_intents": ["price_change"]}
    eng = _engine(intent=ReplyIntent("price_change", 0.99,
                                     "We can offer 94,000.00 GBP", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED, "the escalate list must take precedence"
    assert "always puts in front of a person" in d.rationale
    assert "EmailReplyAutonomyPolicy" in d.rationale, "which policy decided must stay visible"
    # Distinctively NOT the default-deny gate's wording, which would also mention
    # price_change and would also escalate -- that ambiguity is what hid this.
    assert "is not one of the kinds of reply" not in d.rationale


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
    # `"2" in d.rationale` proved almost nothing -- nearly every rationale here
    # contains a 2 somewhere. Assert the sentence this gate actually writes.
    assert "already answered this thread 2 time(s) unattended" in d.rationale
    assert "at the cap of 2 set by EmailReplyAutonomyPolicy" in d.rationale


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
    # This rationale is rendered verbatim on the Action Centre card (and now in the
    # reply-review panel), so it must read as English to the person reviewing it.
    # It used to name the storage table the reply was looked for in.
    assert "proc." not in d.rationale
    assert "supplier_response" not in d.rationale
    assert "could not be found in our records" in d.rationale


def test_a_widened_policy_permits_an_auto_send():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "price": None, "prior_price": None}   # nothing at stake
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert d.decision == "send"


@pytest.mark.parametrize("authority,intent", [
    (GOVERNED, None),                                                    # escalate path
    ({**GOVERNED, "auto_intents": ["acknowledge"]},
     ReplyIntent("acknowledge", 0.99, "Thank you.", True)),              # send path
])
def test_every_fact_carries_a_source(authority, intent):
    """Iterate the FACTS, not the evidence.

    Iterating d.evidence only proves that the items which exist have a source -- it
    cannot notice a fact that has no evidence item at all, which is the failure that
    matters: a decision is re-derivable only if every fact it rests on is traceable.
    """
    row = {**REPLY_ROW, "prior_price": 96000}
    d = _engine(row=row, intent=intent).decide_email_reply("1", authority=authority)
    assert d.evidence, "a decision with no evidence is a guess"
    for item in d.evidence:
        assert item.source, f"fact {item.fact} has no source"
    sourced = {e.fact for e in d.evidence if e.source}
    missing = set(d.facts) - sourced
    assert not missing, f"facts with no evidence item: {sorted(missing)}"


def test_a_send_records_the_governed_values_it_cleared():
    """I1: a send must be re-derivable for all three governed gates, not just one.

    The record used to hold the model's confidence and the thread count, but neither
    the governed minimum that confidence beat nor the cap the count was under -- so two
    of the three gates could not be checked from the record afterwards.
    """
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "prior_price": 96000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert d.facts["min_intent_confidence"] == 0.8
    assert d.facts["max_auto_replies_per_thread"] == 2
    assert d.facts["value_limit_gbp"] == "10000"
    sources = {e.fact: e.source for e in d.evidence}
    assert "min_intent_confidence" in sources["min_intent_confidence"]
    assert "max_auto_replies_per_thread" in sources["max_auto_replies_per_thread"]
    assert "intent_confidence" in sources, "the model's own claim needs a source too"


def test_the_send_rationale_does_not_claim_an_untested_limit():
    """I2: on the unpriced path no limit was consulted -- and there may be none."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": None}
    row = {**REPLY_ROW, "price": None, "prior_price": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert "no price was recorded on the reply, so no amount was tested" in d.rationale
    assert "nothing exceeds the governed limit" not in d.rationale


def test_the_send_rationale_states_the_amount_and_limit_when_one_was_tested():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "prior_price": 96000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert "the amount at stake is 2000 GBP, within the governed limit of 10000 GBP" in d.rationale
    assert "governed minimum of 0.80" in d.rationale


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
    # Plain English: which policy, and what it does not set -- no config key.
    assert "points at no approval threshold for amounts" in d.rationale
    assert "limit_gbp" not in d.rationale
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
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    # Escalated for THIS reason, not merely escalated.
    assert "points at no approval threshold for amounts" in d.rationale
    assert "An absent limit is not an unlimited one" in d.rationale


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
# ----------------------------------------------------------------------------
# Currency. A delta in one currency tested against a limit in another is a wrong
# answer in the permissive direction. This is a gate, never a conversion.
# ----------------------------------------------------------------------------
def test_matching_currencies_reach_the_arithmetic():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_currency": "GBP"}
    row = {**REPLY_ROW, "currency": "GBP", "price": 94000, "prior_price": 96000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.facts["value_at_stake"] == "2000", "the subtraction must still happen"
    assert d.resolution == RESOLVED


def test_a_lowercase_currency_still_matches():
    """'gbp' and 'GBP' are the same denomination; only case differs."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_currency": "GBP"}
    row = {**REPLY_ROW, "currency": "gbp", "prior_price": 96000, "price": 94000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    assert eng.decide_email_reply("1", authority=auth).resolution == RESOLVED


def test_a_differing_reply_currency_escalates_without_converting():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_currency": "GBP"}
    row = {**REPLY_ROW, "currency": "EUR"}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "EUR" in d.rationale and "GBP" in d.rationale
    assert "no comparison was attempted" in d.rationale.lower()
    assert "currency problem, not a pricing dispute" in d.rationale
    # Nothing was computed across denominations, and no rate was invented.
    assert "value_at_stake" not in d.facts
    assert "prior_offer" not in d.facts


def test_an_absent_reply_currency_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_currency": "GBP"}
    for missing in (None, "", "   "):
        row = {**REPLY_ROW, "currency": missing}
        eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
        d = eng.decide_email_reply("1", authority=auth)
        assert d.resolution == ESCALATED, f"currency={missing!r} must not be assumed"
        assert "no currency is recorded" in d.rationale
        assert "value_at_stake" not in d.facts
        source = next(e.source for e in d.evidence if e.fact == "currency")
        assert source == "proc.supplier_response.currency"


def test_an_absent_limit_currency_escalates():
    """We cannot confirm the denominations match if the limit has none."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_currency": None}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "no currency of its own" in d.rationale
    assert d.facts["value_limit_currency"] is None


def test_both_currencies_are_sourced():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_currency": "GBP"}
    row = {**REPLY_ROW, "currency": "GBP", "prior_price": 96000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    sources = {e.fact: e.source for e in d.evidence}
    assert sources["currency"] == "proc.supplier_response.currency"
    assert "email_reply_autonomy" in sources["value_limit_currency"]
    assert "limit_currency" in sources["value_limit_currency"]


def test_the_prior_offers_assumed_currency_is_disclosed_in_the_evidence():
    """One operand of the subtraction has an unstated denomination. Say so.

    payload->'metadata' carries no currency key on any live row, so gate 4a can prove
    the REPLY matches the limit's currency but not the prior offer. That assumption is
    recorded rather than gated -- a gate would fire on every row and put the limit gate
    back to never executing -- but it must be visible, or `value_at_stake` is not
    re-derivable from its evidence.
    """
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_currency": "GBP"}
    row = {**REPLY_ROW, "currency": "GBP", "price": 94000, "prior_price": 96000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)

    # Behaviour is unchanged: this still sends, and still computes the same amount.
    assert d.resolution == RESOLVED
    assert d.facts["value_at_stake"] == "2000"

    disclosure = next(
        (e for e in d.evidence if e.fact == "prior_offer_currency_basis"), None
    )
    assert disclosure is not None, "an assumed operand must be visible as assumed"
    assert "assumed GBP" in str(disclosure.value)
    assert "unstated" in str(disclosure.value)
    # The source names WHERE the prior came from and WHY its currency is unknown.
    assert "counter_price" in disclosure.source
    assert "no currency key" in disclosure.source
    assert "No conversion was applied" in disclosure.source
    assert d.facts["prior_offer_currency_basis"]


def test_no_currency_disclosure_when_no_amount_was_computed():
    """The disclosure belongs to the subtraction. No subtraction, no claim about it."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "prior_price": None, "prior_price_source": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert "value_at_stake" not in d.facts
    assert not [e for e in d.evidence if e.fact == "prior_offer_currency_basis"]


def test_the_over_limit_rationale_discloses_the_assumed_currency():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": "1000"}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "4000" in d.rationale.replace(",", "")
    assert "unstated in the source" in d.rationale


def test_a_currency_mismatch_on_an_unpriced_reply_invents_no_money_concern():
    """No price means no money moves. A stray currency must not manufacture a gate."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_currency": "GBP"}
    row = {**REPLY_ROW, "price": None, "prior_price": None, "currency": "EUR"}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert d.decision == "send"
    assert "currency" not in d.rationale.lower()


# ----------------------------------------------------------------------------
# I3: is the prior offer the one THIS reply answers? The draft is matched on
# unique_id (workflow+supplier), which is not round-specific.
# ----------------------------------------------------------------------------
def test_a_prior_offer_from_a_different_round_escalates():
    """Round N+1 already dispatched when the round-N reply lands."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "round_number": 1, "prior_price": 96000,
           "prior_price_round": 2}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "is round 1" in d.rationale and "belongs to round 2" in d.rationale
    assert "not the offer this reply answers" in d.rationale
    assert "value_at_stake" not in d.facts, "no amount across mismatched rounds"


def test_a_matching_round_proceeds_to_the_arithmetic():
    """The live path: both sides state a round and they agree."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "round_number": 1, "prior_price": 96000,
           "prior_price_round": 1}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert d.facts["value_at_stake"] == "2000"
    assert d.facts["prior_offer_round"] == "1"
    assert "prior_offer_round_basis" not in d.facts, "nothing to disclose when verified"


@pytest.mark.parametrize("reply_round,draft_round,who", [
    (1, None, "the draft"),
    (None, 1, "the reply"),
    (None, None, "the reply"),
])
def test_an_unverifiable_round_is_disclosed_not_escalated(reply_round, draft_round, who):
    """Ruling: disclose, do not escalate -- an always-escalate here would put the
    limit gate back to never executing."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "round_number": reply_round, "prior_price": 96000,
           "prior_price_round": draft_round}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED, "disclosure, not a gate"
    assert d.facts["value_at_stake"] == "2000"
    basis = next(e for e in d.evidence if e.fact == "prior_offer_round_basis")
    assert "unverified" in str(basis.value)
    assert who in str(basis.value)
    assert "most recent draft" in basis.source
    assert "may not be the offer" in basis.source


# ----------------------------------------------------------------------------
# I4: price IS NULL is an extraction outcome, not proof no money is discussed.
# ----------------------------------------------------------------------------
def test_no_extracted_price_but_a_recorded_offer_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "price": None, "prior_price": 96000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "No price was extracted from this reply" in d.rationale
    assert "96000" in d.rationale
    assert d.facts["prior_offer"] == "96000"


def test_no_price_and_no_offer_is_still_genuinely_unpriced():
    """The existing unpriced send path must survive I4."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "price": None, "prior_price": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    assert eng.decide_email_reply("1", authority=auth).resolution == RESOLVED


# ----------------------------------------------------------------------------
# M1: junk and non-finite prices must not read as "no money moves".
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("junk", ["TBC", "circa 90k", "ninety thousand", "-", "£"])
def test_an_unreadable_price_escalates_rather_than_reading_as_unpriced(junk):
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "price": junk, "prior_price": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "is not a finite number" in d.rationale
    assert d.facts["price_unreadable"] == junk


@pytest.mark.parametrize("bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_a_non_finite_price_escalates_instead_of_raising(bad):
    """Decimal('NaN') survives _num, then raises InvalidOperation on the comparison."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW, "price": bad, "prior_price": 96000}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "is not a finite number" in d.rationale


def test_a_non_finite_prior_offer_is_absent_not_arithmetic():
    assert _engine()._prior_offer({"metadata": {"counter_price": "NaN"}}) == (None, None)


# ----------------------------------------------------------------------------
# I6: the suite ran entirely on int/str operands. Production yields Decimal.
# ----------------------------------------------------------------------------
def test_the_arithmetic_runs_on_the_decimal_types_production_actually_uses():
    """proc.supplier_response.price is numeric -> Decimal('94000.0000'), and
    metadata.counter_price parses to Decimal('96000.0'). The live value_at_stake is
    '2000.0000', a string no int-based test ever produced."""
    auth = {**GOVERNED, "auto_intents": ["acknowledge"]}
    row = {**REPLY_ROW,
           "price": Decimal("94000.0000"),
           "prior_price": Decimal("96000.0"),
           "round_number": 1, "prior_price_round": Decimal("1"),
           "currency": "GBP"}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == RESOLVED
    assert d.facts["value_at_stake"] == "2000.0000", "the live string, not '2000'"
    assert d.facts["price"] == "94000.0000"


def test_the_over_limit_gate_bites_on_decimal_operands():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "limit_gbp": "1000"}
    row = {**REPLY_ROW, "price": Decimal("94000.0000"),
           "prior_price": Decimal("96000.0"), "round_number": 1,
           "prior_price_round": Decimal("1")}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED
    assert "moves 2000.0000 GBP" in d.rationale


# ----------------------------------------------------------------------------
# M2: never raises, because Task 7 hands this an authority dict over HTTP.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("broken", [
    {"min_intent_confidence": "high"},
    {"min_intent_confidence": object()},
    {"max_auto_replies_per_thread": "lots"},
    {"escalate_intents": 42},
    {"auto_intents": None, "max_auto_replies_per_thread": []},
])
def test_a_malformed_authority_block_escalates_rather_than_raising(broken):
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], **broken}
    eng = _engine(intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)   # must not raise
    assert d.resolution == ESCALATED
    assert d.subject_type == "email_reply"


def test_a_raising_fetch_escalates_rather_than_propagating(caplog):
    """Fail closed, say why, and keep the driver's words off the review screen.

    A psycopg error message routinely quotes the failing statement -- table and column
    names included -- and this rationale renders verbatim on a buyer's card. The type,
    the message and the traceback belong in the log; the reader gets a sentence and a
    reference that ties the two together.
    """
    import logging
    eng = _engine()
    def boom(_id):
        raise RuntimeError('relation "proc.supplier_response" does not exist')
    eng._fetch_email_reply = boom  # type: ignore
    with caplog.at_level(logging.ERROR, logger="engines.decision_engine"):
        d = eng.decide_email_reply("1", authority=GOVERNED)
    assert d.resolution == ESCALATED
    # It still says why it escalated, and which reply.
    assert "did not finish" in d.rationale
    assert "nothing" in d.rationale.lower() and "sent" in d.rationale
    assert "supplier reply 1" in d.rationale
    # But no driver text, and nothing that names storage.
    assert "RuntimeError" not in d.rationale
    assert "proc." not in d.rationale
    assert "relation" not in d.rationale
    # The information is not lost: type, message and reference are all in the log,
    # and the reference in the rationale is what finds them.
    logged = caplog.text
    assert "RuntimeError" in logged
    assert 'relation "proc.supplier_response" does not exist' in logged
    import re as _re
    ref = _re.search(r"reference ([0-9a-f]{8})", d.rationale)
    assert ref, "the rationale must carry the reference that finds the log entry"
    assert ref.group(1) in logged


def test_a_missing_governed_confidence_minimum_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"], "min_intent_confidence": None}
    eng = _engine(intent=ReplyIntent("acknowledge", 1.0, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED, "an absent minimum is not a minimum of zero"
    # Plain English on screen; the config key stays in `facts`, where a developer
    # re-deriving the decision can still find it.
    assert "sets no minimum confidence for answering a reply unattended" in d.rationale
    assert "min_intent_confidence" not in d.rationale
    assert "EmailReplyAutonomyPolicy" in d.rationale, "which policy decided must stay visible"
    assert d.facts["min_intent_confidence"] is None


def test_a_missing_governed_thread_cap_escalates():
    auth = {**GOVERNED, "auto_intents": ["acknowledge"],
            "max_auto_replies_per_thread": None}
    row = {**REPLY_ROW, "price": None, "prior_price": None}
    eng = _engine(row=row, intent=ReplyIntent("acknowledge", 0.99, "Thank you.", True))
    d = eng.decide_email_reply("1", authority=auth)
    assert d.resolution == ESCALATED, "an absent cap is not an unlimited one"
    assert "sets no limit on how many times the agent may answer one thread" in d.rationale
    assert "max_auto_replies_per_thread" not in d.rationale
    assert "EmailReplyAutonomyPolicy" in d.rationale, "which policy decided must stay visible"
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




# ============================================================================
# The rationale sweeps (fix round 2 rebuild of the round-1 versions).
#
# Every rationale here is read by a buyer, on the Action Centre card and in the
# reply-review panel. None may hand that person a storage identifier or a policy
# config key to decode, and one that leans on governance must say whose.
#
# The round-1 versions could not do the job they claimed. They swept a hand-written
# dict of 10 escalations while `_decide_email_reply` has TWENTY escalating exits, and
# they matched against a fixed denylist of known keys -- so a gate added tomorrow, or
# an old gate leaking a NEW key, passed silently. Both are now derived from the code:
#
#   * the exits are enumerated by parsing the source (AST), and the fixture is asserted
#     to reach every one of them -- proved by TRACING which exit line each fixture case
#     actually executes, not by trusting its name. Add a gate without covering it and
#     the build breaks;
#   * the jargon rule is general, not a list: after quoted values are stripped (an
#     intent like 'price_change' IS data and belongs on screen), no snake_case
#     identifier may remain. A brand-new key fails without anyone updating a denylist.
# ============================================================================
import ast
import os
import re
import sys
from pathlib import Path

# The filename frames will actually carry, taken from the imported code object rather
# than rebuilt from __file__: the module is imported through a sys.path entry that is not
# normalised ("tests/../src/..."), so a path built by hand here matches the AST but never
# matches a live frame -- which silently made the coverage check pass against nothing.
_ENGINE_CO_FILE = DecisionEngine._decide_email_reply.__code__.co_filename
_ENGINE_SRC = Path(os.path.realpath(_ENGINE_CO_FILE))


def _escalating_exit_lines():
    """Line numbers of every escalating exit in the email-reply decision, from source.

    Counted: `return _escalate(...)` and `return Decision(...)` carrying ESCALATED, in
    `_decide_email_reply`, plus the fail-closed exit in `decide_email_reply` itself. NOT
    counted: the `return Decision(...)` inside the nested `_escalate` factory (it is how
    the others are built, not an exit of its own) and the final send, which is not an
    escalation.
    """
    tree = ast.parse(_ENGINE_SRC.read_text())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "DecisionEngine")
    wanted = {"_decide_email_reply", "decide_email_reply"}
    lines = set()
    for fn in cls.body:
        if not isinstance(fn, ast.FunctionDef) or fn.name not in wanted:
            continue
        nested = {n for f in ast.walk(fn)
                  if isinstance(f, ast.FunctionDef) and f is not fn
                  for n in ast.walk(f)}
        for node in ast.walk(fn):
            if node in nested or not isinstance(node, ast.Return) or node.value is None:
                continue
            call = node.value
            if not isinstance(call, ast.Call):
                continue
            name = getattr(call.func, "id", None) or getattr(call.func, "attr", None)
            if name == "_escalate":
                lines.add(node.lineno)
            elif name == "Decision" and any(
                isinstance(kw.value, ast.Name) and kw.value.id == "ESCALATED"
                for kw in call.keywords
            ):
                lines.add(node.lineno)
    return lines


def _traced(call):
    """Run `call`, recording which lines of the decision engine executed.

    Pure stdlib line tracing; any trace function already installed is restored
    afterwards, so this cannot disturb the rest of the run.
    """
    seen = set()

    def tracer(frame, event, arg):
        if frame.f_code.co_filename != _ENGINE_CO_FILE:
            return None
        if event == "line":
            seen.add(frame.f_lineno)
        return tracer

    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = call()
    finally:
        sys.settrace(previous)
    return result, seen


_ACK = ReplyIntent("acknowledge", 0.99, "Thank you.", True)
_WIDE = {**GOVERNED, "auto_intents": ["acknowledge"]}


def _real_authority(policy_rows, *, raising=False):
    """An authority block from the REAL resolver, not a hand-written dict.

    The four fail-closed `reason` strings in authority.py are interpolated straight
    into the rationale a person reads, so at least one path has to arrive here through
    the resolver itself rather than through a literal a test author chose.
    """
    from engines.policy_engine import PolicyEngine
    from src.services.governance_tools.authority import resolve_authority

    engine = PolicyEngine(policy_rows=list(policy_rows))
    if raising:
        def _boom(_slug):
            raise RuntimeError("policy store unavailable")
        engine.get_policy = _boom  # type: ignore[assignment]
    return resolve_authority(engine, ["email_drafting_agent"])["email_drafting_agent"]


def _autonomy_policy_row(rules_overrides=None):
    import json as _json
    rules = {
        "auto_reply_intents": [], "escalate_intents": ["price_change"],
        "defer_value_limit_to": "approval_threshold",
        "max_auto_replies_per_thread": 2, "min_intent_confidence": 0.8,
    }
    rules.update(rules_overrides or {})
    return {
        "policy_id": 11, "policy_name": "EmailReplyAutonomyPolicy",
        "policy_type": "email_autonomy", "policy_desc": "When the agent may reply",
        "policy_details": _json.dumps(
            {"policy_identifier": "email_reply_autonomy", "rules": rules}),
        "policy_linked_agents": "email_drafting_agent",
    }


def _no_threshold_approval_row():
    import json as _json
    return {
        "policy_id": 10, "policy_name": "ApprovalThresholdPolicy",
        "policy_type": "approval", "policy_desc": "Spend authority",
        # No default_threshold_gbp -> the resolver's "sets no threshold amount" reason.
        "policy_details": _json.dumps(
            {"policy_identifier": "approval_threshold", "rules": {"currency": "GBP"}}),
        "policy_linked_agents": "email_drafting_agent",
    }


def _every_escalation():
    """One escalation per gate. Values are (decision, lines executed).

    Keys are for readable failures only -- coverage is proved by the traced lines, so a
    case that stops reaching the gate its name claims fails
    `test_every_escalation_exit_in_the_code_is_swept` rather than passing quietly.
    """
    out = {}

    def case(name, build):
        out[name] = _traced(build)

    # -- authority --------------------------------------------------------------
    case("no_authority_at_all", lambda: _engine().decide_email_reply("1", authority=None))
    case("ungoverned_literal", lambda: _engine().decide_email_reply(
        "1", authority={"governed": False, "auto_intents": [], "escalate_intents": [],
                        "reason": "no usable governed policy 'email_reply_autonomy'"}))
    # Through the REAL resolver: each of authority.py's fail-closed reasons in turn.
    case("resolver_no_policy", lambda: _engine().decide_email_reply(
        "1", authority=_real_authority([])))
    case("resolver_policy_unreadable", lambda: _engine().decide_email_reply(
        "1", authority=_real_authority([], raising=True)))
    case("resolver_no_rules", lambda: _engine().decide_email_reply(
        "1", authority=_real_authority([{**_autonomy_policy_row(), "policy_details": "{}"}])))
    case("resolver_no_threshold_amount", lambda: _engine().decide_email_reply(
        "1", authority=_real_authority(
            [_autonomy_policy_row(), _no_threshold_approval_row()])))
    # -- classification ---------------------------------------------------------
    case("ungrounded", lambda: _engine(
        intent=ReplyIntent("acknowledge", 0.99, "invented sentence", False)
    ).decide_email_reply("1", authority=_WIDE))
    case("no_min_confidence", lambda: _engine(intent=_ACK).decide_email_reply(
        "1", authority={**_WIDE, "min_intent_confidence": None}))
    case("low_confidence", lambda: _engine(
        intent=ReplyIntent("acknowledge", 0.4, "Thank you.", True)
    ).decide_email_reply("1", authority=_WIDE))
    # -- intent lists -----------------------------------------------------------
    case("escalate_listed", lambda: _engine(
        intent=ReplyIntent("price_change", 0.99, "We can offer 94,000.00 GBP", True)
    ).decide_email_reply("1", authority=GOVERNED))
    case("not_auto_listed", lambda: _engine(intent=_ACK).decide_email_reply(
        "1", authority=GOVERNED))
    # -- money ------------------------------------------------------------------
    case("unreadable_price", lambda: _engine(
        row={**REPLY_ROW, "price": "circa 90k"}, intent=_ACK
    ).decide_email_reply("1", authority=_WIDE))
    case("no_price_but_a_prior_offer", lambda: _engine(
        row={**REPLY_ROW, "price": None, "prior_price": Decimal("90000")}, intent=_ACK
    ).decide_email_reply("1", authority=_WIDE))
    case("no_value_limit", lambda: _engine(intent=_ACK).decide_email_reply(
        "1", authority={**_WIDE, "limit_gbp": None}))
    case("reply_has_no_currency", lambda: _engine(
        row={**REPLY_ROW, "currency": None}, intent=_ACK
    ).decide_email_reply("1", authority=_WIDE))
    case("limit_has_no_currency", lambda: _engine(intent=_ACK).decide_email_reply(
        "1", authority={**_WIDE, "limit_currency": None}))
    case("currency_mismatch", lambda: _engine(
        row={**REPLY_ROW, "currency": "EUR"}, intent=_ACK
    ).decide_email_reply("1", authority=_WIDE))
    case("no_prior_offer", lambda: _engine(
        row={**REPLY_ROW, "prior_price": None, "draft_payload": {}}, intent=_ACK
    ).decide_email_reply("1", authority=_WIDE))
    case("round_mismatch", lambda: _engine(
        row={**REPLY_ROW, "round_number": 2, "prior_price_round": Decimal("1")}, intent=_ACK
    ).decide_email_reply("1", authority=_WIDE))
    case("over_the_limit", lambda: _engine(intent=_ACK).decide_email_reply(
        "1", authority={**_WIDE, "limit_gbp": "1000"}))
    # -- thread cap -------------------------------------------------------------
    case("no_thread_cap", lambda: _engine(
        row={**REPLY_ROW, "price": None, "prior_price": None}, intent=_ACK
    ).decide_email_reply("1", authority={**_WIDE, "max_auto_replies_per_thread": None}))
    case("uncounted_history", lambda: _engine(
        row={**REPLY_ROW, "auto_replies_on_thread": None}, intent=_ACK
    ).decide_email_reply("1", authority=_WIDE))
    case("thread_cap_hit", lambda: _engine(
        row={**REPLY_ROW, "auto_replies_on_thread": 2}, intent=_ACK
    ).decide_email_reply("1", authority=_WIDE))
    # -- the reply itself, and the fail-closed handler ---------------------------
    def _missing_reply():
        eng = _engine()
        eng._fetch_email_reply = lambda _id: None  # type: ignore
        return eng.decide_email_reply("999", authority=GOVERNED)
    case("missing_reply", _missing_reply)

    def _raising_fetch():
        eng = _engine()
        def boom(_id):
            raise RuntimeError('relation "proc.supplier_response" does not exist')
        eng._fetch_email_reply = boom  # type: ignore
        return eng.decide_email_reply("1", authority=GOVERNED)
    case("decision_did_not_finish", _raising_fetch)
    return out


_SWEEP = _every_escalation()


def test_every_escalation_exit_in_the_code_is_swept():
    """The sweep must cover the CODE, not a list somebody kept up to date by hand.

    Exits come from the source; coverage comes from tracing what actually ran. Add an
    escalation gate and this fails until the fixture above reaches it -- which is the
    only version of this test that does what the round-1 one claimed to.
    """
    exits = _escalating_exit_lines()
    assert len(exits) >= 19, f"expected ~20 escalating exits, parsed {len(exits)}"
    covered = set()
    for _name, (_d, lines) in _SWEEP.items():
        covered |= lines & exits
    missing = sorted(exits - covered)
    assert not missing, (
        "escalation exits never reached by the sweep, at "
        f"{_ENGINE_SRC.name} lines {missing} -- add a case to _every_escalation()"
    )


def test_all_of_them_really_are_escalations():
    for name, (d, _lines) in _SWEEP.items():
        assert d.resolution == ESCALATED, f"{name} did not escalate"
        assert d.subject_type == "email_reply"


# A quoted value is DATA and belongs on screen ('price_change' is what the classifier
# read). Everything outside quotes is prose, and prose has no snake_case in it.
_QUOTED = re.compile(r"'[^']*'")
_SNAKE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


@pytest.mark.parametrize("gate", sorted(_SWEEP))
def test_no_rationale_hands_the_reader_an_identifier_to_decode(gate):
    d, _lines = _SWEEP[gate]
    prose = _QUOTED.sub("", d.rationale)
    leaked = _SNAKE.findall(prose)
    assert not leaked, f"{gate} leaked {leaked} in: {d.rationale}"
    assert "proc." not in d.rationale, f"{gate}: {d.rationale}"


@pytest.mark.parametrize("gate", sorted(_SWEEP))
def test_an_escalation_that_leans_on_governance_says_whose(gate):
    """Derived, not a hardcoded list of gate names: if the sentence appeals to a
    governed rule at all, the policy that set it must be named, or the reader cannot
    re-derive the decision from what is on screen."""
    d, _lines = _SWEEP[gate]
    leans = ("governed" in d.rationale or "policy" in d.rationale.lower())
    if not leans or not d.policy_name:
        return
    assert d.policy_name in d.rationale, (
        f"{gate} appeals to a governed rule without naming the policy: {d.rationale}"
    )
