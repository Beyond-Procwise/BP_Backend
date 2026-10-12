"""A supplier's final offer gets a recommendation and a reply for a person to approve.

Before this, final-offer wording simply ended the negotiation: status COMPLETED,
no email, nothing raised. On 2026-10-12 a supplier's "final price ... £470, order
by the 20th" went unanswered.

The recommendation uses the same limit the counter plan uses (the walk-away when
there is one, otherwise the target) and never guesses: with no offer or no limit
on record it escalates and says which is missing. The reply is fixed wording,
not model-written. Only an offer within the limit drafts an acceptance; anything
a person has to decide drafts a holding reply that commits to nothing.
"""
import pytest

from src.services.negotiation.final_offer import ACCEPT, ESCALATE, advise, reply_html


class TestTheRecommendation:
    def test_an_offer_within_the_target_is_accepted(self):
        advice = advise(offer=450.0, target=455.0, walkaway=None, currency="GBP")
        assert advice.recommendation == ACCEPT
        assert advice.limit == 455.0 and advice.limit_kind == "target"

    def test_an_offer_exactly_at_the_limit_is_accepted(self):
        assert advise(offer=455.0, target=455.0, walkaway=None, currency="GBP").recommendation == ACCEPT

    def test_the_live_case_470_against_a_455_target_escalates(self):
        advice = advise(offer=470.0, target=455.0, walkaway=None, currency="GBP")
        assert advice.recommendation == ESCALATE
        assert advice.over_by == 15.0
        assert advice.over_pct == pytest.approx(3.3, abs=0.05)
        assert "470.00 GBP" in advice.reason and "15.00 GBP" in advice.reason and "target" in advice.reason

    def test_the_walk_away_is_the_limit_when_there_is_one(self):
        """470 is over the 455 target but inside a 480 walk-away: acceptable."""
        advice = advise(offer=470.0, target=455.0, walkaway=480.0, currency="GBP")
        assert advice.recommendation == ACCEPT
        assert advice.limit == 480.0 and advice.limit_kind == "walk-away"

    def test_over_the_walk_away_escalates(self):
        advice = advise(offer=490.0, target=455.0, walkaway=480.0, currency="GBP")
        assert advice.recommendation == ESCALATE and advice.over_by == 10.0

    def test_no_limit_on_record_escalates_and_says_so(self):
        """The 2026-10-12 conversation had no target stored. Nothing is assumed."""
        advice = advise(offer=470.0, target=None, walkaway=None, currency="GBP")
        assert advice.recommendation == ESCALATE
        assert advice.limit is None and advice.over_by is None
        assert "no target" in advice.reason.lower()

    def test_no_readable_offer_escalates_and_says_so(self):
        advice = advise(offer=None, target=455.0, walkaway=None, currency="GBP")
        assert advice.recommendation == ESCALATE
        assert "could not be read" in advice.reason.lower()

    @pytest.mark.parametrize("bad", [0, -5.0])
    def test_a_non_positive_offer_is_not_an_offer(self, bad):
        assert advise(offer=bad, target=455.0, walkaway=None, currency="GBP").recommendation == ESCALATE

    @pytest.mark.parametrize("bad", [0, -1.0])
    def test_a_non_positive_limit_is_not_a_limit(self, bad):
        advice = advise(offer=470.0, target=bad, walkaway=None, currency="GBP")
        assert advice.recommendation == ESCALATE and advice.limit is None

    def test_a_hold_escalates_an_offer_that_is_within_the_limit(self):
        advice = advise(offer=450.0, target=455.0, walkaway=None, currency="GBP",
                        hold_reason="the policy that governs this agent states no value limit.")
        assert advice.recommendation == ESCALATE
        assert "within the target of 455.00 GBP" in advice.reason
        assert advice.reason.endswith("states no value limit.")

    @pytest.mark.parametrize("empty", [None, "", "   "])
    def test_no_hold_reason_is_no_hold(self, empty):
        assert advise(offer=450.0, target=455.0, walkaway=None, currency="GBP", hold_reason=empty).recommendation == ACCEPT

    def test_a_hold_does_not_change_an_offer_already_over_the_limit(self):
        held = advise(offer=470.0, target=455.0, walkaway=None, currency="GBP", hold_reason="needs sign-off")
        assert held == advise(offer=470.0, target=455.0, walkaway=None, currency="GBP")

    def test_it_is_recorded_as_plain_data(self):
        record = advise(offer=470.0, target=455.0, walkaway=None, currency="GBP").as_dict()
        assert record["recommendation"] == ESCALATE
        assert set(record) >= {"recommendation", "reason", "offer", "currency", "limit", "limit_kind", "over_by", "over_pct"}


class TestTheReply:
    def test_an_acceptance_states_the_price_and_nothing_else(self):
        html = reply_html(advise(offer=450.0, target=455.0, walkaway=None, currency="GBP"), contact_name="Nicholas Geelen")
        assert "Dear Nicholas Geelen," in html
        assert "accept" in html.lower() and "450.00 GBP" in html
        assert "455" not in html, "our own limit must never be told to the supplier"
        assert "Kind regards," in html and "Procurement Team" in html

    def test_an_escalation_commits_to_nothing(self):
        html = reply_html(advise(offer=470.0, target=455.0, walkaway=None, currency="GBP"), contact_name=None)
        lowered = html.lower()
        assert "hello," in lowered
        assert "470.00 GBP" in html
        assert "accept" not in lowered and "agree" not in lowered and "confirm" not in lowered
        assert "455" not in html and "15.00" not in html, "neither the limit nor the gap goes to the supplier"

    def test_an_escalation_with_no_readable_offer_quotes_no_figure(self):
        html = reply_html(advise(offer=None, target=455.0, walkaway=None, currency="GBP"), contact_name=None)
        assert not any(ch.isdigit() for ch in html)

    def test_no_currency_on_record_prints_the_bare_amount(self):
        html = reply_html(advise(offer=450.0, target=455.0, walkaway=None, currency=None), contact_name=None)
        assert "450.00" in html and "None" not in html

    def test_a_contact_name_cannot_inject_markup(self):
        html = reply_html(advise(offer=450.0, target=455.0, walkaway=None, currency="GBP"), contact_name="<b>Nick</b>")
        assert "<b>" not in html
