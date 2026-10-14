"""The agent may not counter beyond the mandate the platform resolved for it.

`resolve_authority` (services/governance_tools/authority.py) reads
EmailReplyAutonomyPolicy, follows its `defer_value_limit_to` to
ApprovalThresholdPolicy, and yields a per-agent block carrying `limit_gbp`.
`negotiation_agent` is one of its linked agents and the live block resolves to
GBP 10,000. The orchestrator injects it into `context.input_data["authority"]`
(orchestrator.py:363) -- and the agent never read it. Nothing bounded what it
could offer a supplier.

Two decisions are encoded here, both taken deliberately:

* The limit is a TOTAL spend. The agent only ever sees a unit price, and
  `proc.supplier_response` carries no quantity, so on the inbound-reply route the
  commitment a counter represents cannot be established. Unknown is out of
  mandate, not within it -- the stance `ungoverned_block` already takes.
* When a counter is out of mandate the NUMBER is withheld, not merely flagged.
  A review flag stops nothing: the email is still drafted around the price.
  `decide_strategy` already has this shape for a refused contract, which returns
  `counter_price: None` rather than a figure it cannot stand behind.
"""
import pytest

from agents.negotiation_agent import NegotiationAgent
from services.governance_tools.authority import ungoverned_block
from tests.test_negotiation_agent import DummyNick

GOVERNED = {
    "agent": "negotiation_agent",
    "governed": True,
    "slug": "email_reply_autonomy",
    "policy_id": 7,
    "policy_name": "EmailReplyAutonomyPolicy",
    "limit_gbp": "10000",
    "limit_currency": "GBP",
    "reason": "resolved from governed policy",
}


@pytest.fixture
def agent():
    return NegotiationAgent(DummyNick())


def _refusal(agent, block, *, counter=100.0, currency="GBP", volume=None):
    return agent._authority_refusal(
        block, counter_price=counter, currency=currency, volume_units=volume
    )


class TestWithinMandate:
    def test_a_countable_commitment_under_the_limit_is_allowed(self, agent):
        # 50 x 100 = 5,000 against a 10,000 limit
        assert _refusal(agent, GOVERNED, counter=50.0, volume=100) is None

    def test_a_commitment_exactly_at_the_limit_is_allowed(self, agent):
        """on_at_or_below is 'approve' in ApprovalThresholdPolicy."""
        assert _refusal(agent, GOVERNED, counter=100.0, volume=100) is None

    def test_no_counter_price_needs_no_mandate(self, agent):
        """Nothing is being committed, so there is nothing to authorise."""
        assert _refusal(agent, GOVERNED, counter=None, volume=100) is None


class TestOutsideMandate:
    def test_a_commitment_over_the_limit_is_refused(self, agent):
        reason = _refusal(agent, GOVERNED, counter=100.01, volume=100)
        assert reason and "10,000" in reason.replace("10000", "10,000")

    def test_an_uncomputable_commitment_is_refused(self, agent):
        """No volume on this route, so the total is unknown -- which is not
        the same as being under the limit."""
        reason = _refusal(agent, GOVERNED, counter=100.0, volume=None)
        assert reason
        assert "quantity" in reason.lower() or "volume" in reason.lower()

    def test_a_different_currency_is_refused_rather_than_compared(self, agent):
        """A GBP limit says nothing about a EUR commitment, and this agent has
        no FX. Comparing the bare numbers would invent a rate of 1.0."""
        reason = _refusal(agent, GOVERNED, counter=50.0, currency="EUR", volume=100)
        assert reason and "EUR" in reason

    def test_a_currency_the_payload_never_stated_is_refused(self, agent):
        reason = _refusal(agent, GOVERNED, counter=50.0, currency=None, volume=100)
        assert reason


class TestUngoverned:
    def test_an_ungoverned_block_is_refused_in_its_own_words(self, agent):
        """`reason` is written to be shown to a buyer verbatim
        (authority.py:32-40), so it must survive into the refusal."""
        block = ungoverned_block("negotiation_agent", "the policy could not be read")
        reason = _refusal(agent, block, counter=50.0, volume=100)
        assert reason and "the policy could not be read" in reason

    def test_a_missing_block_is_refused(self, agent):
        """Absent authority must be indistinguishable from ungoverned authority."""
        assert _refusal(agent, None, counter=50.0, volume=100)

    def test_governed_with_no_stated_limit_is_refused(self, agent):
        """A policy that defers to no approval threshold states no limit. The
        resolver leaves limit_gbp None precisely so this escalates."""
        block = dict(GOVERNED, limit_gbp=None, limit_currency=None)
        assert _refusal(agent, block, counter=50.0, volume=100)

    def test_an_unparseable_limit_narrows_rather_than_widens(self, agent):
        block = dict(GOVERNED, limit_gbp="ten thousand")
        assert _refusal(agent, block, counter=50.0, volume=100)


class TestTheRefusalReachesTheDecision:
    """The gate has to change what leaves the agent, not just return a string."""

    def _decide(self, agent, block):
        decision = {
            "strategy": "counter",
            "counter_price": 88.0,
            "asks": [],
            "price_plan_locked": True,
        }
        agent._apply_authority(
            decision, block, currency="GBP", volume_units=None
        )
        return decision

    def test_the_price_is_withheld(self, agent):
        assert self._decide(agent, GOVERNED)["counter_price"] is None

    def test_the_round_is_marked_for_review(self, agent):
        assert self._decide(agent, GOVERNED)["strategy"] == "review"

    def test_the_decision_says_why_it_was_withheld(self, agent):
        decision = self._decide(agent, GOVERNED)
        assert decision["decision_origin"] == "authority_withheld"
        assert decision["authority_withheld"] is True
        assert decision["rationale"]

    def test_the_governing_policy_is_named_for_the_audit(self, agent):
        decision = self._decide(agent, GOVERNED)
        assert decision["authority"]["policy_name"] == "EmailReplyAutonomyPolicy"
        assert decision["authority"]["policy_id"] == 7

    def test_a_counter_within_mandate_is_left_alone(self, agent):
        decision = {
            "strategy": "counter",
            "counter_price": 50.0,
            "asks": [],
            "price_plan_locked": True,
        }
        agent._apply_authority(decision, GOVERNED, currency="GBP", volume_units=100)
        assert decision["counter_price"] == 50.0
        assert decision["strategy"] == "counter"
        assert not decision.get("authority_withheld")


class TestAWithheldPriceDoesNotLeak:
    """Withholding has to hold everywhere the agent states a number.

    `counter_options` is built from `_optimize_multi_issue`, which knows nothing
    about the mandate, and it is published to `draft_payload["counter_proposals"]`
    and to `AgentOutput.data`. A gate that clears `decision["counter_price"]` and
    leaves that alone has not withheld anything -- it has just moved the figure to
    a different key on the same payload.
    """

    def _run(self, monkeypatch, authority, *, quantity=None):
        from agents.base_agent import AgentContext, AgentOutput, AgentStatus

        agent = NegotiationAgent(DummyNick())
        monkeypatch.setattr(
            agent,
            "_invoke_email_drafting_agent",
            lambda ctx, payload: AgentOutput(
                status=AgentStatus.SUCCESS,
                data={"drafts": [{"supplier_id": "S1"}], "subject": "s", "body": "b"},
            ),
        )
        monkeypatch.setattr(agent, "_await_supplier_responses", lambda **_: [])
        # Resolution must not reach a database from this test; an ungoverned
        # block is what a failed lookup yields in production anyway.
        monkeypatch.setattr(
            agent,
            "_resolve_authority_block",
            lambda ctx: authority,
        )
        context = AgentContext(
            workflow_id="wf-auth",
            agent_id="negotiation",
            user_id="tester",
            input_data={
                "supplier": "S1",
                "current_offer": 1300.0,
                "target_price": 1200.0,
                "rfq_id": "RFQ-AUTH",
                "round": 1,
                "currency": "GBP",
                "supplier_email": ["quotes@supplier.test"],
                **({"quantity": quantity} if quantity is not None else {}),
            },
        )
        return agent.run(context)

    def test_the_decision_names_no_price(self, monkeypatch):
        out = self._run(monkeypatch, ungoverned_block("negotiation_agent", "no policy"))
        assert out.data["decision"]["counter_price"] is None
        assert out.data["decision"]["strategy"] == "review"

    def test_the_counter_proposals_name_no_price(self, monkeypatch):
        out = self._run(monkeypatch, ungoverned_block("negotiation_agent", "no policy"))
        proposals = out.data.get("counter_proposals") or []
        prices = [p.get("price") for p in proposals if isinstance(p, dict)]
        assert not [p for p in prices if p is not None], (
            f"the counter was withheld but a price still shipped in "
            f"counter_proposals: {proposals}"
        )

    def test_the_draft_payload_names_no_price(self, monkeypatch):
        out = self._run(monkeypatch, ungoverned_block("negotiation_agent", "no policy"))
        draft = out.data.get("draft_payload") or {}
        assert not draft.get("counter_price")
        proposals = draft.get("counter_proposals") or []
        assert not [
            p.get("price") for p in proposals
            if isinstance(p, dict) and p.get("price") is not None
        ]

    def test_a_mandated_counter_still_ships_its_price(self, monkeypatch):
        """The gate must not simply silence the agent. Same run, but with a
        quantity the commitment is computable and inside the limit."""
        out = self._run(
            monkeypatch, dict(GOVERNED, limit_gbp="1000000"), quantity=1
        )

        decision = out.data["decision"]
        assert decision["counter_price"] is not None, decision.get("authority_reason")
        assert decision["strategy"] == "counter"
        assert not decision.get("authority_withheld")
        assert decision["authority"]["governed"] is True
        prices = [
            p.get("price")
            for p in (out.data.get("counter_proposals") or [])
            if isinstance(p, dict)
        ]
        assert prices, "a mandated counter should still publish its proposal"


class TestATerminalDecisionIsNotReopened:
    """The mandate escalates a terminal decision; it does not undo it.

    `plan_counter` resolves a supplier's best-and-final into `accept` or
    `decline`, and `_execute_negotiation_round` closes the supplier only on those
    two strategies. Rewriting `accept` to `review` is the bug
    `test_terminal_decision_survives.py` exists to prevent: the supplier is never
    marked ACCEPTED and the loop opens another round bargaining against an offer
    we had already agreed. `_adaptive_strategy` returns early on
    TERMINAL_STRATEGIES for exactly this reason; so must this gate.

    Accepting beyond mandate still has to reach a human -- through
    `human_override_required`, the flag the agent already uses for that, which
    `_run_single_negotiation_locked` reads to close `negotiation_allowed`.
    """

    def _accept(self, agent, block, **kw):
        decision = {
            "strategy": "accept",
            "counter_price": 950.0,
            "asks": [],
            "price_plan_locked": True,
        }
        agent._apply_authority(
            decision, block, currency="GBP", volume_units=kw.get("volume")
        )
        return decision

    def test_accept_survives_an_unmandated_commitment(self, agent):
        assert self._accept(agent, GOVERNED)["strategy"] == "accept"

    def test_accept_beyond_mandate_still_escalates(self, agent):
        decision = self._accept(agent, GOVERNED)
        assert decision["human_override_required"] is True
        assert decision["authority_withheld"] is True
        assert decision["authority_reason"]

    def test_a_decline_commits_nothing_and_is_not_gated(self, agent):
        """plan_counter returns no counter_price on a decline."""
        decision = {"strategy": "decline", "counter_price": None, "asks": []}
        agent._apply_authority(decision, GOVERNED, currency="GBP", volume_units=None)
        assert decision["strategy"] == "decline"
        assert not decision.get("authority_withheld")

    def test_a_mandated_accept_is_untouched(self, agent):
        decision = self._accept(agent, dict(GOVERNED, limit_gbp="1000000"), volume=1)
        assert decision["strategy"] == "accept"
        assert decision["counter_price"] == 950.0
        assert not decision.get("human_override_required")
