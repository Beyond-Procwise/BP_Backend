"""A send with no human approval is agent-initiated, and policy decides.

resolve_authority already reads EmailReplyAutonomyPolicy and is used by the
orchestrator and decision engine. The send path did not consult it, so two
email-governance mechanisms coexisted unaware of each other.
"""

from src.services import email_dispatch_guard as guard
from tests.guardrails.test_send_path_gate import base_kwargs


def _verdict(**over):
    """The real shape resolve_authority returns. No may_send key exists."""
    base = {
        "agent": "email_dispatch_agent",
        "governed": True,
        "slug": "email_reply_autonomy",
        "policy_id": 473,
        "policy_name": "EmailReplyAutonomyPolicy",
        "auto_intents": [],
        "escalate_intents": ["price_change"],
        "limit_gbp": None,
        "limit_currency": None,
        "max_auto_replies_per_thread": 2,
        "min_intent_confidence": 0.8,
        "reason": "resolved from governed policy",
    }
    base.update(over)
    return base


def test_no_approval_and_nothing_autonomous_is_refused():
    """auto_intents is empty on the live policy today."""
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            authority_lookup=lambda agent: _verdict(),
        )
    )
    assert decision.allowed is False
    assert "autonom" in decision.reason.lower() or "approval" in decision.reason.lower()


def test_a_named_autonomous_intent_is_allowed():
    """So the branch is a real gate, not a second way of always denying."""
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            intent="acknowledgement",
            authority_lookup=lambda agent: _verdict(auto_intents=["acknowledgement"]),
        )
    )
    assert decision.allowed is True, decision.reason


def test_an_intent_not_named_in_policy_is_refused():
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            intent="price_change",
            authority_lookup=lambda agent: _verdict(auto_intents=["acknowledgement"]),
        )
    )
    assert decision.allowed is False


def test_a_send_with_no_intent_is_refused_even_when_intents_are_granted():
    """A plain outbound dispatch carries no intent, so nothing matches."""
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            intent=None,
            authority_lookup=lambda agent: _verdict(auto_intents=["acknowledgement"]),
        )
    )
    assert decision.allowed is False


def test_an_ungoverned_agent_is_refused():
    """governed=False means escalate, not proceed."""
    decision = guard.check_dispatch(
        **base_kwargs(
            approval_lookup=lambda **_: None,
            agent_name="email_dispatch_agent",
            intent="acknowledgement",
            authority_lookup=lambda agent: _verdict(
                governed=False, auto_intents=["acknowledgement"]
            ),
        )
    )
    assert decision.allowed is False, (
        "an ungoverned verdict must escalate even when it names the intent"
    )


def test_a_failing_authority_lookup_is_refused():
    def explode(agent):
        raise RuntimeError("policy engine unreachable")

    decision = guard.check_dispatch(
        **base_kwargs(approval_lookup=lambda **_: None, agent_name="x",
                      intent="acknowledgement", authority_lookup=explode)
    )
    assert decision.allowed is False
