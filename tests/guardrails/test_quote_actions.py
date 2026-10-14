"""The quote actions, and what the gate says about each when no policy speaks (QDA B8a).

Sending a quote and overriding the margin floor cannot be taken back, so they sit in
classes the gate refuses by default: with no policy row, nobody may do either. Creating
and pricing a quote change a draft only the company sees; they are ordinary writes, which
the platform allows a Buyer by default. Narrowing those two takes a policy row, not a
different class.
"""

import pytest

from src.services import actions, guardrail
from tests.guardrails.test_guardrail_gate import approver, engine_with, viewer
from tests.guardrails.test_rbac import FakePrincipal

QUOTE_ACTIONS = {
    "sales_quote.create": "write",
    "sales_quote.price": "write",
    "sales_quote.send": "communicate",
    "sales_quote.override_margin_floor": "transact",
}
REFUSED_BY_DEFAULT = ["sales_quote.send", "sales_quote.override_margin_floor",
                      "sales_quote.issue", "sales_quote.approve"]


def buyer():
    return FakePrincipal("sub-buyer", {"cognito:groups": ["bp-buyers"]})


def ask(action, principal, *policies):
    return guardrail.authorize(action, actions.action_class(action), principal, {},
                               policy_engine=engine_with(*policies))


@pytest.mark.parametrize("action, cls", sorted(QUOTE_ACTIONS.items()))
def test_each_quote_action_is_a_known_name_with_its_class(action, cls):
    assert actions.is_known(action) and actions.action_class(action) == cls


@pytest.mark.parametrize("action", REFUSED_BY_DEFAULT)
def test_with_no_policy_nobody_may_do_what_cannot_be_taken_back(action):
    decision = ask(action, approver())
    assert decision.allowed is False and "no policy" in decision.reason.lower()


@pytest.mark.parametrize("action", REFUSED_BY_DEFAULT)
def test_no_principal_is_refused(action):
    assert ask(action, None).allowed is False


@pytest.mark.parametrize("action", REFUSED_BY_DEFAULT)
def test_a_policy_for_another_action_does_not_permit_it(action):
    other = next(a for a in REFUSED_BY_DEFAULT if a != action)
    permit = {"policyId": "p", "policyName": "P", "raw_row": {"version": 1}, "details": {
        "policy_identifier": "p", "required_role": "Approver", "applies_to": [other],
        "rules": {"effect": "allow"}}}
    assert ask(action, approver(), permit).allowed is False
    assert ask(other, approver(), permit).allowed is True


@pytest.mark.parametrize("action", sorted(QUOTE_ACTIONS))
def test_a_viewer_may_do_none_of_them(action):
    assert ask(action, viewer()).allowed is False


@pytest.mark.parametrize("action", ["sales_quote.create", "sales_quote.price"])
def test_drafting_is_an_ordinary_write(action):
    """Stated, so nobody reads the list above as covering these two."""
    assert ask(action, buyer()).allowed is True
