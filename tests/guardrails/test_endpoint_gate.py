"""The configure and delegate endpoints consult the gate.

Until now `authorize` had four call sites and every one of them was on the email
path. Creating an agent, deleting one, reloading the rules that govern
everything, and retraining a model had no gate to consult -- so granting someone
Admin granted powers nothing checked.

These are the first call sites outside email. The helper is shared rather than
copied because five near-identical inline blocks is how the attribution bugs on
this project have happened before.
"""

from __future__ import annotations

import pytest

from src.services import guardrail
from src.api import endpoint_gate
from tests.guardrails.test_rbac import FakePrincipal


def _as(group):
    return FakePrincipal(f"sub-{group}", {"cognito:groups": [group]})


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    monkeypatch.setattr(guardrail.policy_observation, "record", lambda **k: True)
    monkeypatch.setattr(guardrail, "_raise_for_a_human", lambda **k: 1)


@pytest.fixture
def audited(monkeypatch):
    """Capture the audit rows the helper writes."""

    rows = []
    monkeypatch.setattr(
        endpoint_gate.agent_actions,
        "record_action_or_fail",
        lambda **fields: rows.append(fields),
    )
    return rows


def _engine(*policies):
    from tests.guardrails.test_guardrail_gate import engine_with

    return engine_with(*policies)


PERMIT_CREATE = {
    "policyId": "agent_lifecycle_authority",
    "policyName": "AgentLifecycleAuthorityPolicy",
    "details": {
        "policy_identifier": "agent_lifecycle_authority",
        "required_role": "Admin",
        "applies_to": ["agent.create"],
        "rules": {"effect": "allow"},
    },
    "raw_row": {"version": 1},
}


def test_an_admin_is_permitted_and_the_call_proceeds(audited):
    decision = endpoint_gate.require(
        "agent.create", _as("bp-admins"), engine=_engine(PERMIT_CREATE)
    )

    assert decision.allowed is True


def test_a_buyer_is_refused(audited):
    with pytest.raises(endpoint_gate.NotPermitted) as caught:
        endpoint_gate.require(
            "agent.create", _as("bp-buyers"), engine=_engine(PERMIT_CREATE)
        )

    assert caught.value.status_code == 403


def test_no_principal_is_refused(audited):
    """With authentication switched off every caller is anonymous.

    That must not become permission to create an agent.
    """

    with pytest.raises(endpoint_gate.NotPermitted):
        endpoint_gate.require("agent.create", None, engine=_engine(PERMIT_CREATE))


def test_the_refusal_says_which_policy_refused(audited):
    with pytest.raises(endpoint_gate.NotPermitted) as caught:
        endpoint_gate.require(
            "agent.create", _as("bp-buyers"), engine=_engine(PERMIT_CREATE)
        )

    assert "Buyer" in caught.value.detail or "role" in caught.value.detail.lower()


def test_every_attempt_is_audited_allowed_and_refused(audited):
    endpoint_gate.require("agent.create", _as("bp-admins"), engine=_engine(PERMIT_CREATE))
    with pytest.raises(endpoint_gate.NotPermitted):
        endpoint_gate.require("agent.create", _as("bp-buyers"), engine=_engine(PERMIT_CREATE))

    assert [r["status"] for r in audited] == ["allowed", "denied"]
    assert all(r["action_type"] == "agent.create" for r in audited)
    assert all("policy_name" in r["details"] for r in audited)


def test_an_unknown_action_name_is_refused_not_guessed():
    """A typo must not become an ungoverned action.

    An action absent from the vocabulary matches no policy, so the gate would
    defer it forever. Better to fail here, loudly, at the call site.
    """

    with pytest.raises(KeyError):
        endpoint_gate.require("agent.creat", _as("bp-admins"), engine=_engine())


def test_an_unresolved_action_is_refused_and_says_it_was_raised(audited):
    """Nothing speaks to it: the caller is refused and a person is asked."""

    with pytest.raises(endpoint_gate.NotPermitted) as caught:
        endpoint_gate.require("model.train", _as("bp-admins"), engine=_engine())

    assert caught.value.status_code == 403
    assert "review" in caught.value.detail.lower()
