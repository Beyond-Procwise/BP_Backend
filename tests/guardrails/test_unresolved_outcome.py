"""The gate has three answers, not two: yes, no, and "nobody said".

A rule that defers to policy, and a policy that does not resolve it, is not a
grant. It is a question, and a question belongs in front of a person.

Before this, a policy that matched an action and merely failed to deny it became
the permitting policy -- so a supplier-ranking weights table, given an
``applies_to``, authorised report export. And an irreversible action nothing
mentioned was refused without anyone being told. Both were the gate deciding
alone. Now a permit must be stated, and anything unstated escalates.

``allowed`` stays False for an unresolved decision, so a caller that only checks
``if decision.allowed`` fails closed exactly as it did before.
"""

from __future__ import annotations

import pytest

from src.services import guardrail
from tests.guardrails.test_guardrail_gate import engine_with
from tests.guardrails.test_rbac import FakePrincipal


def _as(group):
    return FakePrincipal(f"sub-{group}", {"cognito:groups": [group]})


def _policy(identifier, name, *, applies_to, required_role=None, rules=None):
    details = {"policy_identifier": identifier, "applies_to": applies_to,
               "rules": rules or {}}
    if required_role:
        details["required_role"] = required_role
    return {"policyId": identifier, "policyName": name, "details": details,
            "raw_row": {"version": 1}}


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """Keep observations and escalations off the database."""

    monkeypatch.setattr(guardrail.policy_observation, "record", lambda **k: True)
    raised = []
    monkeypatch.setattr(guardrail, "_raise_for_a_human", lambda **k: raised.append(k))
    return raised


@pytest.fixture
def escalations(_quiet):
    return _quiet


# --- a policy that says nothing has not said yes ---------------------------

def test_a_policy_that_states_no_effect_does_not_permit(escalations):
    """The bug this closes: matching was being read as permitting."""

    weights = _policy("weight_allocation_policy", "WeightAllocationPolicy",
                      applies_to=["report.export"], rules={"default_weights": {}})

    decision = guardrail.authorize("report.export", "share", _as("bp-admins"),
                                   policy_engine=engine_with(weights))

    assert decision.allowed is False
    assert decision.unresolved is True


def test_an_unresolved_decision_is_put_in_front_of_a_person(escalations):
    weights = _policy("weight_allocation_policy", "WeightAllocationPolicy",
                      applies_to=["report.export"], rules={"default_weights": {}})

    guardrail.authorize("report.export", "share", _as("bp-admins"),
                        policy_engine=engine_with(weights))

    assert len(escalations) == 1
    raised = escalations[0]
    assert raised["action"] == "report.export"
    assert raised["principal_subject"] == "sub-bp-admins"


def test_an_irreversible_action_no_policy_mentions_is_unresolved(escalations):
    """Nothing said no either. Refusing quietly is still deciding alone."""

    decision = guardrail.authorize("agent.create", "configure", _as("bp-admins"),
                                   policy_engine=engine_with())

    assert decision.allowed is False
    assert decision.unresolved is True
    assert len(escalations) == 1


# --- a rule that speaks is resolved, and is not escalated ------------------

def test_an_explicit_allow_permits_and_raises_nothing(escalations):
    permit = _policy("export_permit", "ReportExportPolicy",
                     applies_to=["report.export"], required_role="Admin",
                     rules={"effect": "allow"})

    decision = guardrail.authorize("report.export", "share", _as("bp-admins"),
                                   policy_engine=engine_with(permit))

    assert decision.allowed is True
    assert decision.unresolved is False
    assert escalations == []


def test_an_explicit_deny_refuses_and_raises_nothing(escalations):
    """A deny is an answer. It is audited, not queued for someone to decide."""

    refuse = _policy("export_ban", "ReportExportPolicy",
                     applies_to=["report.export"],
                     rules={"effect": "deny", "reason": "export is closed"})

    decision = guardrail.authorize("report.export", "share", _as("bp-admins"),
                                   policy_engine=engine_with(refuse))

    assert decision.allowed is False
    assert decision.unresolved is False
    assert escalations == []


def test_a_role_refusal_is_resolved_by_the_role_policy(escalations):
    """RoleDefinitionPolicy spoke. That is an answer, not a deferral."""

    decision = guardrail.authorize("report.export", "share", _as("bp-viewers"),
                                   policy_engine=engine_with())

    assert decision.allowed is False
    assert decision.unresolved is False
    assert escalations == []


def test_a_reversible_action_is_resolved_by_the_role_policy(escalations):
    """Not a silent grant: the role policy defines these classes as reversible."""

    decision = guardrail.authorize("supplier.read", "read", _as("bp-viewers"),
                                   policy_engine=engine_with())

    assert decision.allowed is True
    assert decision.unresolved is False
    assert escalations == []


# --- the compatibility guarantee ------------------------------------------

def test_unresolved_reads_as_not_allowed_for_existing_callers(escalations):
    """Four call sites check `if not decision.allowed`. They must still refuse."""

    weights = _policy("weight_allocation_policy", "WeightAllocationPolicy",
                      applies_to=["email.send"], rules={})

    decision = guardrail.authorize("email.send", "communicate", _as("bp-approvers"),
                                   policy_engine=engine_with(weights))

    assert decision.allowed is False


def test_an_escalation_that_cannot_be_raised_still_refuses(monkeypatch):
    """A question nobody can be asked is not a licence to proceed."""

    monkeypatch.setattr(guardrail.policy_observation, "record", lambda **k: True)

    def _boom(**kwargs):
        raise RuntimeError("bp_decision is unreachable")

    monkeypatch.setattr(guardrail, "_raise_for_a_human", _boom)

    decision = guardrail.authorize("agent.create", "configure", _as("bp-admins"),
                                   policy_engine=engine_with())

    assert decision.allowed is False
