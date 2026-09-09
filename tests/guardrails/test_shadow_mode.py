"""Shadow mode: find out what a rule would do, before it does it.

Sixteen of nineteen policies are invisible to the gate because they carry no
``applies_to``. Making them visible turns dormant rules into refusals, and
nobody currently knows what would be refused or to whom. Shadow mode answers
that: evaluate exactly as normal, record the verdict, and — for actions
explicitly enrolled, with an expiry — allow anyway.

It is a deliberate hole, so these tests spend most of their effort on the edges
rather than the happy path. The one that matters is
``test_a_non_shadowed_denial_is_still_denied``: shadow mode must not leak.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.services import guardrail
from tests.guardrails.test_guardrail_gate import GateEngine, engine_with
from tests.guardrails.test_rbac import FakePrincipal


def _as(group):
    """A principal in one Cognito group, the shape rbac resolves roles from."""

    return FakePrincipal(f"sub-{group}", {"cognito:groups": [group]})


def _future() -> str:
    return (datetime.now(timezone.utc) + timedelta(days=7)).isoformat()


def _past() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()


def _shadow_policy(entries):
    return {
        "policyId": "shadow_mode",
        "policyName": "ShadowModePolicy",
        "details": {
            "policy_identifier": "shadow_mode",
            "required_role": "Admin",
            "rules": {"shadow_actions": entries},
        },
        "raw_row": {"version": 1},
    }


@pytest.fixture
def observations(monkeypatch):
    """Capture what the gate records instead of writing to the database."""

    captured = []

    def _record(**fields):
        captured.append(fields)
        return True

    monkeypatch.setattr(guardrail.policy_observation, "record", _record)
    return captured


DENY_READ = {
    "policyId": "supplier_read_ban",
    "policyName": "SupplierReadPolicy",
    "details": {
        "policy_identifier": "supplier_read_ban",
        "applies_to": ["supplier.read", "report.export"],
        "rules": {"effect": "deny", "reason": "not permitted yet"},
    },
    "raw_row": {"version": 1},
}


def _engine(entries):
    """A gate that REFUSES supplier.read and report.export by a stated rule.

    Shadow mode only ever softens a rule that said no. A deferral -- nobody's
    rule spoke -- is never shadowed, so these tests need a real denial to shadow
    rather than the default-deny they used to rely on.
    """

    return engine_with(_shadow_policy(entries), DENY_READ)


def test_a_shadowed_denial_is_allowed_and_recorded_as_would_have_denied(observations):
    """The whole point: it does not refuse, but it writes down that it would have."""

    decision = guardrail.authorize(
        "supplier.read",
        "share",  # irreversible, no policy permits it -> default deny
        _as("bp-admins"),
        policy_engine=_engine([{"action": "supplier.read", "until": _future()}]),
    )

    assert decision.allowed is True
    assert "shadow" in decision.reason.lower()

    assert len(observations) == 1
    row = observations[0]
    assert row["action"] == "supplier.read"
    assert row["would_have_denied"] is True
    assert row["shadowed"] is True


def test_a_non_shadowed_denial_is_still_denied(observations):
    """Shadow mode must not leak into actions nobody enrolled.

    This is the test that matters. Everything else here is about making the
    hole safe; this one is about the hole staying where it was put.
    """

    decision = guardrail.authorize(
        "report.export",
        "share",
        _as("bp-admins"),
        policy_engine=_engine([{"action": "supplier.read", "until": _future()}]),
    )

    assert decision.allowed is False
    assert observations[0]["shadowed"] is False
    assert observations[0]["would_have_denied"] is True


def test_email_send_can_never_be_shadowed(observations):
    """Enrolling it must not work. This is enforced in code, not by convention."""

    decision = guardrail.authorize(
        "email.send",
        "communicate",
        None,  # no principal: irreversible actions are refused
        policy_engine=_engine([{"action": "email.send", "until": _future()}]),
    )

    assert decision.allowed is False
    assert observations[0]["shadowed"] is False


def test_approval_email_can_never_be_shadowed(observations):
    decision = guardrail.authorize(
        "approval.email",
        "approve_email",
        None,
        policy_engine=_engine([{"action": "approval.email", "until": _future()}]),
    )

    assert decision.allowed is False


def test_an_expired_enrolment_enforces_again(observations):
    """Shadow mode must not become permanent by nobody getting round to it."""

    decision = guardrail.authorize(
        "supplier.read",
        "share",
        _as("bp-admins"),
        policy_engine=_engine([{"action": "supplier.read", "until": _past()}]),
    )

    assert decision.allowed is False
    assert observations[0]["shadowed"] is False


def test_an_enrolment_without_an_expiry_is_not_shadowed(observations):
    """A missing expiry is not an unlimited one."""

    decision = guardrail.authorize(
        "supplier.read",
        "share",
        _as("bp-admins"),
        policy_engine=_engine([{"action": "supplier.read"}]),
    )

    assert decision.allowed is False


def test_allow_decisions_are_recorded_too(observations):
    """Otherwise "we observed no denials" is indistinguishable from "we were not observing"."""

    decision = guardrail.authorize(
        "deal.read",
        "read",  # reversible, and no policy denies this one
        _as("bp-viewers"),
        policy_engine=_engine([]),
    )

    assert decision.allowed is True
    assert len(observations) == 1
    assert observations[0]["would_have_denied"] is False


def test_a_shadowed_denial_that_cannot_be_recorded_still_denies(monkeypatch):
    """No record, no shadow.

    Allowing without recording gives neither the safety of the refusal nor the
    data the refusal was traded for. If the observation cannot be written, the
    real decision stands.
    """

    monkeypatch.setattr(
        guardrail.policy_observation, "record", lambda **fields: False
    )

    decision = guardrail.authorize(
        "supplier.read",
        "share",
        _as("bp-admins"),
        policy_engine=_engine([{"action": "supplier.read", "until": _future()}]),
    )

    assert decision.allowed is False


def test_recording_never_breaks_the_gate(monkeypatch):
    """An exploding recorder must not turn a decision into a crash."""

    def _boom(**fields):
        raise RuntimeError("observation table is gone")

    monkeypatch.setattr(guardrail.policy_observation, "record", _boom)

    decision = guardrail.authorize(
        "deal.read",
        "read",
        _as("bp-viewers"),
        policy_engine=_engine([]),
    )

    assert decision.allowed is True


def test_shadow_config_absent_means_nothing_is_shadowed(observations):
    """No policy row, no shadowing. The default is to enforce."""

    decision = guardrail.authorize(
        "supplier.read",
        "share",
        _as("bp-admins"),
        policy_engine=engine_with(),
    )

    assert decision.allowed is False


def test_a_deferral_is_never_shadowed_even_when_enrolled(observations):
    """Shadow mode softens a refusal. It must not soften a question.

    Allowing something through while asking whether it is allowed grants the
    very thing in question, and the answer arrives after the fact.
    """

    decision = guardrail.authorize(
        "agent.create",  # nothing speaks to it -> unresolved, not denied
        "configure",
        _as("bp-admins"),
        policy_engine=_engine([{"action": "agent.create", "until": _future()}]),
    )

    assert decision.allowed is False
    assert decision.unresolved is True
    assert observations[0]["shadowed"] is False
