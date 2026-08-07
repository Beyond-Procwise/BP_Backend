"""The gate must be provably closed before it is trusted.

Each test here removes one thing the gate depends on and asserts it denies.
A gate that only ever says yes has not been shown to work.
"""

import pytest

from src.services import guardrail
from tests.guardrails.test_rbac import (
    ROLE_ASSIGNMENT,
    ROLE_DEFINITION,
    FakePolicyEngine,
    FakePrincipal,
)


ALLOW_SEND = {
    "policyId": "email_dispatch_approval",
    "policyName": "EmailDispatchApprovalPolicy",
    "details": {
        "policy_identifier": "email_dispatch_approval",
        "required_role": "Approver",
        "applies_to": ["email.send"],
        "rules": {"approval_required": True},
    },
    "raw_row": {"version": 1},
}


class GateEngine(FakePolicyEngine):
    """Adds the action lookup the gate needs."""

    def policies_for_action(self, action):
        return [
            p
            for p in self._policies.values()
            if action in (p.get("details", {}).get("applies_to") or [])
        ]


def engine_with(*extra):
    policies = {
        "role_definition": ROLE_DEFINITION,
        "role_assignment": ROLE_ASSIGNMENT,
    }
    for policy in extra:
        policies[policy["policyId"]] = policy
    return GateEngine(policies)


def approver():
    return FakePrincipal("sub-approver", {"cognito:groups": ["bp-approvers"]})


def viewer():
    return FakePrincipal("sub-viewer", {"cognito:groups": ["bp-viewers"]})


def test_approver_with_an_allowing_policy_is_permitted():
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with(ALLOW_SEND)
    )
    assert decision.allowed is True
    assert decision.policy_name == "EmailDispatchApprovalPolicy"


def test_viewer_may_not_communicate():
    decision = guardrail.authorize(
        "email.send", "communicate", viewer(), {}, policy_engine=engine_with(ALLOW_SEND)
    )
    assert decision.allowed is False
    assert "Viewer" in decision.reason


def test_no_principal_is_denied_an_irreversible_action():
    decision = guardrail.authorize(
        "email.send", "communicate", None, {}, policy_engine=engine_with(ALLOW_SEND)
    )
    assert decision.allowed is False
    assert "no authenticated principal" in decision.reason


def test_irreversible_action_with_no_policy_is_denied():
    """Default-deny: silence is not permission."""
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with()
    )
    assert decision.allowed is False
    assert "no policy" in decision.reason.lower()


def test_reversible_action_with_no_policy_is_allowed():
    decision = guardrail.authorize(
        "deal.read", "read", viewer(), {}, policy_engine=engine_with()
    )
    assert decision.allowed is True


def test_deny_beats_allow():
    denying = {
        "policyId": "email_block",
        "policyName": "EmailBlockPolicy",
        "details": {
            "policy_identifier": "email_block",
            "required_role": "Approver",
            "applies_to": ["email.send"],
            "rules": {"effect": "deny", "reason": "dispatch frozen"},
        },
        "raw_row": {"version": 1},
    }
    decision = guardrail.authorize(
        "email.send",
        "communicate",
        approver(),
        {},
        policy_engine=engine_with(ALLOW_SEND, denying),
    )
    assert decision.allowed is False
    assert "dispatch frozen" in decision.reason


def test_required_role_on_the_policy_is_enforced():
    admin_only = {
        "policyId": "email_admin_only",
        "policyName": "EmailAdminOnlyPolicy",
        "details": {
            "policy_identifier": "email_admin_only",
            "required_role": "Admin",
            "applies_to": ["email.send"],
            "rules": {},
        },
        "raw_row": {"version": 1},
    }
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with(admin_only)
    )
    assert decision.allowed is False
    assert "Admin" in decision.reason


def test_an_exploding_engine_denies_rather_than_raises():
    class Exploding:
        def get_policy(self, slug):
            raise RuntimeError("database is gone")

        def policies_for_action(self, action):
            raise RuntimeError("database is gone")

    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=Exploding()
    )
    assert decision.allowed is False
    assert decision.evidence.get("error")


def test_decision_carries_the_policy_version_for_audit():
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with(ALLOW_SEND)
    )
    assert decision.policy_version == 1


def test_the_real_policy_rows_can_actually_open_the_gate():
    """A fixture richer than production data hides an unopenable gate.

    policies_for_action reads details->'applies_to'. If no live row carries it,
    authorize() default-denies forever and no policy edit can change that.
    """
    import os

    import psycopg2
    from dotenv import load_dotenv

    from src.engines.policy_engine import PolicyEngine

    load_dotenv()

    def factory():
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=10,
        )

    engine = PolicyEngine(connection_factory=factory)
    matched = engine.policies_for_action("email.send")
    assert matched, (
        "no live policy row declares applies_to ['email.send'] -- the gate "
        "cannot be opened by any caller or any policy edit"
    )


def test_a_real_decision_carries_the_policy_version():
    """G8 requires the audited policy's version, and a fixture cannot prove it.

    The unit fixtures set raw_row["version"] by hand, so they pass whether or
    not the loader actually selects the column.
    """
    import os

    import psycopg2
    from dotenv import load_dotenv

    from src.engines.policy_engine import PolicyEngine

    load_dotenv()

    def factory():
        return psycopg2.connect(
            host=os.getenv("DB_HOST"),
            port=os.getenv("DB_PORT", 5432),
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            connect_timeout=10,
        )

    class Approver:
        subject = "sub-approver"
        claims = {"cognito:groups": ["bp-approvers"]}

    engine = PolicyEngine(connection_factory=factory)
    decision = guardrail.authorize(
        "email.send", "communicate", Approver(), {}, policy_engine=engine
    )

    assert decision.allowed is True, decision.reason
    assert isinstance(decision.policy_version, int), (
        "policy_version is None against the live database -- the loader is not "
        "selecting bp_policy.version, so every audit row would record no version"
    )


@pytest.mark.parametrize("bad_context", ["oops", 42, ["a", "b"], object()])
def test_a_malformed_context_never_escapes_as_an_exception(bad_context):
    """A gate that raises is a gate the caller reads as 'no denial'."""
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), bad_context,
        policy_engine=engine_with(ALLOW_SEND),
    )
    assert isinstance(decision, guardrail.Decision)


@pytest.mark.parametrize("bad_role", ["Admiin", "admin", "SuperAdmin", "  "])
def test_an_unresolvable_required_role_denies(bad_role):
    """A typo in a policy row must not silently remove its own restriction."""
    policy = {
        "policyId": "email_admin_only",
        "policyName": "EmailAdminOnlyPolicy",
        "details": {
            "policy_identifier": "email_admin_only",
            "required_role": bad_role,
            "applies_to": ["email.send"],
            "rules": {},
        },
        "raw_row": {"version": 1},
    }
    decision = guardrail.authorize(
        "email.send", "communicate", approver(), {}, policy_engine=engine_with(policy)
    )
    assert decision.allowed is False
    assert "no policy defines" in decision.reason
