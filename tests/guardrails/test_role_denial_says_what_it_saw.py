"""A refusal for lack of role records what the role was resolved FROM.

Found live 2026-10-12: the product's own administrator opened "Emails to approve"
and was refused as a Viewer. The record said "role Viewer may not perform
approve_email" and nothing else, so nobody could tell whether the sign-in had
carried the wrong groups, no groups at all, or groups the mapping did not know.
(It carried none.) The names of the groups on a token are not secret, and they are
the whole answer to "why am I a Viewer?".
"""
from types import SimpleNamespace

import pytest

from src.services import guardrail, rbac


class _Engine:
    """Roles and the group mapping, as the two policy rows state them."""

    ROWS = {
        "role_assignment": {"claim": "cognito:groups", "no_principal_role": "Viewer", "unmapped_group_role": "Viewer",
                            "multiple_groups": "highest_rank",
                            "group_to_role": {"PROCWISE_ADMIN": "Admin", "PROCWISE_VIEWER": "Viewer"}},
        "role_definition": {"roles": {"Admin": {"rank": 4, "allow": ["read", "approve_email"]},
                                      "Viewer": {"rank": 1, "allow": ["read"]}},
                            "reversible_classes": ["read"], "irreversible_classes": ["approve_email"]},
    }

    def get_policy(self, slug):
        rules = self.ROWS.get(slug)
        return {"details": {"rules": rules}} if rules else None

    def policies_for_action(self, action):
        return []


@pytest.fixture(autouse=True)
def _no_direct_grants(monkeypatch):
    monkeypatch.setattr(rbac, "_direct_roles", lambda subject: [])


def _decide(claims):
    principal = SimpleNamespace(subject="sub-1", claims=claims)
    return guardrail.authorize("approval.email", "approve_email", principal, {}, policy_engine=_Engine())


def test_a_sign_in_with_no_groups_says_so():
    decision = _decide({})
    assert decision.allowed is False and "role Viewer may not perform" in decision.reason
    assert decision.evidence["role_from"] == {"groups": [], "direct_grants": []}


def test_groups_the_mapping_does_not_know_are_named():
    # Not `procwise_admin`: group names match whatever the case, so that one IS known.
    decision = _decide({"cognito:groups": ["procwise_auditors", "Everyone"]})
    assert decision.allowed is False
    assert decision.evidence["role_from"]["groups"] == ["procwise_auditors", "Everyone"]


def test_a_group_in_another_case_is_known_not_refused():
    decision = _decide({"cognito:groups": ["procwise_admin"]})
    assert decision.evidence.get("role_from") is None
    assert "role Viewer may not perform" not in decision.reason


def test_a_known_low_group_is_named_too():
    assert _decide({"cognito:groups": ["PROCWISE_VIEWER"]}).evidence["role_from"]["groups"] == ["PROCWISE_VIEWER"]


def test_a_direct_grant_that_was_not_enough_is_named(monkeypatch):
    monkeypatch.setattr(rbac, "_direct_roles", lambda subject: ["Viewer"])
    assert _decide({}).evidence["role_from"]["direct_grants"] == ["Viewer"]


def test_nothing_else_from_the_token_is_recorded():
    decision = _decide({"cognito:groups": ["PROCWISE_VIEWER"], "email": "someone@example.com", "custom:secret": "x"})
    assert set(decision.evidence["role_from"]) == {"groups", "direct_grants"}
    assert "someone@example.com" not in str(decision.evidence)


def test_a_role_that_is_enough_is_unaffected():
    assert _decide({"cognito:groups": ["PROCWISE_ADMIN"]}).evidence.get("role_from") is None
