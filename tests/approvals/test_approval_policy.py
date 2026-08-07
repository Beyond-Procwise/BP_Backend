"""The capability that lets a person approve an email is a policy row, not code.

A customer who wants email approval restricted to Approvers changes one row.
These read the LIVE rows: a fixture richer than the database is how this
project has repeatedly shipped a guard that was already dead.
"""

import os

import psycopg2
import pytest
from dotenv import load_dotenv

from src.engines.policy_engine import PolicyEngine
from src.services import rbac

load_dotenv()


def _factory():
    return psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=10,
    )


@pytest.fixture(scope="module")
def engine():
    return PolicyEngine(connection_factory=_factory)


def test_buyer_and_above_may_approve_an_email(engine):
    assert rbac.may("Buyer", "approve_email", policy_engine=engine) is True
    assert rbac.may("Approver", "approve_email", policy_engine=engine) is True
    assert rbac.may("Admin", "approve_email", policy_engine=engine) is True


def test_a_viewer_may_not_approve_an_email(engine):
    assert rbac.may("Viewer", "approve_email", policy_engine=engine) is False


def test_approving_an_email_is_not_approving_money(engine):
    """The two capabilities are deliberately distinct, not one rank doing both."""
    assert rbac.may("Buyer", "transact", policy_engine=engine) is False
    assert rbac.may("Approver", "transact", policy_engine=engine) is True


def test_approve_email_is_listed_as_irreversible_in_policy(engine):
    """Assert the policy row directly.

    Going through rbac.is_irreversible proves nothing here: it fail-closes on
    anything absent from reversible_classes, so it returns True whether or not
    this migration ever ran. The point of this test is that the capability was
    explicitly classified, so read the classification.
    """
    rules = engine.get_policy("role_definition")["details"]["rules"]
    assert "approve_email" in rules["irreversible_classes"]


def test_the_approval_action_resolves_to_a_policy(engine):
    matched = engine.policies_for_action("approval.email")
    assert matched, "no live policy declares applies_to ['approval.email']"
    assert any(
        (p.get("details") or {}).get("required_role") == "Buyer" for p in matched
    )


def test_dispatch_approval_requires_buyer_and_denies_on_content_mismatch(engine):
    policy = engine.get_policy("email_dispatch_approval")
    assert policy is not None
    details = policy["details"]
    assert details["required_role"] == "Buyer"
    assert details["rules"]["on_content_mismatch"] == "deny"
