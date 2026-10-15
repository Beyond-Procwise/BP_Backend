"""The `inspect` action class, proved against the real policy rows (bp_testdb).

mailbox.view must reach Admin and Security reviewer only; `read` is granted to every role so it
could not do that. mailbox.bind (configure) must stay Admin-only.
"""
import os
import pytest

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB")

from src.services import actions, guardrail, rbac


class P:
    def __init__(self, group):
        self.subject = f"test-{group}"
        self.claims = {"cognito:groups": [group]}


GROUPS = {
    "Admin": "PROCWISE_ADMIN",
    "SecurityReviewer": "PROCWISE_SECURITY_REVIEWER",
    "Viewer": "PROCWISE_VIEWER",
    "Buyer": "PROCWISE_CATEGORY_MANAGER",
    "Approver": "PROCWISE_CHIEF_PROCUREMENT_OFFICER",
}


@pytest.fixture(autouse=True)
def _fresh_policy():
    rbac.reset_policy_cache()
    yield
    rbac.reset_policy_cache()


def _decide(action, role):
    return guardrail._evaluate(action, actions.action_class(action), P(GROUPS[role]), {})


def test_groups_resolve_to_the_expected_roles():
    for role, group in GROUPS.items():
        assert rbac.effective_role(P(group)) == role


@pytest.mark.parametrize("role", ["Admin", "SecurityReviewer"])
def test_inspectors_may_view_the_mailbox(role):
    d = _decide("mailbox.view", role)
    assert d.allowed, d.reason


@pytest.mark.parametrize("role", ["Viewer", "Buyer", "Approver"])
def test_everyone_else_is_refused_the_mailbox_view(role):
    d = _decide("mailbox.view", role)
    assert not d.allowed and not d.unresolved, d.reason


def test_a_caller_with_no_principal_is_refused():
    d = guardrail._evaluate("mailbox.view", "inspect", None, {})
    assert not d.allowed


def test_binding_stays_admin_only():
    assert _decide("mailbox.bind", "Admin").allowed
    for role in ("SecurityReviewer", "Viewer", "Buyer", "Approver"):
        assert not _decide("mailbox.bind", role).allowed, role


def test_security_reviewer_cannot_write_or_approve_internal_mail():
    for action in ("mail.internal.create", "mail.internal.approve"):
        assert not _decide(action, "SecurityReviewer").allowed


@pytest.mark.parametrize("action", ["supplier.read", "invoice.read", "email.draft.read", "contract.read"])
def test_security_reviewer_sees_no_commercial_data(action):
    from src.services import actions as a
    if not a.is_known(action):
        pytest.skip(f"{action} not in the vocabulary")
    assert a.action_class(action) == "read"
    assert not _decide(action, "SecurityReviewer").allowed


@pytest.mark.xfail(strict=True, reason="rank tie with Viewer; no safe rank exists (see report, fix round 1 item 2)")
def test_reviewer_and_viewer_groups_resolve_the_same_whichever_order():
    a = rbac.effective_role(type("Q", (), {"subject": "x", "claims": {"cognito:groups": ["PROCWISE_VIEWER", "PROCWISE_SECURITY_REVIEWER"]}})())
    b = rbac.effective_role(type("Q", (), {"subject": "x", "claims": {"cognito:groups": ["PROCWISE_SECURITY_REVIEWER", "PROCWISE_VIEWER"]}})())
    assert a == b
