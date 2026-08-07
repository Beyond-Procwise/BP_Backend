"""The five ordered checks, each proven by making it fail.

These are unit tests over the guard, not over SES. Each one removes exactly
one precondition and asserts the send is refused, and the last asserts the
happy path still passes so the guard is not simply always-deny.
"""

import pytest

from src.services import email_dispatch_guard as guard
from tests.guardrails.test_email_sensitivity import SENSITIVITY_POLICY
from tests.guardrails.test_guardrail_gate import GateEngine
from tests.guardrails.test_rbac import (
    ROLE_ASSIGNMENT,
    ROLE_DEFINITION,
    FakePrincipal,
)


ALLOWLIST_POLICY = {
    "policyId": "email_recipient_allowlist",
    "policyName": "EmailRecipientAllowlistPolicy",
    "details": {
        "policy_identifier": "email_recipient_allowlist",
        "required_role": "Approver",
        "applies_to": ["email.send"],
        "rules": {
            "match": "exact_casefold",
            "on_unknown_recipient": "deny_and_raise_review",
            "allow_recipients_from_email_body": False,
        },
    },
    "raw_row": {"version": 1},
}

APPROVAL_POLICY = {
    "policyId": "email_dispatch_approval",
    "policyName": "EmailDispatchApprovalPolicy",
    "details": {
        "policy_identifier": "email_dispatch_approval",
        "required_role": "Approver",
        "applies_to": ["email.send"],
        "rules": {
            "approval_required": True,
            "require_actioned_by": True,
            "trust_input_payload": False,
            "on_missing_approval": "deny",
        },
    },
    "raw_row": {"version": 1},
}

VOLUME_POLICY = {
    "policyId": "email_volume",
    "policyName": "EmailVolumePolicy",
    "details": {
        "policy_identifier": "email_volume",
        "required_role": "Admin",
        "rules": {"max_per_run": 2, "max_per_user_per_day": 50},
    },
    "raw_row": {"version": 1},
}


def engine():
    return GateEngine(
        {
            "role_definition": ROLE_DEFINITION,
            "role_assignment": ROLE_ASSIGNMENT,
            "email_sensitivity": SENSITIVITY_POLICY,
            "email_recipient_allowlist": ALLOWLIST_POLICY,
            "email_dispatch_approval": APPROVAL_POLICY,
            "email_volume": VOLUME_POLICY,
        }
    )


def approver():
    return FakePrincipal("sub-approver", {"cognito:groups": ["bp-approvers"]})


class FakeConn:
    """Answers only the two lookups the guard makes against the database."""

    def __init__(self, allowlist=("buyer@supplier-b.com",), clearance="internal"):
        self.allowlist = {a.casefold() for a in allowlist}
        self.clearance = clearance

    def lookup_supplier_emails(self, supplier_id):
        return set(self.allowlist)

    def lookup_supplier_clearance(self, supplier_id):
        return self.clearance


def base_kwargs(**overrides):
    kwargs = dict(
        conn=FakeConn(),
        draft={
            "unique_id": "PROC-WF-1",
            "rfq_id": "RFQ-1",
            "workflow_id": "WF-1",
            "supplier_id": "SUP-1",
            "recipients": ["buyer@supplier-b.com"],
        },
        recipients=["buyer@supplier-b.com"],
        subject="Request for quotation",
        body="Please quote for 100 units.",
        attachments=None,
        principal=approver(),
        run_count=0,
        policy_engine=engine(),
        approval_lookup=lambda **_: {
            "approval_id": 1,
            "status": "approved",
            "actioned_by": "buyer@ourcompany.com",
        },
        internal_domains=["ourcompany.com"],
        peer_prices=[],
    )
    kwargs.update(overrides)
    return kwargs


def test_happy_path_is_allowed():
    decision = guard.check_dispatch(**base_kwargs())
    assert decision.allowed is True, decision.reason


def test_check_1_no_approval_denies():
    decision = guard.check_dispatch(**base_kwargs(approval_lookup=lambda **_: None))
    assert decision.allowed is False
    assert "approval" in decision.reason.lower()


def test_check_1_input_payload_claiming_approval_is_ignored():
    """Trusting the caller's own word is exactly the hole being closed."""
    draft = dict(base_kwargs()["draft"])
    draft["approved"] = True
    draft["sent_status"] = True
    decision = guard.check_dispatch(
        **base_kwargs(draft=draft, approval_lookup=lambda **_: None)
    )
    assert decision.allowed is False


def test_check_2_recipient_not_on_supplier_master_denies():
    decision = guard.check_dispatch(
        **base_kwargs(recipients=["stranger@elsewhere.com"])
    )
    assert decision.allowed is False
    assert "allow-list" in decision.reason.lower()


def test_check_3_competitor_price_to_internal_supplier_denies():
    decision = guard.check_dispatch(
        **base_kwargs(
            body="Supplier B quoted 12,450.00 for this line.",
            peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
        )
    )
    assert decision.allowed is False
    assert "clearance" in decision.reason.lower()


def test_check_3_cleared_supplier_may_receive_commercial_content():
    decision = guard.check_dispatch(
        **base_kwargs(
            conn=FakeConn(clearance="commercial_confidential"),
            body="Supplier B quoted 12,450.00 for this line.",
            peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
        )
    )
    assert decision.allowed is True, decision.reason


def test_check_4_viewer_is_denied():
    decision = guard.check_dispatch(
        **base_kwargs(
            principal=FakePrincipal("sub-v", {"cognito:groups": ["bp-viewers"]})
        )
    )
    assert decision.allowed is False


def test_check_4_no_principal_is_denied():
    decision = guard.check_dispatch(**base_kwargs(principal=None))
    assert decision.allowed is False
    assert "no authenticated principal" in decision.reason


def test_check_5_run_cap_denies():
    decision = guard.check_dispatch(**base_kwargs(run_count=2))
    assert decision.allowed is False
    assert "volume" in decision.reason.lower()


def test_stored_draft_is_authoritative_for_recipients():
    draft = {"recipients": ["buyer@supplier-b.com"]}
    assert guard.resolve_recipients(draft, None) == ["buyer@supplier-b.com"]


def test_caller_may_narrow_recipients_but_not_add():
    draft = {"recipients": ["a@supplier-b.com", "b@supplier-b.com"]}
    assert guard.resolve_recipients(draft, ["a@supplier-b.com"]) == ["a@supplier-b.com"]
    assert guard.resolve_recipients(draft, ["new@elsewhere.com"]) == []


def test_peer_prices_come_from_the_approval_not_the_draft():
    """The draft has no deal_id. Taking it from there kills the detector."""

    class DealAwareConn(FakeConn):
        def lookup_peer_prices(self, deal_id, supplier_id):
            assert deal_id == "DEAL-9", f"guard looked up deal {deal_id!r}"
            return [{"supplier_id": "SUP-2", "amount": "12450.00"}]

    decision = guard.check_dispatch(
        **base_kwargs(
            conn=DealAwareConn(),
            body="Supplier B quoted 12,450.00 for this line.",
            peer_prices=None,  # production path: the guard must resolve them
            approval_lookup=lambda **_: {
                "approval_id": 1,
                "status": "approved",
                "actioned_by": "buyer@ourcompany.com",
                "deal_id": "DEAL-9",
            },
        )
    )
    assert decision.allowed is False
    assert "clearance" in decision.reason.lower()
