"""The five ordered checks, each proven by making it fail.

These are unit tests over the guard, not over SES. Each one removes exactly
one precondition and asserts the send is refused, and the last asserts the
happy path still passes so the guard is not simply always-deny.
"""

import pytest

from src.services import email_dispatch_guard as guard
from src.services.approval_content import content_hash
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
    """Answers the lookups the guard makes against the database."""

    def __init__(
        self,
        allowlist=("buyer@supplier-b.com",),
        clearance="internal",
        daily_send_count=0,
    ):
        self.allowlist = {a.casefold() for a in allowlist}
        self.clearance = clearance
        self.daily_send_count = daily_send_count

    def lookup_supplier_emails(self, supplier_id):
        return set(self.allowlist)

    def lookup_supplier_clearance(self, supplier_id):
        return self.clearance

    def lookup_daily_send_count(self, principal_subject):
        return self.daily_send_count


BASE_DRAFT = {
    "unique_id": "PROC-WF-1",
    "rfq_id": "RFQ-1",
    "workflow_id": "WF-1",
    "supplier_id": "SUP-1",
    "recipients": ["buyer@supplier-b.com"],
}

# The approval hash is bound to BASE_DRAFT, not to the `body`/`subject`/
# `recipients` kwargs individual tests below pass to check_dispatch --
# content_hash reads them off the draft mapping itself, which none of these
# tests mutate (test_check_1_input_payload_claiming_approval_is_ignored is
# the one exception, and it supplies its own approval_lookup returning None,
# so it never reaches the hash check).
BASE_APPROVED_HASH = content_hash(BASE_DRAFT)


def base_kwargs(**overrides):
    kwargs = dict(
        conn=FakeConn(),
        draft=dict(BASE_DRAFT),
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
            "grounding": {"content_hash": BASE_APPROVED_HASH},
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
                "grounding": {"content_hash": BASE_APPROVED_HASH},
            },
        )
    )
    assert decision.allowed is False
    assert "clearance" in decision.reason.lower()


SIGNATURE_BODY = (
    "Please quote for 100 units.\n\n"
    "Kind regards,\n"
    "Jane Doe\n"
    "Procurement Manager\n"
    "jane.doe@ourcompany.com"
)


def test_the_senders_own_signature_does_not_block_the_send():
    """The guard must hand classify() the sender, or every signed email is
    read as leaking an internal contact and refused to every supplier."""
    decision = guard.check_dispatch(
        **base_kwargs(
            body=SIGNATURE_BODY,
            sender="jane.doe@ourcompany.com",
            internal_domains=["ourcompany.com"],
        )
    )
    assert decision.allowed is True, decision.reason


def test_a_colleagues_address_still_blocks_the_send():
    decision = guard.check_dispatch(
        **base_kwargs(
            body=SIGNATURE_BODY,
            sender="someone.else@ourcompany.com",
            internal_domains=["ourcompany.com"],
        )
    )
    assert decision.allowed is False
    assert "clearance" in decision.reason.lower()


def test_check_3_a_failed_peer_price_lookup_denies_rather_than_passes():
    """_daily_send_count already distinguishes "no principal" from "the
    lookup failed" by returning None for both and denying on None.
    _peer_prices used to conflate its own two cases -- [] meant BOTH "no
    competing quote on file" (proceed) and "the query exploded" (should
    deny) -- so a broken bp_quote_trgt query silently downgraded the
    sensitivity check to a pass on the highest-value leak it exists to
    catch. A genuine empty peer list must still proceed; see
    test_check_3_cleared_supplier_may_receive_commercial_content and the
    happy path for that direction.

    No ``lookup_peer_prices`` hook is defined on the fake connection here --
    that would force the test-double branch, whose exception the OLD code
    let escape uncaught, and which happened to get denied anyway by
    check_dispatch's own generic catch-all (a false proof: it denies either
    way, for the wrong reason). Instead this forces the REAL production
    branch -- ``cursor()`` on the raw SQL path -- to fail, exactly like a
    broken bp_quote_trgt query would.
    """

    class ExplodingPeerPrices(FakeConn):
        def cursor(self):
            raise RuntimeError("bp_quote_trgt is unreachable")

    decision = guard.check_dispatch(
        **base_kwargs(
            conn=ExplodingPeerPrices(),
            peer_prices=None,  # force the guard to call _peer_prices itself
            approval_lookup=lambda **_: {
                "approval_id": 1,
                "status": "approved",
                "actioned_by": "buyer@ourcompany.com",
                "deal_id": "DEAL-9",
                "grounding": {"content_hash": BASE_APPROVED_HASH},
            },
        )
    )
    assert decision.allowed is False
    assert "competing quotes" in decision.reason.lower()
    assert decision.policy_name == "EmailSensitivityPolicy"


def test_check_5_daily_cap_denies_when_the_lookup_is_at_or_over_the_limit():
    """max_per_user_per_day is declared by VOLUME_POLICY (50) and must be
    enforced, not merely read. A lookup returning >= the limit denies."""
    decision = guard.check_dispatch(
        **base_kwargs(conn=FakeConn(daily_send_count=50))
    )
    assert decision.allowed is False
    assert "daily" in decision.reason.lower()


def test_check_5_daily_cap_allows_comfortably_under_the_limit():
    decision = guard.check_dispatch(
        **base_kwargs(conn=FakeConn(daily_send_count=1))
    )
    assert decision.allowed is True, decision.reason


def test_denials_carry_policy_attribution_not_just_a_name():
    """G8 requires the audited policy's id and its version, not merely a
    display name. Before this fix, checks 1, 2, 3 and 5 built their Decision
    with policy_name only -- policy_id was always None, and policy_version
    was None unless check 4 (the only check that ever called
    guardrail.authorize) happened to decide the outcome.
    """
    d1 = guard.check_dispatch(**base_kwargs(approval_lookup=lambda **_: None))
    assert d1.policy_id == "email_dispatch_approval"
    assert d1.policy_version == 1

    d2 = guard.check_dispatch(**base_kwargs(recipients=["stranger@elsewhere.com"]))
    assert d2.policy_id == "email_recipient_allowlist"
    assert d2.policy_version == 1

    d3 = guard.check_dispatch(
        **base_kwargs(
            body="Supplier B quoted 12,450.00 for this line.",
            peer_prices=[{"supplier_id": "SUP-2", "amount": "12450.00"}],
        )
    )
    assert d3.policy_id == "email_sensitivity"
    assert d3.policy_version == 1

    d5 = guard.check_dispatch(**base_kwargs(run_count=2))
    assert d5.policy_id == "email_volume"
    assert d5.policy_version == 1


def test_check_5_a_failed_daily_count_lookup_denies_rather_than_passes():
    """An unenforceable cap is not an absent cap: a broken lookup must deny,
    not silently let the send through unmetered."""

    class ExplodingDailyCount(FakeConn):
        def lookup_daily_send_count(self, principal_subject):
            raise RuntimeError("bp_agent_actions is unreachable")

    decision = guard.check_dispatch(**base_kwargs(conn=ExplodingDailyCount()))
    assert decision.allowed is False
    assert "daily" in decision.reason.lower() or "denying" in decision.reason.lower()
