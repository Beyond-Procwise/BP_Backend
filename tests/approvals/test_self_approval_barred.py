"""Nobody approves their own request.

`proc.bp_approval.actioned_by` has been the authenticated principal since the
surface was rebuilt, so an approval cannot be forged onto someone else. What was
never checked is the other direction: whether the person signing is the person
who asked. The August migration i6_fix_remove_self_approval_allowed deleted an
unused policy key and nothing replaced it, so between then and now the rule
agreed in M18.72 existed only in a decision record.

It is reachable, not theoretical: `POST /workflows/email/prepare` lets a person
persist an edited email as a draft, and `POST /approvals/dispatch/{unique_id}`
lets a person approve one. The same person could do both.

The requester is read from the stored draft and never from the request body,
for the same reason the approver is: a caller-supplied actor is a forgery with
extra steps.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import approvals as approvals_router

REQUESTER = "sub-buyer-001"
SOMEONE_ELSE = "sub-approver-002"


class _PrincipalAs:
    def __init__(self, subject):
        self.subject = subject
        self.email = f"{subject}@ourcompany.com"
        self.claims = {"cognito:groups": []}


@pytest.fixture(autouse=True)
def _stub_audit_write(monkeypatch):
    """No real bp_agent_actions rows from a router unit test. Tests that care
    about the audit row install their own spy, which wins."""
    monkeypatch.setattr(approvals_router, "record_action_or_fail", lambda **kwargs: None)


@pytest.fixture(autouse=True)
def _capability_allows(monkeypatch):
    """The capability check is not what is under test here; it passes."""
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(
            allowed=True, reason="ok", policy_id="email_approval_capability",
            policy_name="EmailApprovalCapabilityPolicy", policy_version=4),
    )


def _policy(self_approval, monkeypatch):
    """Install a capability policy whose rules say what this test needs.

    ``self_approval=_ABSENT`` leaves the key out entirely, which is the state
    every deployment is in until the migration runs.
    """
    rules = {"effect": "allow", "revoke_scope": "own_or_higher_rank"}
    if self_approval is not _ABSENT:
        rules["self_approval"] = self_approval
    policy = {
        "policyName": "EmailApprovalCapabilityPolicy",
        "details": {"policy_identifier": "email_approval_capability", "rules": rules},
        "raw_row": {"policy_id": 704, "version": 4},
    }
    monkeypatch.setattr(approvals_router, "_capability_policy", lambda engine: policy)
    monkeypatch.setattr(approvals_router.rbac, "policy_engine", lambda *a, **k: object())


_ABSENT = object()


def _draft(monkeypatch, requested_by):
    monkeypatch.setattr(
        approvals_router, "_load_draft",
        lambda uid, conn=None: {"unique_id": uid, "rfq_id": "RFQ-1",
                                "supplier_id": "SUP-1", "workflow_id": "WF-1",
                                "subject": "s", "body": "b",
                                "requested_by": requested_by},
    )


def _recorded(monkeypatch):
    """Spy on the approval write, so 'refused' means no row rather than a 4xx
    with a signature already in the table."""
    seen = {}

    def fake_record(**kwargs):
        seen.update(kwargs)
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    return seen


def _client_as(subject):
    app = FastAPI()
    app.include_router(approvals_router.router)
    app.dependency_overrides[approvals_router.require_user] = lambda: _PrincipalAs(subject)
    return TestClient(app)


# ---------------------------------------------------------------------------
# the bar
# ---------------------------------------------------------------------------
def test_the_requester_cannot_approve_their_own_draft(monkeypatch):
    """The finding itself. This currently returns 200 and records a signature."""
    _policy("deny", monkeypatch)
    _draft(monkeypatch, requested_by=REQUESTER)
    written = _recorded(monkeypatch)

    response = _client_as(REQUESTER).post("/approvals/dispatch/UID-1", json={})

    assert response.status_code == 403, (
        f"a person approved their own request: {response.status_code} {response.text}")
    assert not written, "the approval was recorded despite being refused"


def test_the_refusal_explains_itself_and_says_what_to_do(monkeypatch):
    """A 403 that reads like a permissions bug sends someone to an
    administrator to have their role widened, which is the wrong fix. The
    message has to name the reason and the remedy."""
    _policy("deny", monkeypatch)
    _draft(monkeypatch, requested_by=REQUESTER)
    _recorded(monkeypatch)

    detail = str(_client_as(REQUESTER).post(
        "/approvals/dispatch/UID-1", json={}).json().get("detail", "")).lower()

    assert "requested" in detail, detail          # the reason
    assert "someone else" in detail, detail       # the remedy


def test_the_refusal_is_audited_with_the_policy_that_made_it(monkeypatch):
    """Same shape as the rest of the guardrail: a Decision carrying policy_id,
    policy_name and policy_version, written to bp_agent_actions."""
    _policy("deny", monkeypatch)
    _draft(monkeypatch, requested_by=REQUESTER)
    _recorded(monkeypatch)
    rows = []
    monkeypatch.setattr(approvals_router, "record_action_or_fail",
                        lambda **kwargs: rows.append(kwargs))

    _client_as(REQUESTER).post("/approvals/dispatch/UID-1", json={})

    denials = [r for r in rows if r.get("status") == "denied"]
    assert denials, f"the refusal produced no audit row: {rows}"
    details = denials[-1].get("details") or {}
    assert details.get("policy_name") == "EmailApprovalCapabilityPolicy", details
    assert details.get("policy_version") == 4, details
    assert details.get("policy_id") is not None, details


def test_somebody_else_may_approve_it(monkeypatch):
    """The bar must be narrow. Barring self-approval and barring approval are
    not the same thing."""
    _policy("deny", monkeypatch)
    _draft(monkeypatch, requested_by=REQUESTER)
    written = _recorded(monkeypatch)

    response = _client_as(SOMEONE_ELSE).post("/approvals/dispatch/UID-1", json={})

    assert response.status_code == 200, response.text
    assert written.get("actioned_by") == SOMEONE_ELSE


def test_a_draft_nobody_requested_is_still_approvable(monkeypatch):
    """Agents draft most of these. An agent-created draft has no human
    requester to collide with, and must not become unapprovable."""
    _policy("deny", monkeypatch)
    _draft(monkeypatch, requested_by=None)
    written = _recorded(monkeypatch)

    response = _client_as(REQUESTER).post("/approvals/dispatch/UID-1", json={})

    assert response.status_code == 200, response.text
    assert written.get("actioned_by") == REQUESTER


# ---------------------------------------------------------------------------
# the rule is policy, and a missing rule denies
# ---------------------------------------------------------------------------
def test_a_missing_self_approval_rule_denies(monkeypatch):
    """The state every deployment is in before the migration runs. An absent
    rule is not permission -- if it were, forgetting to migrate would silently
    reopen the thing the migration exists to close."""
    _policy(_ABSENT, monkeypatch)
    _draft(monkeypatch, requested_by=REQUESTER)
    _recorded(monkeypatch)

    response = _client_as(REQUESTER).post("/approvals/dispatch/UID-1", json={})

    assert response.status_code == 403, (
        f"a missing rule was read as permission: {response.status_code}")


def test_an_unreadable_policy_denies(monkeypatch):
    """Same reasoning one step further out: no policy at all is not consent."""
    monkeypatch.setattr(approvals_router, "_capability_policy", lambda engine: None)
    monkeypatch.setattr(approvals_router.rbac, "policy_engine", lambda *a, **k: object())
    _draft(monkeypatch, requested_by=REQUESTER)
    _recorded(monkeypatch)

    response = _client_as(REQUESTER).post("/approvals/dispatch/UID-1", json={})

    assert response.status_code == 403


def test_a_customer_may_permit_self_approval_deliberately(monkeypatch):
    """It is a policy, not a constant. A customer who writes the rule down owns
    the consequence -- but they have to write it down."""
    _policy("allow", monkeypatch)
    _draft(monkeypatch, requested_by=REQUESTER)
    written = _recorded(monkeypatch)

    response = _client_as(REQUESTER).post("/approvals/dispatch/UID-1", json={})

    assert response.status_code == 200, response.text
    assert written.get("actioned_by") == REQUESTER


def test_an_unrecognised_value_denies(monkeypatch):
    """A typo narrows, never widens -- the rule this surface already follows
    for revoke_scope."""
    _policy("Deny, obviously", monkeypatch)
    _draft(monkeypatch, requested_by=REQUESTER)
    _recorded(monkeypatch)

    assert _client_as(REQUESTER).post("/approvals/dispatch/UID-1", json={}).status_code == 403


# ---------------------------------------------------------------------------
# the requester comes from the record, never from the caller
# ---------------------------------------------------------------------------
def test_the_body_cannot_declare_who_requested_it(monkeypatch):
    """The mirror of the rule that the body cannot name the approver: a caller
    who could name the requester could name somebody else and approve freely."""
    _policy("deny", monkeypatch)
    _draft(monkeypatch, requested_by=REQUESTER)
    _recorded(monkeypatch)

    response = _client_as(REQUESTER).post(
        "/approvals/dispatch/UID-1",
        json={"reason": "fine", "requested_by": SOMEONE_ELSE},
    )

    assert response.status_code == 403, (
        "the request body was allowed to reassign who requested the draft")


# ---------------------------------------------------------------------------
# the same rule on the other approvable thing
# ---------------------------------------------------------------------------
def test_whoever_started_a_negotiation_cannot_approve_its_round(monkeypatch):
    """A round has no artefact, so the requester is the workflow's initiator,
    read from proc.workflow_execution."""
    _policy("deny", monkeypatch)
    monkeypatch.setattr(approvals_router.approval_store,
                        "negotiation_workflow_exists", lambda **k: True)
    monkeypatch.setattr(approvals_router.approval_store,
                        "workflow_initiator", lambda workflow_id: REQUESTER)
    written = _recorded(monkeypatch)

    response = _client_as(REQUESTER).post("/approvals/round/WF-1/2", json={})

    assert response.status_code == 403, response.text
    assert not written, "the round approval was recorded despite being refused"


def test_someone_else_may_approve_a_round(monkeypatch):
    _policy("deny", monkeypatch)
    monkeypatch.setattr(approvals_router.approval_store,
                        "negotiation_workflow_exists", lambda **k: True)
    monkeypatch.setattr(approvals_router.approval_store,
                        "workflow_initiator", lambda workflow_id: REQUESTER)
    written = _recorded(monkeypatch)

    response = _client_as(SOMEONE_ELSE).post("/approvals/round/WF-1/2", json={})

    assert response.status_code == 200, response.text
    assert written.get("actioned_by") == SOMEONE_ELSE


def test_an_agent_started_workflow_is_still_approvable(monkeypatch):
    """Every workflow initiator in the live tables today is 'AgentNick',
    'system', 'human' or NULL -- never a person. None of those can equal a
    Cognito subject, so this rule does not fire on the round path yet. It must
    also not break it."""
    _policy("deny", monkeypatch)
    monkeypatch.setattr(approvals_router.approval_store,
                        "negotiation_workflow_exists", lambda **k: True)
    monkeypatch.setattr(approvals_router.approval_store,
                        "workflow_initiator", lambda workflow_id: "AgentNick")
    written = _recorded(monkeypatch)

    response = _client_as(REQUESTER).post("/approvals/round/WF-1/2", json={})

    assert response.status_code == 200, response.text
    assert written.get("actioned_by") == REQUESTER
