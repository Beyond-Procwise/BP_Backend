"""The approver is whoever the token says, and nobody else.

The previous attempt at this surface was reverted because it took the
approver's name from the request body over an unauthenticated route, which
let anyone forge a human approval. These tests exist mainly to keep that
shut.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import approvals as approvals_router


class _Principal:
    subject = "sub-buyer-001"
    email = "buyer@ourcompany.com"
    claims = {"cognito:groups": ["bp-buyers"]}


class _PrincipalAs:
    """A principal with a caller-chosen subject, for ownership-scope tests."""

    def __init__(self, subject):
        self.subject = subject
        self.email = f"{subject}@ourcompany.com"
        self.claims = {"cognito:groups": []}


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(approvals_router.router)
    app.dependency_overrides[approvals_router.require_user] = lambda: _Principal()
    return TestClient(app)


@pytest.fixture
def anonymous_client():
    app = FastAPI()
    app.include_router(approvals_router.router)
    app.dependency_overrides[approvals_router.require_user] = lambda: None
    return TestClient(app)


def _client_as(principal):
    app = FastAPI()
    app.include_router(approvals_router.router)
    app.dependency_overrides[approvals_router.require_user] = lambda: principal
    return TestClient(app)


def _allow(*a, **k):
    return approvals_router.guardrail.Decision(allowed=True, reason="ok")


def _deny(*a, **k):
    return approvals_router.guardrail.Decision(
        allowed=False, reason="role Viewer may not perform approve_email"
    )


def test_the_body_cannot_name_the_approver(client, monkeypatch):
    """The single most important test in this file."""
    recorded = {}

    def fake_record(**kwargs):
        recorded.update(kwargs)
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    monkeypatch.setattr(
        approvals_router, "_load_draft", lambda uid, conn=None: {"unique_id": uid}
    )
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(allowed=True, reason="ok"),
    )

    response = client.post(
        "/approvals/dispatch/PROC-WF-1",
        json={"actioned_by": "ceo@ourcompany.com", "user_id": "ceo@ourcompany.com"},
    )

    assert response.status_code == 200
    assert recorded["actioned_by"] == "sub-buyer-001", (
        "the approver came from the request body, not the token"
    )


def test_an_unauthenticated_caller_cannot_approve(anonymous_client, monkeypatch):
    called = {"recorded": False}

    def fake_record(**kwargs):
        called["recorded"] = True
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    response = anonymous_client.post("/approvals/dispatch/PROC-WF-1", json={})
    assert response.status_code in (401, 403)
    assert called["recorded"] is False, "an approval was written with no principal"


def test_a_denied_capability_writes_nothing(client, monkeypatch):
    called = {"recorded": False}

    def fake_record(**kwargs):
        called["recorded"] = True
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    monkeypatch.setattr(
        approvals_router, "_load_draft", lambda uid, conn=None: {"unique_id": uid}
    )
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(
            allowed=False, reason="role Viewer may not perform approve_email"
        ),
    )

    response = client.post("/approvals/dispatch/PROC-WF-1", json={})
    assert response.status_code == 403
    assert called["recorded"] is False


def test_approving_a_missing_draft_is_refused(client, monkeypatch):
    monkeypatch.setattr(approvals_router, "_load_draft", lambda uid, conn=None: None)
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(allowed=True, reason="ok"),
    )
    response = client.post("/approvals/dispatch/PROC-WF-NOPE", json={})
    assert response.status_code == 404


def test_the_content_hash_is_recorded(client, monkeypatch):
    recorded = {}

    def fake_record(**kwargs):
        recorded.update(kwargs)
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    monkeypatch.setattr(
        approvals_router, "_load_draft",
        lambda uid, conn=None: {
            "unique_id": uid, "subject": "RFQ", "body": "Please quote.",
            "recipients": ["buyer@supplier-b.com"],
        },
    )
    monkeypatch.setattr(
        approvals_router.guardrail, "authorize",
        lambda *a, **k: approvals_router.guardrail.Decision(allowed=True, reason="ok"),
    )

    client.post("/approvals/dispatch/PROC-WF-1", json={})
    assert recorded["grounding_extra"]["content_hash"]


# ---------------------------------------------------------------------------
# POST /approvals/round/{workflow_id}/{round_num}
# ---------------------------------------------------------------------------


def test_round_an_unauthenticated_caller_cannot_approve(anonymous_client, monkeypatch):
    called = {"recorded": False}

    def fake_record(**kwargs):
        called["recorded"] = True
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    response = anonymous_client.post("/approvals/round/PROC-WF-1/1", json={})
    assert response.status_code in (401, 403)
    assert called["recorded"] is False, "a round approval was written with no principal"


def test_round_a_denied_capability_writes_nothing(client, monkeypatch):
    called = {"recorded": False}

    def fake_record(**kwargs):
        called["recorded"] = True
        return 1

    monkeypatch.setattr(approvals_router.approval_store, "record_approval", fake_record)
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _deny)

    response = client.post("/approvals/round/PROC-WF-1/1", json={})
    assert response.status_code == 403
    assert called["recorded"] is False


# ---------------------------------------------------------------------------
# GET /approvals/pending
# ---------------------------------------------------------------------------


def test_pending_an_unauthenticated_caller_sees_nothing(anonymous_client, monkeypatch):
    called = {"listed": False}

    def fake_list(**kwargs):
        called["listed"] = True
        return []

    monkeypatch.setattr(
        approvals_router.approval_store, "list_pending_dispatch_approvals", fake_list
    )
    response = anonymous_client.get("/approvals/pending")
    assert response.status_code in (401, 403)
    assert called["listed"] is False, "pending drafts were listed with no principal"


def test_pending_a_denied_capability_lists_nothing(client, monkeypatch):
    called = {"listed": False}

    def fake_list(**kwargs):
        called["listed"] = True
        return []

    monkeypatch.setattr(
        approvals_router.approval_store, "list_pending_dispatch_approvals", fake_list
    )
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _deny)

    response = client.get("/approvals/pending")
    assert response.status_code == 403
    assert called["listed"] is False


# ---------------------------------------------------------------------------
# POST /approvals/{approval_id}/revoke
# ---------------------------------------------------------------------------


def test_revoke_an_unauthenticated_caller_cannot_revoke(anonymous_client, monkeypatch):
    called = {"revoked": False}

    def fake_revoke(**kwargs):
        called["revoked"] = True
        return 2

    monkeypatch.setattr(approvals_router.approval_store, "revoke_approval", fake_revoke)
    response = anonymous_client.post("/approvals/1/revoke", json={})
    assert response.status_code in (401, 403)
    assert called["revoked"] is False, "an approval was revoked with no principal"


def test_revoke_a_denied_capability_revokes_nothing(client, monkeypatch):
    called = {"revoked": False}

    def fake_revoke(**kwargs):
        called["revoked"] = True
        return 2

    monkeypatch.setattr(approvals_router.approval_store, "revoke_approval", fake_revoke)
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _deny)

    response = client.post("/approvals/1/revoke", json={})
    assert response.status_code == 403
    assert called["revoked"] is False


class _FakeRevokeScopeEngine:
    """Enough of PolicyEngine.get_policy for the revoke-scope decision.

    Real rank numbers (Viewer 1 < Buyer 2 < Approver 3 < Admin 4), read live
    from proc.bp_policy's role_definition row and mirrored here so the
    ownership-scope logic is exercised for real rather than through a
    monkeypatched verdict.
    """

    _ROLE_DEFINITION = {
        "details": {
            "rules": {
                "roles": {
                    "Viewer": {"rank": 1},
                    "Buyer": {"rank": 2},
                    "Approver": {"rank": 3},
                    "Admin": {"rank": 4},
                }
            }
        }
    }

    def __init__(self, revoke_scope="own_or_higher_rank"):
        self._capability = {
            "details": {
                "required_role": "Buyer",
                "rules": {"revoke_scope": revoke_scope} if revoke_scope is not None else {},
            }
        }

    def get_policy(self, slug):
        if slug == "email_approval_capability":
            return self._capability
        if slug == "role_definition":
            return self._ROLE_DEFINITION
        return None


def _revoke_client(monkeypatch, *, caller_subject, owner_subject, revoke_scope, caller_role):
    """A client wired for the ownership-scope decision, nothing else mocked away."""

    principal = _PrincipalAs(caller_subject)
    c = _client_as(principal)

    monkeypatch.setattr(approvals_router.guardrail, "authorize", _allow)
    monkeypatch.setattr(
        approvals_router.approval_store,
        "get_approval",
        lambda **kwargs: {"approval_id": kwargs["approval_id"], "actioned_by": owner_subject},
    )
    monkeypatch.setattr(
        approvals_router.rbac, "policy_engine",
        lambda: _FakeRevokeScopeEngine(revoke_scope),
    )
    monkeypatch.setattr(
        approvals_router.rbac, "effective_role",
        lambda p, policy_engine=None: caller_role,
    )
    return c


def test_revoke_the_original_approver_may_revoke_their_own(monkeypatch):
    recorded = {}

    def fake_revoke(**kwargs):
        recorded.update(kwargs)
        return 2

    monkeypatch.setattr(approvals_router.approval_store, "revoke_approval", fake_revoke)
    c = _revoke_client(
        monkeypatch,
        caller_subject="sub-buyer-001",
        owner_subject="sub-buyer-001",
        revoke_scope="own_or_higher_rank",
        caller_role="Buyer",
    )
    response = c.post("/approvals/1/revoke", json={})
    assert response.status_code == 200
    assert recorded["actioned_by"] == "sub-buyer-001"


def test_revoke_a_lower_ranked_caller_may_not_revoke_someone_elses(monkeypatch):
    called = {"revoked": False}

    def fake_revoke(**kwargs):
        called["revoked"] = True
        return 2

    monkeypatch.setattr(approvals_router.approval_store, "revoke_approval", fake_revoke)
    c = _revoke_client(
        monkeypatch,
        caller_subject="sub-buyer-002",
        owner_subject="sub-buyer-001",
        revoke_scope="own_or_higher_rank",
        caller_role="Buyer",  # same rank as the required_role floor -- not senior
    )
    response = c.post("/approvals/1/revoke", json={})
    assert response.status_code == 403
    assert called["revoked"] is False


def test_revoke_a_higher_ranked_caller_may_revoke_someone_elses(monkeypatch):
    recorded = {}

    def fake_revoke(**kwargs):
        recorded.update(kwargs)
        return 2

    monkeypatch.setattr(approvals_router.approval_store, "revoke_approval", fake_revoke)
    c = _revoke_client(
        monkeypatch,
        caller_subject="sub-approver-001",
        owner_subject="sub-buyer-001",
        revoke_scope="own_or_higher_rank",
        caller_role="Approver",  # strictly outranks the Buyer floor
    )
    response = c.post("/approvals/1/revoke", json={})
    assert response.status_code == 200
    assert recorded["actioned_by"] == "sub-approver-001"


def test_revoke_any_approver_scope_allows_revoking_someone_elses(monkeypatch):
    """today's pre-fix behaviour, still selectable by policy."""
    recorded = {}

    def fake_revoke(**kwargs):
        recorded.update(kwargs)
        return 2

    monkeypatch.setattr(approvals_router.approval_store, "revoke_approval", fake_revoke)
    c = _revoke_client(
        monkeypatch,
        caller_subject="sub-buyer-002",
        owner_subject="sub-buyer-001",
        revoke_scope="any_approver",
        caller_role="Buyer",
    )
    response = c.post("/approvals/1/revoke", json={})
    assert response.status_code == 200
    assert recorded["actioned_by"] == "sub-buyer-002"


def test_revoke_own_only_scope_refuses_even_a_higher_rank(monkeypatch):
    called = {"revoked": False}

    def fake_revoke(**kwargs):
        called["revoked"] = True
        return 2

    monkeypatch.setattr(approvals_router.approval_store, "revoke_approval", fake_revoke)
    c = _revoke_client(
        monkeypatch,
        caller_subject="sub-admin-001",
        owner_subject="sub-buyer-001",
        revoke_scope="own_only",
        caller_role="Admin",
    )
    response = c.post("/approvals/1/revoke", json={})
    assert response.status_code == 403
    assert called["revoked"] is False


def test_revoke_scope_defaults_to_own_or_higher_rank_when_absent():
    engine = _FakeRevokeScopeEngine(revoke_scope=None)
    assert approvals_router._revoke_scope(engine) == "own_or_higher_rank"


def test_revoke_scope_defaults_to_own_or_higher_rank_when_unrecognised():
    engine = _FakeRevokeScopeEngine(revoke_scope="whatever_is_convenient")
    assert approvals_router._revoke_scope(engine) == "own_or_higher_rank"


def test_revoking_a_missing_approval_is_refused(client, monkeypatch):
    monkeypatch.setattr(approvals_router.guardrail, "authorize", _allow)
    monkeypatch.setattr(
        approvals_router.approval_store, "get_approval", lambda **kwargs: None
    )
    response = client.post("/approvals/999999/revoke", json={})
    assert response.status_code == 404
