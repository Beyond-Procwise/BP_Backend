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
