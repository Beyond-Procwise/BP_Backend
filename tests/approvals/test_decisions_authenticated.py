"""These endpoints resolve findings and release supplier email replies, so who
acted is load-bearing -- and it was taken from the request body.

The router has seven endpoints: four writes (decide/act on a finding,
decide/act on an email reply) and three reads (list decisions, fetch one,
read the supplier message behind an escalated email decision). All seven must
refuse an unauthenticated caller. The three reads need authentication only --
no capability check -- so they are proven separately from the writes, which
must also record the actor from the token and never from the body.

Every write test asserts that nothing was written when the principal is
absent, not merely that a 401 came back. Each engine entry point a route
could reach is replaced with a spy that (a) records its own name into a
shared, per-test `calls` list and (b) returns a benign value rather than
raising -- so if a future edit moved the auth check to AFTER the write (or
dropped it), the request would come back 200 with the spy's canned payload,
and the explicit `assert calls == []` below is what catches that, not the
status code. A status-code-only assertion would still pass in that broken
world; this would not.
"""

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routers import decisions as decisions_router


class _Principal:
    subject = "sub-approver-001"
    email = "approver@ourcompany.com"
    claims = {"cognito:groups": ["bp-approvers"]}


class _FakeDecision:
    """Stands in for whatever decide_finding/decide_email_reply would have
    produced. Never legitimately reached in these tests -- only exists so
    that if the guard fails to fire, the endpoint still runs to completion
    (200) instead of erroring on an unrelated AttributeError, which would
    mask the real regression this file exists to catch."""

    decision = "noop"

    def to_dict(self):
        return {"decision": "noop"}


def _spy(calls, name, retval):
    def _inner(*_a, **_k):
        calls.append(name)
        return retval

    return _inner


def _agent_nick(calls):
    """An agent_nick whose DB entry point records itself instead of touching
    a real connection -- proof that an unauthenticated read never gets that
    far either."""

    return SimpleNamespace(
        get_db_connection=_spy(calls, "get_db_connection", None),
        policy_engine=SimpleNamespace(get_policy=lambda slug: None),
    )


@pytest.fixture
def anonymous_client(monkeypatch):
    calls: list = []
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.decide_finding",
        _spy(calls, "decide_finding", _FakeDecision()),
        raising=True,
    )
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.record",
        _spy(calls, "record", 1),
        raising=True,
    )
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.execute",
        _spy(calls, "execute", {"applied": True, "action": "noop"}),
        raising=True,
    )
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.decide_email_reply",
        _spy(calls, "decide_email_reply", _FakeDecision()),
        raising=True,
    )
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.act_on_email_reply",
        _spy(calls, "act_on_email_reply", {"applied": True, "action": "noop"}),
        raising=True,
    )
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.trace",
        _spy(calls, "trace", {"decision_id": 1}),
        raising=True,
    )

    app = FastAPI()
    app.include_router(decisions_router.router)
    app.dependency_overrides[decisions_router.require_user] = lambda: None
    app.state.agent_nick = _agent_nick(calls)
    client = TestClient(app)
    client.calls = calls  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# The four writes: authentication AND a real actor. Nothing is written.
# ---------------------------------------------------------------------------


def test_deciding_a_finding_requires_authentication(anonymous_client):
    response = anonymous_client.post("/decisions/finding/F-1", json={})
    assert response.status_code in (401, 403)
    assert anonymous_client.calls == [], (
        "decide_finding/record must never run without a principal"
    )


def test_acting_on_a_finding_requires_authentication(anonymous_client):
    response = anonymous_client.post(
        "/decisions/finding/F-1/action", json={"action": "resolve"}
    )
    assert response.status_code in (401, 403)
    assert anonymous_client.calls == [], "execute must never run without a principal"


def test_deciding_an_email_reply_requires_authentication(anonymous_client):
    response = anonymous_client.post("/decisions/email-reply/1", json={})
    assert response.status_code in (401, 403)
    assert anonymous_client.calls == [], (
        "decide_email_reply/record must never run without a principal"
    )


def test_deciding_an_email_reply_requires_authentication_with_no_body(anonymous_client):
    # body is Optional[DecideRequest] = None on this route -- must still refuse.
    response = anonymous_client.post("/decisions/email-reply/1")
    assert response.status_code in (401, 403)
    assert anonymous_client.calls == []


def test_acting_on_an_email_reply_requires_authentication(anonymous_client):
    response = anonymous_client.post(
        "/decisions/email-reply/7/action", json={"action": "send"}
    )
    assert response.status_code in (401, 403)
    assert anonymous_client.calls == [], (
        "act_on_email_reply must never run without a principal"
    )


# ---------------------------------------------------------------------------
# The three reads: authentication is enough, no capability check. Nothing is
# read from the database either -- the guard fires before get_db_connection.
# ---------------------------------------------------------------------------


def test_listing_decisions_requires_authentication(anonymous_client):
    response = anonymous_client.get("/decisions")
    assert response.status_code in (401, 403)
    assert anonymous_client.calls == [], (
        "the queue must never be read without a principal"
    )


def test_fetching_a_decision_requires_authentication(anonymous_client):
    response = anonymous_client.get("/decisions/7")
    assert response.status_code in (401, 403)
    assert anonymous_client.calls == [], "trace must never run without a principal"


def test_reading_an_email_reply_message_requires_authentication(anonymous_client):
    response = anonymous_client.get("/decisions/email-reply/7/message")
    assert response.status_code in (401, 403)
    assert anonymous_client.calls == [], (
        "the supplier message must never be read without a principal"
    )


# ---------------------------------------------------------------------------
# The body no longer carries an actor at all.
# ---------------------------------------------------------------------------


def test_the_request_models_no_longer_accept_an_actor():
    assert "user_id" not in decisions_router.DecideRequest.model_fields, (
        "a body-supplied actor is exactly the forgery this removes"
    )
    assert "user_id" not in decisions_router.ActionRequest.model_fields
    assert "user_id" not in decisions_router.EmailActionRequest.model_fields


# ---------------------------------------------------------------------------
# The authenticated path: the actor recorded is the token's subject, not
# anything the caller could have put in the body.
# ---------------------------------------------------------------------------


@pytest.fixture
def authenticated_app(monkeypatch):
    app = FastAPI()
    app.include_router(decisions_router.router)
    app.dependency_overrides[decisions_router.require_user] = lambda: _Principal()
    app.state.agent_nick = _agent_nick([])
    return app


def test_deciding_a_finding_records_the_token_subject_as_actor(authenticated_app, monkeypatch):
    captured = {}

    class _Decision:
        decision = "resolve"

        def to_dict(self):
            return {"decision": "resolve"}

    def fake_decide_finding(self, finding_id, requested=None):
        return _Decision()

    def fake_record(self, decision, *, workflow_id=None, agent=None, created_by=None):
        captured["created_by"] = created_by
        return 1

    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.decide_finding",
        fake_decide_finding,
        raising=True,
    )
    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.record", fake_record, raising=True
    )

    client = TestClient(authenticated_app)
    response = client.post("/decisions/finding/F-1", json={})
    assert response.status_code == 200
    assert captured["created_by"] == _Principal.subject


def test_acting_on_a_finding_records_the_token_subject_as_actor(authenticated_app, monkeypatch):
    captured = {}

    def fake_execute(self, finding_id, action, *, user_id=None, value=None, override_reason=None):
        captured["user_id"] = user_id
        return {"applied": True, "action": action}

    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.execute", fake_execute, raising=True
    )

    client = TestClient(authenticated_app)
    response = client.post("/decisions/finding/F-1/action", json={"action": "dismiss"})
    assert response.status_code == 200
    assert captured["user_id"] == _Principal.subject


def test_acting_on_an_email_reply_records_the_token_subject_as_actor(
    authenticated_app, monkeypatch
):
    captured = {}

    def fake_act(self, decision_id, action, *, user_id=None, override_reason=None):
        captured["user_id"] = user_id
        return {"applied": True, "action": action, "decision_id": decision_id}

    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.act_on_email_reply",
        fake_act,
        raising=True,
    )

    client = TestClient(authenticated_app)
    response = client.post("/decisions/email-reply/7/action", json={"action": "send"})
    assert response.status_code == 200
    assert captured["user_id"] == _Principal.subject


def test_a_caller_supplied_user_id_in_the_body_cannot_become_the_actor(
    authenticated_app, monkeypatch
):
    """The forgery this whole task removes: even if a caller still posts
    user_id (an older client, or someone probing), it must never reach the
    actor -- only the token's subject can."""

    captured = {}

    def fake_execute(self, finding_id, action, *, user_id=None, value=None, override_reason=None):
        captured["user_id"] = user_id
        return {"applied": True, "action": action}

    monkeypatch.setattr(
        "engines.decision_engine.DecisionEngine.execute", fake_execute, raising=True
    )

    client = TestClient(authenticated_app)
    response = client.post(
        "/decisions/finding/F-1/action",
        json={"action": "dismiss", "user_id": "someone-forged"},
    )
    assert response.status_code == 200
    assert captured["user_id"] == _Principal.subject
    assert captured["user_id"] != "someone-forged"
