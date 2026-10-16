"""/mail/internal through the real app: identity comes from the token, refusals arrive as
fields, and nothing is reachable without signing in."""
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.routers import internal_email as r
from services import internal_email, mail_policy
from tests.services.test_internal_email import DEFAULT, FakeProvider, FakeStore

B = "/mail/internal"


class _NullConn:
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.fixture
def real(monkeypatch):
    from api.main import app
    who = SimpleNamespace(principal=SimpleNamespace(subject="u1", email="buyer@yourcompany.com",
                                                    username="buyer"))
    app.dependency_overrides[r.require_user] = lambda: who.principal
    gates = []
    monkeypatch.setattr(r, "gate", lambda action, principal, **k: gates.append(action))
    monkeypatch.setattr(r, "_conn", lambda: _NullConn())
    store, provider = FakeStore(), FakeProvider()
    monkeypatch.setattr(internal_email, "_store", lambda conn: store)
    monkeypatch.setattr(internal_email, "get_provider", lambda conn: provider)
    monkeypatch.setattr(internal_email, "read_integration", lambda conn: {"status": "connected"})
    monkeypatch.setattr(internal_email, "MailboxBindingRepository",
                        lambda conn=None: SimpleNamespace(get_default_sender=lambda: DEFAULT))
    monkeypatch.setattr(internal_email, "record_action", lambda **k: None)
    policy = SimpleNamespace(internal_send="with_user_approval", company_domains=("yourcompany.com",))
    monkeypatch.setattr(mail_policy, "load", lambda conn=None: policy)
    c = TestClient(app, raise_server_exceptions=False)
    c.who, c.gates, c.store, c.provider, c.policy = who, gates, store, provider, policy
    yield c
    app.dependency_overrides.pop(r.require_user, None)


def _create(c, **extra):
    body = {"kind": "forward", "recipients": ["finance@yourcompany.com"], "subject": "s", "body": "b"}
    body.update(extra)
    return c.post(B, json=body)


def test_create_is_gated_and_the_author_is_the_token_email(real):
    res = _create(real, author="ceo@yourcompany.com")
    assert res.status_code == 200, res.text
    assert res.json()["author"] == "buyer@yourcompany.com" and res.json()["status"] == "pending"
    assert real.gates == ["mail.internal.create"]
    assert real.provider.sent == []


def test_approve_by_a_different_principal_is_403_not_author(real):
    i = _create(real).json()["internal_email_id"]
    real.who.principal = SimpleNamespace(subject="u2", email="admin@yourcompany.com", username="admin")
    res = real.post(f"{B}/{i}/approve")
    assert res.status_code == 403
    assert res.json()["detail"] == {"code": "not_author",
                                    "message": "Only the person who wrote this email can approve it"}
    assert real.provider.sent == []


def test_a_body_cannot_name_the_approver(real):
    i = _create(real).json()["internal_email_id"]
    real.who.principal = SimpleNamespace(subject="u2", email="admin@yourcompany.com", username="admin")
    res = real.post(f"{B}/{i}/approve", json={"approver": "buyer@yourcompany.com"})
    assert res.status_code == 403 and real.provider.sent == []


def test_a_principal_without_an_email_cannot_approve(real):
    i = _create(real).json()["internal_email_id"]
    real.who.principal = SimpleNamespace(subject="u1", email=None, username="buyer@yourcompany.com")
    res = real.post(f"{B}/{i}/approve")
    assert res.status_code == 403 and res.json()["detail"]["code"] == "not_author"


def test_author_approves_and_it_is_sent(real):
    i = _create(real).json()["internal_email_id"]
    real.who.principal = SimpleNamespace(subject="u1", email="BUYER@yourcompany.com", username="b")
    res = real.post(f"{B}/{i}/approve")
    assert res.status_code == 200, res.text
    assert res.json()["status"] == "sent"
    assert real.gates == ["mail.internal.create", "mail.internal.approve"]
    assert real.provider.sent[0]["from"] == "procurement@yourcompany.com"
    again = real.post(f"{B}/{i}/approve")
    assert again.status_code == 409 and again.json()["detail"]["code"] == "already_done"


@pytest.mark.parametrize("setup,status,code", [
    (lambda c: setattr(c.policy, "internal_send", "off"), 422, "policy_off"),
    (lambda c: setattr(c.policy, "company_domains", ("other.com",)), 422, "not_internal"),
])
def test_refusals_at_approval_arrive_as_fields(real, setup, status, code):
    i = _create(real).json()["internal_email_id"]
    setup(real)
    res = real.post(f"{B}/{i}/approve")
    assert res.status_code == status and res.json()["detail"]["code"] == code
    assert isinstance(res.json()["detail"]["message"], str) and real.provider.sent == []


def test_outside_recipient_refused_at_create(real):
    res = _create(real, recipients=["x@gmail.com"])
    assert res.status_code == 422
    assert res.json()["detail"] == {
        "code": "not_internal",
        "message": "x@gmail.com is outside your company. Use the normal supplier email instead."}


def test_unknown_id_is_404(real):
    res = real.post(f"{B}/999/approve")
    assert res.status_code == 404 and res.json()["detail"]["code"] == "not_found"


def test_pending_lists_only_mine(real):
    _create(real)
    real.who.principal = SimpleNamespace(subject="u2", email="other@yourcompany.com", username="o")
    _create(real)
    items = real.get(f"{B}/pending").json()["items"]
    assert [x["author"] for x in items] == ["other@yourcompany.com"]


def test_edit_and_decline_are_author_only(real):
    i = _create(real).json()["internal_email_id"]
    edited = real.patch(f"{B}/{i}", json={"recipients": ["finance@yourcompany.com"],
                                         "subject": "s2", "body": "b2"})
    assert edited.status_code == 200 and edited.json()["subject"] == "s2"
    real.who.principal = SimpleNamespace(subject="u2", email="admin@yourcompany.com", username="a")
    assert real.patch(f"{B}/{i}", json={"recipients": ["finance@yourcompany.com"],
                                       "subject": "x", "body": "x"}).status_code == 403
    assert real.post(f"{B}/{i}/decline").status_code == 403
    real.who.principal = SimpleNamespace(subject="u1", email="buyer@yourcompany.com", username="b")
    res = real.post(f"{B}/{i}/decline")
    assert res.status_code == 200 and res.json()["status"] == "declined"


def test_write_routes_refused_by_the_gate_do_nothing(real, monkeypatch):
    i = _create(real).json()["internal_email_id"]
    def deny(action, principal, **k):
        raise HTTPException(status_code=403, detail="not permitted")
    monkeypatch.setattr(r, "gate", deny)
    assert _create(real).status_code == 403
    assert real.patch(f"{B}/{i}", json={"recipients": ["f@yourcompany.com"], "subject": "x",
                                       "body": "x"}).status_code == 403
    assert real.post(f"{B}/{i}/approve").status_code == 403
    assert real.post(f"{B}/{i}/decline").status_code == 403
    assert len(real.store.rows) == 1 and real.store.rows[i]["status"] == "pending"
    assert real.provider.sent == []


def test_unauthenticated_is_refused(monkeypatch):
    """Enforce mode, whatever the sandbox's .env says; the bodies are valid, so the only
    reason to refuse is the missing token."""
    from api.main import app
    from api import auth as _auth

    class _RefuseAll:
        def verify(self, token):
            raise _auth.AuthError("no tokens are valid in this test")
    monkeypatch.setattr(_auth, "_mode", "enforce")
    monkeypatch.setattr(_auth, "_verifier", _RefuseAll())
    app.dependency_overrides.pop(r.require_user, None)
    c = TestClient(app, raise_server_exceptions=False)
    create = {"kind": "forward", "recipients": ["f@yourcompany.com"], "subject": "s", "body": "b"}
    edit = {"recipients": ["f@yourcompany.com"], "subject": "s", "body": "b"}
    for method, url, body in [("post", B, create), ("get", f"{B}/pending", None),
                              ("patch", f"{B}/1", edit), ("post", f"{B}/1/approve", None),
                              ("post", f"{B}/1/decline", None)]:
        res = getattr(c, method)(url, **({"json": body} if body else {}))
        assert res.status_code == 401, (method, url, res.status_code)
