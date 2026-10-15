import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from api.routers import mail_integration as r
from services.mail_m365 import admin, consent
from services.mail_m365.config import M365Settings
from services.style.graph_source import MailboxAccessDenied, MailboxUnreachable
from services.mail_m365.graph_client import Throttled

S = M365Settings("cid", "arn:aws:secretsmanager:eu-west-1:1:secret:a",
                 "https://app.procwise.co.uk", "arn:aws:secretsmanager:eu-west-1:1:secret:b", 120)


class P:
    subject, email, username = "u1", "admin@cust.com", "admin"


class _NullConn:
    def __enter__(self): return self
    def __exit__(self, *a): return False


@pytest.fixture
def client(monkeypatch):
    app = FastAPI()
    app.include_router(r.router)
    app.include_router(r.callback_router)
    app.dependency_overrides[r.require_user] = lambda: P()
    calls = []
    monkeypatch.setattr(r, "gate", lambda action, principal, **kw: calls.append(action))
    monkeypatch.setattr(r, "load_settings", lambda: S)
    monkeypatch.setattr(r, "_conn", lambda: _NullConn())
    monkeypatch.setattr(r, "load_state_key", lambda s: b"k" * 32)
    c = TestClient(app)
    c.gate_calls = calls
    return c


def test_get_status_is_gated_and_reports_can_configure(client, monkeypatch):
    monkeypatch.setattr(admin, "status", lambda conn, settings: {"connected": False})
    res = client.get("/admin/integrations/mail")
    assert client.gate_calls == ["mailbox.bind"]
    assert res.json()["canConfigure"] is True


def test_gate_refusal_is_403(client, monkeypatch):
    def deny(action, principal, **kw):
        raise HTTPException(status_code=403, detail="not permitted")
    monkeypatch.setattr(r, "gate", deny)
    assert client.get("/admin/integrations/mail").status_code == 403


def test_all_admin_routes_gate_before_doing_anything(client, monkeypatch):
    def deny(action, principal, **kw):
        raise HTTPException(status_code=403, detail="no")
    monkeypatch.setattr(r, "gate", deny)
    b = "/admin/integrations/mail"
    calls = [("post", f"{b}/m365/connect", None), ("get", f"{b}/m365/setup-script", None),
             ("post", f"{b}/mailboxes", {"address": "a@c.com", "control_mailbox": "b@c.com"}),
             ("post", f"{b}/mailboxes/1/verify", {"control_mailbox": "b@c.com"}),
             ("post", f"{b}/mailboxes/1/test-send", None),
             ("patch", f"{b}/mailboxes/1", {"is_default": True}),
             ("delete", f"{b}/mailboxes/1", None), ("delete", b, None)]
    for method, url, body in calls:
        res = getattr(client, method)(url, **({"json": body} if body else {}))
        assert res.status_code == 403, (method, url)


def test_not_configured_says_so(client, monkeypatch):
    monkeypatch.setattr(r, "load_settings", lambda: None)
    res = client.post("/admin/integrations/mail/m365/connect")
    assert res.status_code == 503
    assert "not configured on this server" in res.json()["detail"]


def test_refusal_message_is_passed_through(client, monkeypatch):
    def refuse(*a, **k):
        raise admin.MailboxRefused("duplicate", address="p@c.com")
    monkeypatch.setattr(admin, "add_mailbox", refuse)
    res = client.post("/admin/integrations/mail/mailboxes",
                      json={"address": "p@c.com", "control_mailbox": "me@c.com"})
    assert res.status_code == 409
    assert res.json()["detail"] == {"code": "duplicate", "message": "p@c.com is already connected."}


def test_callback_needs_no_token_and_redirects(client, monkeypatch):
    client.app.dependency_overrides.clear()
    monkeypatch.setattr(consent, "complete", lambda *a, **k: {"status": "connected"})
    res = client.get("/admin/integrations/mail/m365/callback",
                     params={"state": "s", "tenant": "t", "admin_consent": "True"},
                     follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"].endswith("mail=connected")


def test_callback_with_bad_state_redirects_with_error(client, monkeypatch):
    client.app.dependency_overrides.clear()
    def bad(*a, **k):
        raise consent.StateRejected("reused")
    monkeypatch.setattr(consent, "complete", bad)
    res = client.get("/admin/integrations/mail/m365/callback",
                     params={"state": "s"}, follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"].endswith("mail_error=reused")


def test_callback_never_echoes_unknown_error_text(client, monkeypatch):
    def bad(*a, **k):
        raise consent.ConsentNotGiven("<script>alert(1)</script>&x=y")
    monkeypatch.setattr(consent, "complete", bad)
    res = client.get("/admin/integrations/mail/m365/callback",
                     params={"state": "s", "error": "<script>"}, follow_redirects=False)
    loc = res.headers["location"]
    assert loc.endswith("mail_error=consent_failed")
    assert "script" not in loc


@pytest.mark.parametrize("code", ["access_denied", "bad_tenant", "tenant_mismatch", "not_granted"])
def test_callback_known_consent_codes_pass(client, monkeypatch, code):
    def bad(*a, **k):
        raise consent.ConsentNotGiven(code)
    monkeypatch.setattr(consent, "complete", bad)
    res = client.get("/admin/integrations/mail/m365/callback", params={"state": "s"},
                     follow_redirects=False)
    assert res.headers["location"].endswith(f"mail_error={code}")


def test_setup_script_bad_id_is_400(client, monkeypatch):
    monkeypatch.setattr(r, "read_integration",
                        lambda conn: {"tenant_id": "11111111-1111-1111-1111-111111111111"})
    res = client.get("/admin/integrations/mail/m365/setup-script",
                     params={"service_principal_object_id": "x'; Remove-Item *"})
    assert res.status_code == 400
    assert "Remove-Item" not in res.text


def test_setup_script_ok(client, monkeypatch):
    monkeypatch.setattr(r, "read_integration",
                        lambda conn: {"tenant_id": "11111111-1111-1111-1111-111111111111"})
    monkeypatch.setattr(r, "render_script", lambda *a: "SCRIPT")
    assert client.get("/admin/integrations/mail/m365/setup-script").text == "SCRIPT"


def test_make_default_value_error_is_422(client, monkeypatch):
    def bad(*a, **k):
        raise ValueError("Send a test email first.")
    monkeypatch.setattr(admin, "make_default", bad)
    res = client.patch("/admin/integrations/mail/mailboxes/1", json={"is_default": True})
    assert res.status_code == 422
    assert res.json()["detail"] == "Send a test email first."


@pytest.mark.parametrize("exc,status", [
    (MailboxAccessDenied("secret trace"), 403),
    (MailboxUnreachable("secret trace"), 502),
    (Throttled(30), 503),
])
def test_test_send_errors_are_clean(client, monkeypatch, exc, status):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(admin, "test_send", boom)
    res = client.post("/admin/integrations/mail/mailboxes/1/test-send")
    assert res.status_code == status
    assert "secret trace" not in res.text and "throttled by" not in res.text
    assert isinstance(res.json()["detail"], str)


def test_callback_is_the_only_unauthenticated_mail_route(monkeypatch):
    monkeypatch.setattr(r, "load_settings", lambda: None)   # never reach AWS
    from api.main import app
    from fastapi.testclient import TestClient as TC
    from api import auth as _auth

    class _Refuse:
        def verify(self, token):
            raise _auth.AuthError("no")
    prev = (_auth._mode, _auth._verifier)
    _auth._mode, _auth._verifier = "enforce", _Refuse()
    try:
        c = TC(app, raise_server_exceptions=False)
        b = "/admin/integrations/mail"
        for method, url in [("get", b), ("post", f"{b}/m365/connect"),
                            ("get", f"{b}/m365/setup-script"), ("post", f"{b}/mailboxes"),
                            ("post", f"{b}/mailboxes/1/verify"),
                            ("post", f"{b}/mailboxes/1/test-send"),
                            ("patch", f"{b}/mailboxes/1"), ("delete", f"{b}/mailboxes/1"),
                            ("delete", b)]:
            assert getattr(c, method)(url).status_code in (401, 403), (method, url)
        # the callback is deliberately open: it answers without a token (redirect or 503)
        res = c.get(f"{b}/m365/callback", follow_redirects=False)
        assert res.status_code == 503
    finally:
        _auth._mode, _auth._verifier = prev


# ---- through the REAL app: middleware (output safety) and exception handlers run ----

@pytest.fixture
def real(monkeypatch):
    from api.main import app
    app.dependency_overrides[r.require_user] = lambda: P()
    monkeypatch.setattr(r, "gate", lambda *a, **k: None)
    monkeypatch.setattr(r, "load_settings", lambda: S)
    monkeypatch.setattr(r, "_conn", lambda: _NullConn())
    monkeypatch.setattr(r, "load_state_key", lambda s: b"k" * 32)
    yield TestClient(app, raise_server_exceptions=False)
    app.dependency_overrides.pop(r.require_user, None)


B = "/admin/integrations/mail"


def test_real_app_connect_url_arrives_intact(real, monkeypatch):
    url = ("https://login.microsoftonline.com/organizations/v2.0/adminconsent?client_id=cid"
           "&redirect_uri=https%3A%2F%2Fapp.procwise.co.uk%2Fadmin%2Fintegrations%2Fmail%2Fm365%2Fcallback"
           "&scope=x&state=abc.def")
    monkeypatch.setattr(consent, "begin", lambda *a, **k: url)
    res = real.post(f"{B}/m365/connect")
    assert res.status_code == 200
    assert res.json() == {"authorization_url": url}


def test_real_app_refusal_arrives_as_fields(real, monkeypatch):
    def refuse(*a, **k):
        raise admin.MailboxRefused("duplicate", address="p@c.com")
    monkeypatch.setattr(admin, "add_mailbox", refuse)
    res = real.post(f"{B}/mailboxes", json={"address": "p@c.com", "control_mailbox": "me@c.com"})
    assert res.status_code == 409
    d = res.json()["detail"]
    assert d["code"] == "duplicate" and d["message"] == "p@c.com is already connected."


def test_real_app_setup_script_bad_id_is_400(real, monkeypatch):
    monkeypatch.setattr(r, "read_integration",
                        lambda conn: {"tenant_id": "11111111-1111-1111-1111-111111111111"})
    res = real.get(f"{B}/m365/setup-script", params={"service_principal_object_id": "bad'; x"})
    assert res.status_code == 400
    assert "valid Object ID" in res.json()["detail"]


def test_real_app_make_default_422_keeps_its_words(real, monkeypatch):
    def bad(*a, **k):
        raise ValueError("Send a test email first.")
    monkeypatch.setattr(admin, "make_default", bad)
    res = real.patch(f"{B}/mailboxes/1", json={"is_default": True})
    assert res.status_code == 422
    assert res.json()["detail"] == "Send a test email first."


@pytest.mark.parametrize("exc,status", [
    (MailboxAccessDenied("secret trace"), 403),
    (MailboxUnreachable("secret trace"), 502),
    (Throttled(30), 503),
])
def test_real_app_test_send_errors_are_clean(real, monkeypatch, exc, status):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(admin, "test_send", boom)
    res = real.post(f"{B}/mailboxes/1/test-send")
    assert res.status_code == status
    assert "secret trace" not in res.text
    assert "[withheld]" not in res.text
    assert isinstance(res.json()["detail"], str) and len(res.json()["detail"]) > 20


def test_real_app_callback_unexpected_error_still_redirects(real, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(consent, "complete", boom)
    res = real.get(f"{B}/m365/callback", params={"state": "s"}, follow_redirects=False)
    assert res.status_code == 302
    assert res.headers["location"].endswith("mail_error=consent_failed")


# ---- GET /mail/sender: any signed-in user, never gated ----

from types import SimpleNamespace as _NS
from services.mail_m365 import sending as _sending


def _decision(kind, address=None, reason=None):
    b = _NS(mailbox_address=address) if address else None
    return _sending.Decision(kind, b, reason)


def _gate_must_not_run(*a, **k):
    raise AssertionError("the sender endpoint must not call the endpoint gate")


@pytest.fixture
def sender(real, monkeypatch):
    monkeypatch.setattr(r, "gate", _gate_must_not_run)
    return real


def test_sender_ses(sender, monkeypatch):
    monkeypatch.setattr(_sending, "decide", lambda conn: _decision("ses"))
    res = sender.get("/mail/sender")
    assert res.status_code == 200
    assert res.json() == {"mode": "ses", "address": None, "can_send": True, "reason": None}


def test_sender_m365(sender, monkeypatch):
    monkeypatch.setattr(_sending, "decide", lambda conn: _decision("graph", "proc@cust.com"))
    res = sender.get("/mail/sender")
    assert res.status_code == 200
    assert res.json() == {"mode": "m365", "address": "proc@cust.com", "can_send": True,
                          "reason": None}


def test_sender_held_with_and_without_binding(sender, monkeypatch):
    why = "company mailbox unavailable: proc@cust.com is disconnected"
    monkeypatch.setattr(_sending, "decide", lambda conn: _decision("held", "proc@cust.com", why))
    res = sender.get("/mail/sender")
    assert res.status_code == 200
    assert res.json() == {"mode": "held", "address": "proc@cust.com", "can_send": False,
                          "reason": why}
    monkeypatch.setattr(_sending, "decide", lambda conn: _decision("held", None, "no default"))
    assert sender.get("/mail/sender").json()["address"] is None


def test_sender_decide_raising_is_held_200(sender, monkeypatch):
    def boom(conn):
        raise RuntimeError("db down: secret-dsn")
    monkeypatch.setattr(_sending, "decide", boom)
    res = sender.get("/mail/sender")
    assert res.status_code == 200
    assert res.json() == {"mode": "held", "address": None, "can_send": False,
                          "reason": "company mailbox status unavailable"}
    assert "secret-dsn" not in res.text


def test_sender_unauthenticated_is_refused(monkeypatch):
    from api.main import app
    from api import auth as _auth

    class _RefuseAll:
        def verify(self, token):
            raise _auth.AuthError("no tokens are valid in this test")

    monkeypatch.setattr(_auth, "_mode", "enforce")
    monkeypatch.setattr(_auth, "_verifier", _RefuseAll())
    monkeypatch.setattr(r, "gate", _gate_must_not_run)
    app.dependency_overrides.pop(r.require_user, None)
    res = TestClient(app, raise_server_exceptions=False).get("/mail/sender")
    assert res.status_code == 401
