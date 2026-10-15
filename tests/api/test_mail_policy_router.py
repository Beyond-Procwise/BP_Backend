"""Email policy routes, through the real app (so the exception handlers run)."""
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.routers import mail_integration as r
from services import mail_policy as mp

B = "/admin/integrations/mail"


class P:
    subject, email, username = "u1", "admin@cust.com", "admin"


def _policy(version=3):
    return mp.EmailPolicy(version=version, internal_send="off", company_domains=("cust.com",),
                          external_recipients="known_suppliers_only",
                          read_scope="replies_and_suppliers", on_disconnect="hold",
                          access_level="read_send", notices_from_mailbox="off",
                          poll_seconds=120, restriction_group="Procwise Mailboxes",
                          history=({"version": 3, "field": "x"},))


@pytest.fixture
def real(monkeypatch):
    from api.main import app
    app.dependency_overrides[r.require_user] = lambda: P()
    calls = []

    def gate(action, principal, **kw):
        calls.append(action)
        if action not in allowed:
            raise HTTPException(status_code=403, detail="not permitted")
    allowed = {"mailbox.view", "mailbox.bind"}
    monkeypatch.setattr(r, "gate", gate)
    monkeypatch.setattr(r, "_can_configure", lambda p: "mailbox.bind" in allowed)
    monkeypatch.setattr(mp, "load", lambda conn=None: _policy())
    c = TestClient(app, raise_server_exceptions=False)
    c.calls, c.allowed = calls, allowed
    yield c
    app.dependency_overrides.pop(r.require_user, None)


def test_get_policy_as_admin(real):
    res = real.get(f"{B}/policy")
    assert res.status_code == 200
    j = res.json()
    assert j["canConfigure"] is True
    assert j["policy"]["version"] == 3 and j["policy"]["company_domains"] == ["cust.com"]
    assert j["history"] == [{"version": 3, "field": "x"}]
    assert real.calls == ["mailbox.view"]


def test_get_policy_as_security_reviewer_cannot_configure(real):
    real.allowed.discard("mailbox.bind")
    res = real.get(f"{B}/policy")
    assert res.status_code == 200 and res.json()["canConfigure"] is False


def test_patch_as_security_reviewer_is_403(real, monkeypatch):
    real.allowed.discard("mailbox.bind")
    monkeypatch.setattr(mp, "save", lambda *a, **k: pytest.fail("must not save"))
    assert real.patch(f"{B}/policy", json={"changes": {"poll_seconds": 90}, "version": 3}).status_code == 403


def test_patch_ok_saves_as_the_caller(real, monkeypatch):
    seen = {}

    def save(changes, *, by, expected_version, conn=None):
        seen.update(changes=changes, by=by, v=expected_version)
        return _policy(4)
    monkeypatch.setattr(mp, "save", save)
    res = real.patch(f"{B}/policy", json={"changes": {"poll_seconds": 90}, "version": 3})
    assert res.status_code == 200 and res.json()["policy"]["version"] == 4
    assert seen == {"changes": {"poll_seconds": 90}, "by": "admin@cust.com", "v": 3}
    assert real.calls == ["mailbox.bind"]


def test_patch_stale_version_is_409(real, monkeypatch):
    def save(*a, **k):
        raise mp.PolicyConflict("changed by somebody else")
    monkeypatch.setattr(mp, "save", save)
    res = real.patch(f"{B}/policy", json={"changes": {"poll_seconds": 90}, "version": 2})
    assert res.status_code == 409
    assert res.json()["detail"]["message"] == "changed by somebody else"


def test_patch_bad_value_is_422_with_the_validators_words(real, monkeypatch):
    def save(*a, **k):
        raise ValueError("poll_seconds: must be between 60 and 3600 seconds")
    monkeypatch.setattr(mp, "save", save)
    res = real.patch(f"{B}/policy", json={"changes": {"poll_seconds": 5}, "version": 3})
    assert res.status_code == 422
    assert res.json()["detail"]["message"] == "poll_seconds: must be between 60 and 3600 seconds"


def test_policy_unavailable_is_503_on_read_and_write(real, monkeypatch):
    def boom(*a, **k):
        raise mp.PolicyUnavailable("email policy: the policy row is missing")
    monkeypatch.setattr(mp, "load", boom)
    monkeypatch.setattr(mp, "save", boom)
    assert real.get(f"{B}/policy").status_code == 503
    assert real.patch(f"{B}/policy", json={"changes": {"poll_seconds": 90}, "version": 3}).status_code == 503


def test_setup_script_follows_the_policy(real, monkeypatch):
    monkeypatch.setattr(r, "load_settings", lambda: type("S", (), {"client_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"})())

    class C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(r, "_conn", lambda: C())
    monkeypatch.setattr(r, "read_integration", lambda c: {"tenant_id": "11111111-2222-3333-4444-555555555555"})
    res = real.get(f"{B}/m365/setup-script")
    assert res.status_code == 200
    assert '-Role "Application Mail.Read"' in res.text and "Mail.ReadWrite" not in res.text
    assert "Procwise Mailboxes" in res.text


# ---- _can_configure is the real function here ----

def test_can_configure_admin_true_reviewer_false_error_false(monkeypatch):
    from services import rbac
    roles = {"admin": "Admin", "rev": "SecurityReviewer"}
    monkeypatch.setattr(rbac, "effective_role", lambda p: roles[p.subject])
    monkeypatch.setattr(rbac, "may", lambda role, cls: role == "Admin" and cls == "configure")

    class U:
        def __init__(self, s): self.subject = s
    assert r._can_configure(U("admin")) is True
    assert r._can_configure(U("rev")) is False

    def boom(p): raise RuntimeError("policy engine down")
    monkeypatch.setattr(rbac, "effective_role", boom)
    assert r._can_configure(U("admin")) is False


def test_policy_unavailable_is_the_same_class_the_router_catches():
    # one canonical import root: mail_policy's own PolicyUnavailable IS the router's
    import importlib
    assert r.mail_policy.PolicyUnavailable is importlib.import_module("services.mail_policy").PolicyUnavailable
    assert r.mail_policy is mp


def test_real_unavailable_from_mail_policy_gives_503(real, monkeypatch):
    # raised by mail_policy's own code path (missing row), not a stub exception
    class Cur:
        def execute(self, *a): pass
        def fetchone(self): return None
    class Conn:
        def cursor(self): return Cur()
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(mp, "load", lambda conn=None: mp._build(*mp._fetch(Cur())[1:]))
    assert real.get(f"{B}/policy").status_code == 503
