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


# ---- restrictions and the evidence report ----

class _NoConn:
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Provider:
    def prove(self, conn):
        return [{"restriction": "no_mail_permissions", "check": "Mail permissions",
                 "answer": "none", "result": "pass", "at": "2026-10-16T10:00:00+00:00"}]

    def status(self, conn):
        from datetime import datetime, timezone
        return {"connected": True, "mailboxes": [
            {"binding_id": 1, "address": "procurement@cust.com", "active": True,
             "is_default": True, "health_state": "OK", "verified_at": datetime.now(timezone.utc),
             "last_test_send_at": None, "withdrawn": False}]}


@pytest.fixture
def wired(real, monkeypatch):
    monkeypatch.setattr(r, "_conn", lambda: _NoConn())
    monkeypatch.setattr(r, "get_provider", lambda conn: _Provider())
    audit = []
    monkeypatch.setattr(r, "record_action", lambda **k: audit.append(k))
    real.audit = audit
    return real


def test_restrictions_as_admin(wired):
    res = wired.get(f"{B}/restrictions")
    assert res.status_code == 200, res.text
    j = res.json()
    assert [i["id"] for i in j["restrictions"]][:3] == ["no_mail_permissions", "mailbox_scope",
                                                        "actions_limited"]
    assert j["summary"]["total"] == 8
    assert j["summary"]["in_force"] == sum(i["status"] == "in_force" for i in j["restrictions"])
    assert j["canConfigure"] is True
    assert wired.calls == ["mailbox.view"]
    # the words reach the screen as written (the output scrubber leaves them alone)
    rs = {i["id"]: i for i in j["restrictions"]}
    assert rs["read_scope"]["explanation"].startswith("Replies to emails Procwise sent")


def test_restrictions_with_unreadable_policy_still_answers(wired, monkeypatch):
    def boom(*a, **k):
        raise mp.PolicyUnavailable("missing")
    monkeypatch.setattr(mp, "load", boom)
    res = wired.get(f"{B}/restrictions")
    assert res.status_code == 200
    rs = {i["id"]: i for i in res.json()["restrictions"]}
    assert rs["read_scope"]["status"] == "not_proven"


def test_restrictions_reviewer_cannot_configure(wired):
    wired.allowed.discard("mailbox.bind")
    assert wired.get(f"{B}/restrictions").json()["canConfigure"] is False


def test_restrictions_and_pdf_refused_without_view(wired):
    wired.allowed.clear()
    assert wired.get(f"{B}/restrictions").status_code == 403
    assert wired.get(f"{B}/evidence.pdf").status_code == 403
    assert wired.audit == []


def test_evidence_pdf_download(wired):
    from datetime import datetime, timezone
    res = wired.get(f"{B}/evidence.pdf")
    assert res.status_code == 200, res.text
    assert res.headers["content-type"] == "application/pdf"
    assert res.content[:4] == b"%PDF"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert res.headers["content-disposition"] == \
        f'attachment; filename="email-restrictions-{today}.pdf"'
    assert wired.calls == ["mailbox.view"]
    assert [a["action_type"] for a in wired.audit] == ["evidence_report_downloaded"]
    assert wired.audit[0]["details"]["generated_by"] == "admin@cust.com"


# ---- Security reviewer, through the REAL gate and the role policy in bp_testdb ----

_live = pytest.mark.skipif(__import__("os").getenv("PROCWISE_TEST_LIVE_DB") != "1",
                           reason="live DB")

WRITES = [
    ("post", "/m365/connect", None, "connect"),
    ("post", "/mailboxes", {"address": "a@cust.com", "control_mailbox": "b@cust.com"}, "add"),
    ("post", "/mailboxes/1/verify", {"control_mailbox": "b@cust.com"}, "verify"),
    ("post", "/mailboxes/1/test-send", None, "test_send"),
    ("patch", "/mailboxes/1", {"is_default": True}, "default"),
    ("delete", "/mailboxes/1", None, "remove"),
    ("delete", "", None, "disconnect"),
    ("patch", "/policy", {"changes": {"poll_seconds": 90}, "version": 1}, "policy_save"),
]


@pytest.fixture
def reviewer(monkeypatch):
    import uuid
    from api.main import app
    from src.services import rbac

    class Reviewer:
        subject = f"test-secrev-{uuid.uuid4().hex[:12]}"
        email = "reviewer@cust.com"
        username = "reviewer"
        claims = {"cognito:groups": ["PROCWISE_SECURITY_REVIEWER"]}

    rbac.reset_policy_cache()
    app.dependency_overrides[r.require_user] = lambda: Reviewer()
    monkeypatch.setattr(r, "get_provider", lambda conn: _Provider())
    c = TestClient(app, raise_server_exceptions=False)
    c.subject = Reviewer.subject
    yield c
    app.dependency_overrides.pop(r.require_user, None)
    rbac.reset_policy_cache()


def _denied_rows(subject):
    from services.db import get_conn
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("""SELECT details->>'op' FROM proc.bp_agent_actions
                        WHERE phase = 'authorize' AND action_type = 'mailbox.bind'
                          AND status = 'denied' AND details->>'principal' = %s""", (subject,))
        return sorted(row[0] if not isinstance(row, dict) else row["?column?"]
                      for row in cur.fetchall())


@_live
def test_security_reviewer_is_refused_every_write_and_each_refusal_is_audited(reviewer, monkeypatch):
    from services.mail_m365 import admin
    for fn in ("add_mailbox", "verify_mailbox", "test_send", "make_default", "remove_mailbox",
               "disconnect"):
        monkeypatch.setattr(admin, fn, lambda *a, **k: pytest.fail("a write ran past the gate"))
    monkeypatch.setattr(mp, "save", lambda *a, **k: pytest.fail("policy saved past the gate"))
    for method, path, body, _op in WRITES:
        kw = {"json": body} if body is not None else {}
        res = getattr(reviewer, method)(f"{B}{path}", **kw)
        assert res.status_code == 403, (method, path, res.status_code, res.text)
    assert _denied_rows(reviewer.subject) == sorted(op for *_x, op in WRITES)


@_live
def test_security_reviewer_may_read_restrictions_and_the_evidence_report(reviewer):
    res = reviewer.get(f"{B}/restrictions")
    assert res.status_code == 200, res.text
    assert res.json()["canConfigure"] is False
    pdf = reviewer.get(f"{B}/evidence.pdf")
    assert pdf.status_code == 200, pdf.text
    assert pdf.content[:4] == b"%PDF"


# ---- the Microsoft check runs after the database connection is released ----

def test_restrictions_prove_runs_outside_the_connection(wired, monkeypatch):
    events = []

    class Conn:
        def __enter__(self):
            events.append("open"); return self
        def __exit__(self, *a):
            events.append("close"); return False

    class Prov(_Provider):
        def status(self, conn):
            events.append("status"); return super().status(conn)
        def prove(self, conn):
            assert conn is None, "prove must not be handed the request's connection"
            events.append("prove"); return super().prove(conn)

    monkeypatch.setattr(r, "_conn", lambda: Conn())
    monkeypatch.setattr(r, "get_provider", lambda conn: Prov())
    for url in (f"{B}/restrictions", f"{B}/evidence.pdf"):
        events.clear()
        assert wired.get(url).status_code == 200
        assert events == ["open", "status", "close", "prove"], events


def test_evidence_reads_the_mailboxes_once(wired, monkeypatch):
    reads = []

    class Prov(_Provider):
        def status(self, conn):
            reads.append(1); return super().status(conn)
    monkeypatch.setattr(r, "get_provider", lambda conn: Prov())
    assert wired.get(f"{B}/evidence.pdf").status_code == 200
    assert len(reads) == 1
