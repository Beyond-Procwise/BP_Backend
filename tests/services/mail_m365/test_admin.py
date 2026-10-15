# tests/services/mail_m365/test_admin.py
from datetime import datetime, timezone

import pytest

from services.mail_m365 import admin
from services.mail_m365.config import M365Settings
from services.style.mailbox import MailboxBinding, ScopeVerificationFailed

S = M365Settings("cid", "arn:aws:secretsmanager:eu-west-1:1:secret:a",
                 "https://app.procwise.co.uk", "arn:aws:secretsmanager:eu-west-1:1:secret:b", 120)


class FakeClient:
    def __init__(self, readable=(), sent=None):
        self.readable = set(readable)
        self.sent = sent if sent is not None else []

    def can_read(self, m):
        return (m in self.readable, "stub")

    def probe_control(self, m):
        return (m in self.readable, "stub")

    def send_mime(self, m, mime):
        self.sent.append((m, mime))


@pytest.fixture
def env(monkeypatch):
    state = {"integration": {"status": "connected", "tenant_id": "t1"}, "rows": {}, "next": 1}

    class Repo:
        def __init__(self, conn=None): pass

        def list_org(self, include_inactive=False):
            return [b for b in state["rows"].values() if include_inactive or b.is_active]

        def create(self, **kw):
            b = MailboxBinding(state["next"], kw["user_ref"], kw["provider"],
                               kw["mailbox_address"], kw["role"], kw["credential_ref"],
                               None, None, None, None, "OK", False)
            state["rows"][b.binding_id] = b
            state["next"] += 1
            return b

        def verify_scope(self, binding_id, *, control_mailbox, probe):
            allowed, _ = probe(control_mailbox)
            if allowed:
                raise ScopeVerificationFailed("over")
            b = state["rows"][binding_id]
            b = MailboxBinding(*[getattr(b, f) for f in (
                "binding_id", "user_ref", "provider", "mailbox_address", "role",
                "credential_ref", "scope_policy_ref")], datetime.now(timezone.utc),
                "ev", None, "OK", True)
            state["rows"][binding_id] = b
            return b

        def get(self, i):
            return state["rows"].get(i)

        def deactivate(self, i, *, reason):
            b = state["rows"][i]
            state["rows"][i] = MailboxBinding(**{**b.__dict__, "is_active": False})
            return state["rows"][i]

        def record_test_send(self, i):
            b = state["rows"][i]
            state["rows"][i] = MailboxBinding(**{**b.__dict__,
                                                 "last_test_send_at": datetime.now(timezone.utc)})
            return state["rows"][i]

    monkeypatch.setattr(admin, "MailboxBindingRepository", Repo)
    monkeypatch.setattr(admin, "merge_integration",
                        lambda conn, f, actor, **kw: state["integration"].update(f) or state["integration"])
    monkeypatch.setattr(admin, "read_integration", lambda conn: dict(state["integration"]))
    return state


def test_add_activates_when_control_is_denied(env):
    c = FakeClient(readable={"procurement@cust.com"})
    out = admin.add_mailbox(None, " Procurement@Cust.com ", "me@cust.com", actor="a",
                            settings=S, client_factory=lambda t: c)
    assert out["mailbox"]["address"] == "procurement@cust.com"   # Review Focus 4
    assert out["mailbox"]["active"] is True


def test_over_scoped_is_refused_in_plain_english(env):
    c = FakeClient(readable={"procurement@cust.com", "me@cust.com"})
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: c)
    assert e.value.code == "over_scoped"
    assert "not restricted" in e.value.message


def test_unreadable_mailbox_is_refused_before_creating_a_binding(env):
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: FakeClient())
    assert e.value.code == "cannot_read"
    assert env["rows"] == {}


def test_duplicate_address_is_refused(env):                      # Review Focus 1
    c = FakeClient(readable={"procurement@cust.com"})
    admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                      settings=S, client_factory=lambda t: c)
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "PROCUREMENT@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: c)
    assert e.value.code == "duplicate"


def test_control_equal_to_mailbox_is_refused(env):              # Review Focus 4
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "Procurement@cust.com", " procurement@CUST.com", actor="a",
                          settings=S, client_factory=lambda t: FakeClient())
    assert e.value.code == "same_as_control"


def test_not_connected_is_refused(env):
    env["integration"] = {}
    with pytest.raises(admin.MailboxRefused) as e:
        admin.add_mailbox(None, "p@c.com", "me@c.com", actor="a", settings=S,
                          client_factory=lambda t: FakeClient())
    assert e.value.code == "not_connected"


def test_test_send_sends_once_and_records_it(env):
    sent = []
    c = FakeClient(readable={"procurement@cust.com"}, sent=sent)
    b = admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: c)["mailbox"]
    admin.test_send(None, b["binding_id"], "me@cust.com", actor="a", settings=S,
                    client_factory=lambda t: c)
    assert len(sent) == 1 and sent[0][0] == "procurement@cust.com"
    assert b"To: me@cust.com" in sent[0][1]
    assert env["rows"][b["binding_id"]].last_test_send_at is not None


def test_disconnect_deactivates_all_and_merges_status(env):
    c = FakeClient(readable={"procurement@cust.com"})
    admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                      settings=S, client_factory=lambda t: c)
    out = admin.disconnect(None, actor="boss")
    assert out["ok"] and all(not b.is_active for b in env["rows"].values())
    assert env["integration"]["status"] == "disconnected"
    assert env["integration"]["tenant_id"] == "t1"
    assert env["integration"]["disconnected_by"] == "boss"


def test_disconnect_writes_only_the_status_fields_via_merge(env, monkeypatch):
    calls = []
    monkeypatch.setattr(admin, "merge_integration",
                        lambda conn, f, actor, **kw: calls.append(f))
    admin.disconnect(None, actor="boss")
    assert calls == [{"status": "disconnected", "disconnected_by": "boss"}]   # no whole-row copy


def test_remove_unknown_mailbox_is_refused(env):
    with pytest.raises(admin.MailboxRefused) as e:
        admin.remove_mailbox(None, 999, actor="a")
    assert e.value.code == "not_found"


def test_verify_refuses_control_equal_to_mailbox_any_case(env):
    c = FakeClient(readable={"procurement@cust.com"})
    b = admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                          settings=S, client_factory=lambda t: c)["mailbox"]
    with pytest.raises(admin.MailboxRefused) as e:
        admin.verify_mailbox(None, b["binding_id"], " PROCUREMENT@cust.com ", actor="a",
                             settings=S, client_factory=lambda t: c)
    assert e.value.code == "same_as_control"


def test_status_lists_mailboxes_when_connected(env):
    c = FakeClient(readable={"procurement@cust.com"})
    admin.add_mailbox(None, "procurement@cust.com", "me@cust.com", actor="a",
                      settings=S, client_factory=lambda t: c)
    out = admin.status(object(), settings=S)
    assert out["connected"] and out["tenant_id"] == "t1"
    assert [m["address"] for m in out["mailboxes"]] == ["procurement@cust.com"]
