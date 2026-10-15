"""Emails to colleagues: stored pending, sent from the default shared mailbox only when
their author approves, under the email policy as it is at that moment."""
import copy
import email
import os
import uuid
from types import SimpleNamespace

import pytest

from services import internal_email, mail_policy
from services.style.mailbox import HEALTH_OK, MailboxBinding

LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").lower() in ("1", "true", "yes", "on")
live = pytest.mark.skipif(not LIVE, reason="needs a real PostgreSQL (PROCWISE_TEST_LIVE_DB=1)")

DEFAULT = MailboxBinding(
    binding_id=7, user_ref="org", provider="graph", mailbox_address="procurement@yourcompany.com",
    role="send_receive", credential_ref="arn:aws:secretsmanager:eu-west-1:1:secret:x",
    scope_policy_ref=None, scope_verified_at=None, scope_evidence_ref=None,
    last_health_check=None, health_state=HEALTH_OK, is_active=True, is_default=True)


class FakeProvider:
    def __init__(self, limit=3_000_000, fail=None):
        self.sent, self.limit, self.fail = [], limit, fail

    def size_limit(self):
        return self.limit

    def send(self, binding, mime):
        if self.fail:
            raise self.fail
        m = email.message_from_bytes(mime)
        self.sent.append({"from": m["From"], "to": m["To"], "subject": m["Subject"],
                          "mailbox": binding.mailbox_address})


class FakeStore:
    """In memory, with the same "only from this status" rule the SQL has."""

    def __init__(self):
        self.rows, self.next_id = {}, 1

    def get(self, i):
        r = self.rows.get(i)
        return copy.deepcopy(r) if r else None

    def insert(self, **f):
        i, self.next_id = self.next_id, self.next_id + 1
        self.rows[i] = {"internal_email_id": i, "status": "pending", "held_reason": None,
                        "mailbox_address": None, "approved_by": None, "approved_at": None,
                        "sent_at": None, "message_id": None, "created_at": None, **f}
        return self.get(i)

    def pending_for(self, author):
        return [self.get(i) for i, r in self.rows.items()
                if r["author"].lower() == author.lower() and r["status"] == "pending"]

    def _move(self, i, frm, **changes):
        r = self.rows.get(i)
        if r is None or r["status"] != frm:
            return None
        r.update(changes)
        return self.get(i)

    def update_content(self, i, *, author, recipients, subject, body, content_hash):
        if self.rows[i]["author"].lower() != author.lower():
            return None
        return self._move(i, "pending", recipients=recipients, subject=subject, body=body,
                          content_hash=content_hash)

    def decline(self, i, *, author):
        if self.rows[i]["author"].lower() != author.lower():
            return None
        return self._move(i, "pending", status="declined")

    def hold(self, i, reason, *, from_status, mailbox):
        r = self._move(i, from_status, status="held", held_reason=reason)
        if r is not None and mailbox:
            self.rows[i]["mailbox_address"] = mailbox
            r = self.get(i)
        return r

    def claim(self, i, *, approver, mailbox, content_hash):
        if self.rows[i]["content_hash"] != content_hash:
            return None
        return self._move(i, "pending", status="approved", approved_by=approver,
                          mailbox_address=mailbox, approved_at="now")

    def mark_sent(self, i, *, message_id):
        return self._move(i, "approved", status="sent", message_id=message_id, sent_at="now")


@pytest.fixture
def policy(monkeypatch):
    p = SimpleNamespace(internal_send="with_user_approval", company_domains=("yourcompany.com",))
    loads = []

    def load(conn=None):
        loads.append(1)
        return p
    monkeypatch.setattr(mail_policy, "load", load)
    p.loads = loads
    return p


@pytest.fixture
def mailbox(monkeypatch):
    state = SimpleNamespace(default=DEFAULT, integration={"provider": "m365", "status": "connected"})

    class Repo:
        def __init__(self, conn=None):
            pass

        def get_default_sender(self):
            return state.default
    monkeypatch.setattr(internal_email, "MailboxBindingRepository", Repo)
    monkeypatch.setattr(internal_email, "read_integration", lambda conn: state.integration)
    return state


@pytest.fixture
def audit(monkeypatch):
    rows = []
    monkeypatch.setattr(internal_email, "record_action", lambda **k: rows.append(k))
    return rows


class Svc:
    def __init__(self, provider, store=None, conn=None):
        self.provider, self.store, self.conn = provider, store, conn

    def create(self, **k):
        return internal_email.create(self.conn, **k)

    def list_pending(self, **k):
        return internal_email.list_pending(self.conn, **k)

    def edit(self, i, **k):
        return internal_email.edit(self.conn, i, **k)

    def decline(self, i, **k):
        return internal_email.decline(self.conn, i, **k)

    def approve_and_send(self, i, **k):
        return internal_email.approve_and_send(self.conn, i, provider=self.provider, **k)


@pytest.fixture
def svc(monkeypatch, policy, mailbox, audit):
    store = FakeStore()
    monkeypatch.setattr(internal_email, "_store", lambda conn: store)
    return Svc(FakeProvider(), store)


def _new(svc, author="b@yourcompany.com", recipients=("f@yourcompany.com",), **k):
    fields = dict(kind="forward", subject="s", body="b", attachments=[], source_ref=None)
    fields.update(k)
    return svc.create(author=author, recipients=list(recipients), **fields)


def _refused(fn, *a, **k):
    with pytest.raises(internal_email.InternalEmailRefused) as r:
        fn(*a, **k)
    return r.value


# ---- the brief's five ----

def test_only_the_author_can_approve(svc):                              # Review Focus 2
    e = svc.create(author="buyer@yourcompany.com", kind="forward", recipients=["finance@yourcompany.com"], subject="s", body="b", attachments=[], source_ref=None)
    with pytest.raises(internal_email.InternalEmailRefused) as r:
        svc.approve_and_send(e["internal_email_id"], approver="admin@yourcompany.com")
    assert r.value.code == "not_author"
    assert r.value.message == "Only the person who wrote this email can approve it"
    assert svc.provider.sent == [] and svc.store.rows[e["internal_email_id"]]["status"] == "pending"


def test_policy_switched_off_after_create_holds_the_send(svc, policy):  # Review Focus 1
    e = svc.create(author="b@yourcompany.com", kind="forward", recipients=["f@yourcompany.com"], subject="s", body="b", attachments=[], source_ref=None)
    policy.internal_send = "off"
    with pytest.raises(internal_email.InternalEmailRefused) as r:
        svc.approve_and_send(e["internal_email_id"], approver="b@yourcompany.com")
    assert r.value.code == "policy_off" and svc.provider.sent == []


def test_external_recipient_refused_at_create(svc):
    with pytest.raises(internal_email.InternalEmailRefused) as r:
        svc.create(author="b@yourcompany.com", kind="forward", recipients=["x@gmail.com"], subject="s", body="b", attachments=[], source_ref=None)
    assert r.value.code == "not_internal"
    assert r.value.message == "x@gmail.com is outside your company. Use the normal supplier email instead."


def test_edit_changes_the_hash_and_approval_covers_the_new_content(svc):
    e = svc.create(author="b@yourcompany.com", kind="reply", recipients=["f@yourcompany.com"], subject="s", body="b", attachments=[], source_ref=None)
    e2 = svc.edit(e["internal_email_id"], author="b@yourcompany.com", subject="s2", body="b2", recipients=["f@yourcompany.com"])
    assert e2["content_hash"] != e["content_hash"]
    svc.approve_and_send(e["internal_email_id"], approver="B@YourCompany.com")
    assert svc.provider.sent[0]["subject"] == "s2"


def test_approved_email_goes_from_the_default_shared_mailbox(svc):
    e = svc.create(author="b@yourcompany.com", kind="forward", recipients=["f@yourcompany.com"], subject="s", body="b", attachments=[], source_ref=None)
    svc.approve_and_send(e["internal_email_id"], approver="b@yourcompany.com")
    assert svc.provider.sent[0]["from"] == "procurement@yourcompany.com"


# ---- create ----

def test_create_stores_pending_and_sends_nothing(svc):
    e = _new(svc, recipients=[" Finance <FINANCE@YourCompany.com>", "finance@yourcompany.com"])
    assert e["status"] == "pending" and e["recipients"] == ["finance@yourcompany.com"]
    assert e["content_hash"] and svc.provider.sent == []


def test_create_refused_when_policy_is_off(svc, policy):
    policy.internal_send = "off"
    r = _refused(_new, svc)
    assert r.code == "policy_off" and r.message == "Emails to colleagues are switched off in your email policy"
    assert svc.store.rows == {}


def test_an_unreadable_policy_refuses_and_never_sends(svc, monkeypatch):
    e = _new(svc)
    def boom(conn=None):
        raise mail_policy.PolicyUnavailable("missing")
    monkeypatch.setattr(mail_policy, "load", boom)
    r = _refused(svc.approve_and_send, e["internal_email_id"], approver="b@yourcompany.com")
    assert r.code == "policy_off" and svc.provider.sent == []


@pytest.mark.parametrize("raw", ["not an address", "@yourcompany.com", "a@", ""])
def test_unparseable_recipient_refused_with_its_raw_text(svc, raw):
    r = _refused(_new, svc, recipients=[raw])
    assert r.code == "not_internal" and raw in r.message


def test_no_recipients_refused(svc):
    assert _refused(_new, svc, recipients=[]).code == "not_internal"


def test_attachments_are_not_accepted_yet(svc):
    with pytest.raises(ValueError):
        _new(svc, attachments=[{"filename": "a.pdf"}])


def test_no_email_on_the_sign_in_cannot_create(svc):
    assert _refused(_new, svc, author="").code == "not_author"


# ---- approve ----

def test_author_match_ignores_case_and_records_approver(svc, audit):
    e = _new(svc, author="Buyer@YourCompany.com")
    sent = svc.approve_and_send(e["internal_email_id"], approver="buyer@yourcompany.com")
    assert sent["status"] == "sent" and sent["approved_by"] == "buyer@yourcompany.com"
    assert sent["message_id"] and sent["mailbox_address"] == "procurement@yourcompany.com"
    assert svc.provider.sent[0]["to"] == "f@yourcompany.com"
    assert [a["action_type"] for a in audit] == ["internal_email_sent"]


@pytest.mark.parametrize("approver", [None, "", "   "])
def test_approver_without_an_email_is_not_the_author(svc, approver):
    e = _new(svc)
    assert _refused(svc.approve_and_send, e["internal_email_id"], approver=approver).code == "not_author"


def test_policy_is_read_again_at_approval(svc, policy):
    e = _new(svc)
    before = len(policy.loads)
    svc.approve_and_send(e["internal_email_id"], approver="b@yourcompany.com")
    assert len(policy.loads) == before + 1


def test_recipient_no_longer_inside_the_company_at_approval(svc, policy):
    e = _new(svc)
    policy.company_domains = ("othercompany.com",)
    r = _refused(svc.approve_and_send, e["internal_email_id"], approver="b@yourcompany.com")
    assert r.code == "not_internal" and svc.provider.sent == []


def test_content_changed_without_a_new_hash_is_refused(svc):
    e = _new(svc)
    svc.store.rows[e["internal_email_id"]]["body"] = "something nobody approved"
    r = _refused(svc.approve_and_send, e["internal_email_id"], approver="b@yourcompany.com")
    assert r.code == "changed" and svc.provider.sent == []


def test_unknown_email_is_not_found(svc):
    assert _refused(svc.approve_and_send, 999, approver="b@yourcompany.com").code == "not_found"


def test_second_approval_is_already_done(svc):
    e = _new(svc)
    svc.approve_and_send(e["internal_email_id"], approver="b@yourcompany.com")
    r = _refused(svc.approve_and_send, e["internal_email_id"], approver="b@yourcompany.com")
    assert r.code == "already_done" and len(svc.provider.sent) == 1


@pytest.mark.parametrize("change,reason", [
    ({"default": None}, internal_email.HELD_NO_DEFAULT),
    ({"integration": {"provider": "m365", "status": "disconnected"}}, internal_email.HELD_NOT_CONNECTED),
    ({"default": DEFAULT.__class__(**{**DEFAULT.__dict__, "health_state": "REVOKED"})},
     "The company mailbox procurement@yourcompany.com is not available right now, so this email is held"),
])
def test_no_usable_mailbox_holds_with_a_plain_reason(svc, mailbox, audit, change, reason):
    e = _new(svc)
    for k, v in change.items():
        setattr(mailbox, k, v)
    r = _refused(svc.approve_and_send, e["internal_email_id"], approver="b@yourcompany.com")
    row = svc.store.rows[e["internal_email_id"]]
    assert r.code == "no_mailbox" and r.message == reason
    assert row["status"] == "held" and row["held_reason"] == reason and svc.provider.sent == []
    assert [a["action_type"] for a in audit] == ["internal_email_held"]


def test_send_failure_holds_and_never_returns_to_pending(svc, audit):
    svc.provider.fail = RuntimeError("Graph 500: raw microsoft text")
    e = _new(svc)
    out = svc.approve_and_send(e["internal_email_id"], approver="b@yourcompany.com")
    assert out["status"] == "held" and out["held_reason"] == internal_email.HELD_REJECTED
    assert "microsoft text" not in out["held_reason"]
    assert out["approved_by"] == "b@yourcompany.com"
    assert _refused(svc.approve_and_send, e["internal_email_id"], approver="b@yourcompany.com").code == "already_done"
    assert [a["action_type"] for a in audit] == ["internal_email_held"]


def test_too_large_for_the_provider_is_held(svc):
    svc.provider.limit = 100
    e = _new(svc)
    out = svc.approve_and_send(e["internal_email_id"], approver="b@yourcompany.com")
    assert out["status"] == "held" and svc.provider.sent == []
    assert out["held_reason"] == ("This email is too large to send from your company mailbox "
                                  "(limit 0 MB including attachments)")


# ---- edit / decline / list ----

def test_only_the_author_can_edit_or_decline(svc):
    e = _new(svc)
    i = e["internal_email_id"]
    assert _refused(svc.edit, i, author="x@yourcompany.com", subject="s", body="b",
                    recipients=["f@yourcompany.com"]).code == "not_author"
    assert _refused(svc.decline, i, actor="x@yourcompany.com").code == "not_author"
    assert svc.decline(i, actor="B@yourcompany.com")["status"] == "declined"
    assert _refused(svc.approve_and_send, i, approver="b@yourcompany.com").code == "already_done"


def test_edit_cannot_add_an_outside_recipient(svc):
    e = _new(svc)
    r = _refused(svc.edit, e["internal_email_id"], author="b@yourcompany.com", subject="s",
                 body="b", recipients=["x@gmail.com"])
    assert r.code == "not_internal"


def test_list_pending_is_the_authors_own(svc):
    _new(svc, author="B@yourcompany.com")
    _new(svc, author="other@yourcompany.com")
    assert [r["author"] for r in svc.list_pending(author="b@YOURCOMPANY.com")] == ["B@yourcompany.com"]


# ---- the SQL, against bp_testdb ----

@pytest.fixture
def live_svc(policy, mailbox, audit):
    from services.db import get_conn
    tag = f"ie-test-{uuid.uuid4().hex[:8]}@yourcompany.com"
    with get_conn() as conn:
        s = Svc(FakeProvider(), conn=conn)
        s.author = tag
        try:
            yield s
        finally:
            conn.cursor().execute("DELETE FROM proc.bp_internal_email WHERE author = %s", (tag,))
            if not getattr(conn, "autocommit", True):
                conn.commit()


@live
def test_live_create_list_edit_approve(live_svc):
    s = live_svc
    e = _new(s, author=s.author, recipients=["Finance <FINANCE@yourcompany.com>"])
    assert e["recipients"] == ["finance@yourcompany.com"] and e["status"] == "pending"
    assert [r["internal_email_id"] for r in s.list_pending(author=s.author.upper())] == [e["internal_email_id"]]
    e2 = s.edit(e["internal_email_id"], author=s.author, subject="s2", body="b2",
                recipients=["finance@yourcompany.com"])
    assert e2["content_hash"] != e["content_hash"]
    sent = s.approve_and_send(e["internal_email_id"], approver=s.author.upper())
    assert sent["status"] == "sent" and sent["sent_at"] and sent["message_id"]
    assert sent["mailbox_address"] == "procurement@yourcompany.com"
    assert s.provider.sent[0]["subject"] == "s2"
    assert s.list_pending(author=s.author) == []


@live
def test_live_double_approval_sends_once(live_svc, monkeypatch):
    """Two approvals that both read the row while it was pending: only one may send."""
    s = live_svc
    e = _new(s, author=s.author)
    stale = internal_email._SqlStore(s.conn).get(e["internal_email_id"])
    s.approve_and_send(e["internal_email_id"], approver=s.author)
    monkeypatch.setattr(internal_email._SqlStore, "get", lambda self, i: dict(stale))
    r = _refused(s.approve_and_send, e["internal_email_id"], approver=s.author)
    assert r.code == "already_done" and len(s.provider.sent) == 1


@live
def test_live_edit_racing_an_approval_is_not_sent_under_the_old_check(live_svc, monkeypatch):
    s = live_svc
    e = _new(s, author=s.author)
    stale = internal_email._SqlStore(s.conn).get(e["internal_email_id"])
    s.edit(e["internal_email_id"], author=s.author, subject="new", body="new",
           recipients=["f@yourcompany.com"])
    monkeypatch.setattr(internal_email._SqlStore, "get", lambda self, i: dict(stale))
    r = _refused(s.approve_and_send, e["internal_email_id"], approver=s.author)
    assert r.code == "already_done" and s.provider.sent == []


@live
def test_live_send_failure_is_held(live_svc):
    s = live_svc
    s.provider.fail = RuntimeError("boom")
    e = _new(s, author=s.author)
    out = s.approve_and_send(e["internal_email_id"], approver=s.author)
    assert out["status"] == "held" and out["held_reason"] == internal_email.HELD_REJECTED
