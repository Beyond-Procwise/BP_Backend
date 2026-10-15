from datetime import datetime, timezone

from services import email_service as es
from services.mail_m365 import sending
from services.style.mailbox import MailboxBinding

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)


def B(active=True, health="OK", default=True):
    return MailboxBinding(7, "org", "graph", "procurement@cust.com", "send_receive",
                          "arn:aws:secretsmanager:x:1:secret:a", None, NOW, "ev", None,
                          health, active, default, NOW)


class Repo:
    def __init__(self, default=None, any_=None):
        self._d, self._a = default, any_

    def get_default_sender(self):
        return self._d

    def get_org_default_any(self):
        return self._a


def test_not_connected_uses_ses():
    assert sending.decide(repo=Repo(), integration={}).kind == "ses"


def test_connected_and_healthy_uses_graph():
    b = B()
    d = sending.decide(repo=Repo(default=b, any_=b), integration={"status": "connected"})
    assert d.kind == "graph" and d.binding is b


def test_connected_but_revoked_is_held():
    b = B(active=False, health="REVOKED", default=False)
    d = sending.decide(repo=Repo(default=None, any_=b), integration={"status": "connected"})
    assert d.kind == "held" and "REVOKED" in d.reason
    assert d.reason == "company mailbox unavailable: procurement@cust.com is REVOKED"


def test_connected_without_default_is_held():
    d = sending.decide(repo=Repo(), integration={"status": "connected"})
    assert d.kind == "held" and "no default mailbox" in d.reason
    assert d.reason == ("company mailbox unavailable: Microsoft 365 is connected "
                        "but no default mailbox is set")


def test_degraded_default_is_held():
    b = B(health="DEGRADED")
    d = sending.decide(repo=Repo(default=b, any_=b), integration={"status": "connected"})
    assert d.kind == "held"
    assert d.reason == "company mailbox unavailable: procurement@cust.com is DEGRADED"


def test_disconnected_after_use_is_held_not_ses():  # Review Focus 3
    b = B(active=False, default=False)
    d = sending.decide(repo=Repo(default=None, any_=b), integration={"status": "disconnected"})
    assert d.kind == "held"
    assert d.reason == "company mailbox unavailable: procurement@cust.com is disconnected"


def test_disconnected_but_default_still_flagged_active_is_held():
    b = B()
    d = sending.decide(repo=Repo(default=b, any_=b), integration={"status": "disconnected"})
    assert d.kind == "held"


def test_removed_default_with_queued_drafts_is_held_never_ses():  # Review Focus 3
    # The default was removed (no active default) but the company had sent from it before.
    b = B(active=False, default=False)
    for integration in ({"status": "connected"}, {"status": "disconnected"}):
        d = sending.decide(repo=Repo(default=None, any_=b), integration=integration)
        assert d.kind == "held"


class _Svc(es.EmailService):
    def __init__(self):  # skip agent_nick wiring
        import logging
        self.logger = logging.getLogger("t")
        self.smtp_calls = 0

    def _fetch_smtp_credentials(self, **k):
        self.smtp_calls += 1
        return "u", "p"

    def _deliver_via_smtp(self, *a, **k):
        self.smtp_calls += 1


def test_held_send_never_touches_smtp(monkeypatch):
    monkeypatch.setattr(sending, "decide",
                        lambda **k: sending.Decision("held", None, "company mailbox unavailable: x"))
    svc = _Svc()
    res = svc.send_email("s", "<p>b</p>", ["sup@x.com"], "ignored@procwise.co.uk", company_mailbox=True)
    assert res.success is False and res.held_reason.startswith("company mailbox unavailable")
    assert svc.smtp_calls == 0


def test_decision_failure_is_held_not_ses(monkeypatch):
    def boom(**k):
        raise RuntimeError("db down")
    monkeypatch.setattr(sending, "decide", boom)
    svc = _Svc()
    res = svc.send_email("s", "b", ["sup@x.com"], "x@y", company_mailbox=True)
    assert res.success is False and "db down" in res.held_reason and svc.smtp_calls == 0


def test_ses_decision_still_sends_over_smtp(monkeypatch):
    monkeypatch.setattr(sending, "decide", lambda **k: sending.Decision("ses", None, None))
    svc = _Svc()
    res = svc.send_email("s", "b", ["sup@x.com"], "x@y", company_mailbox=True)
    assert res.success and res.transport == "ses" and res.held_reason is None
    assert svc.smtp_calls == 2  # credentials + delivery


def test_graph_send_uses_the_shared_mailbox_as_from(monkeypatch):
    b = B()
    sent = {}
    monkeypatch.setattr(sending, "decide", lambda **k: sending.Decision("graph", b, None))
    monkeypatch.setattr(sending, "send_via_graph",
                        lambda binding, mime, **k: sent.update(addr=binding.mailbox_address, mime=mime))
    svc = _Svc()
    res = svc.send_email("s", "<p>b</p>", ["sup@x.com"], "ignored@procwise.co.uk",
                         headers={"X-Procwise-RFQ-ID": "RFQ-1"}, company_mailbox=True)
    assert res.success and res.transport == "graph" and svc.smtp_calls == 0
    assert b"From: procurement@cust.com" in sent["mime"]
    assert b"ignored@procwise.co.uk" not in sent["mime"]
    assert b"X-Procwise-RFQ-ID: RFQ-1" in sent["mime"]


def test_graph_failure_is_held_not_retried_on_ses(monkeypatch):
    b = B()
    monkeypatch.setattr(sending, "decide", lambda **k: sending.Decision("graph", b, None))

    def boom(*a, **k):
        raise RuntimeError("HTTP 500")
    monkeypatch.setattr(sending, "send_via_graph", boom)
    svc = _Svc()
    res = svc.send_email("s", "b", ["sup@x.com"], "x@y", company_mailbox=True)
    assert res.success is False and "HTTP 500" in res.held_reason and svc.smtp_calls == 0
    assert res.held_reason == "company mailbox unavailable: HTTP 500"


def test_send_via_graph_sends_from_the_binding_mailbox(monkeypatch):
    from contextlib import contextmanager
    from services import db
    from services.mail_m365 import config

    class _C:
        pass

    @contextmanager
    def fake_conn():
        yield _C()
    monkeypatch.setattr(db, "get_conn", fake_conn)
    monkeypatch.setattr(config, "read_integration", lambda c: {"tenant_id": "T1"})
    calls = []

    class Client:
        def send_mime(self, mailbox, mime):
            calls.append((mailbox, mime))

    sending.send_via_graph(B(), b"MIME", client_factory=lambda t: calls.append(t) or Client(),
                           settings=object())
    assert calls == ["T1", ("procurement@cust.com", b"MIME")]


def test_unflagged_send_never_decides_and_uses_smtp_even_when_connected(monkeypatch):
    def boom(**k):
        raise AssertionError("decide must not run for non-supplier mail")
    monkeypatch.setattr(sending, "decide", boom)
    svc = _Svc()
    res = svc.send_email("s", "b", ["ops@x.com"], "alerts@procwise.co.uk")
    assert res.success and res.transport == "ses" and svc.smtp_calls == 2


def test_missing_tenant_is_a_clear_error(monkeypatch):
    from contextlib import contextmanager
    from services import db
    from services.mail_m365 import config
    from services.style.graph_source import MailboxUnreachable

    @contextmanager
    def fake_conn():
        yield object()
    monkeypatch.setattr(db, "get_conn", fake_conn)
    monkeypatch.setattr(config, "read_integration", lambda c: {})
    import pytest
    with pytest.raises(MailboxUnreachable, match="tenant not recorded"):
        sending.send_via_graph(B(), b"x", client_factory=lambda t: None, settings=object())
