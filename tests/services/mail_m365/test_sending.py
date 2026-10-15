from datetime import datetime, timezone

import pytest

from services import email_service as es
from services.mail_m365 import sending
from services.style.mailbox import MailboxBinding

NOW = datetime(2026, 10, 15, tzinfo=timezone.utc)
CONNECTED = {"provider": "m365", "status": "connected", "tenant_id": "T1"}
DISCONNECTED = {"provider": "m365", "status": "disconnected", "tenant_id": "T1"}


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
    d = sending.decide(repo=Repo(default=b, any_=b), integration=CONNECTED)
    assert d.kind == "graph" and d.binding is b


def test_connected_but_revoked_is_held():
    b = B(active=False, health="REVOKED", default=False)
    d = sending.decide(repo=Repo(default=None, any_=b), integration=CONNECTED)
    assert d.kind == "held"
    assert d.reason == ("Access to procurement@cust.com was withdrawn in Microsoft 365, "
                        "so this email is held")


def test_connected_without_default_is_held():
    d = sending.decide(repo=Repo(), integration=CONNECTED)
    assert d.kind == "held"
    assert d.reason == ("Microsoft 365 is connected but no default mailbox is chosen yet, "
                        "so this email is held")


def test_degraded_default_is_held():
    b = B(health="DEGRADED")
    d = sending.decide(repo=Repo(default=b, any_=b), integration=CONNECTED)
    assert d.kind == "held"
    assert d.reason == ("The company mailbox procurement@cust.com is temporarily unavailable, "
                        "so this email is held")


def test_disconnected_after_use_is_held_not_ses():  # Review Focus 3
    b = B(active=False, default=False)
    d = sending.decide(repo=Repo(default=None, any_=b), integration=DISCONNECTED)
    assert d.kind == "held"
    assert d.reason == ("Microsoft 365 is disconnected, so this email is held rather than "
                        "sent from another address")


def test_disconnected_but_default_still_flagged_active_is_held():
    b = B()
    d = sending.decide(repo=Repo(default=b, any_=b), integration=DISCONNECTED)
    assert d.kind == "held"


def test_removed_default_with_queued_drafts_is_held_never_ses():  # Review Focus 3
    # The default was removed (no active default) but the company had sent from it before.
    b = B(active=False, default=False)
    for integration in (CONNECTED, DISCONNECTED):
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
                        lambda **k: sending.Decision("held", None, sending.HELD_NO_DEFAULT))
    svc = _Svc()
    res = svc.send_email("s", "<p>b</p>", ["sup@x.com"], "ignored@procwise.co.uk", company_mailbox=True)
    assert res.success is False and res.held_reason == sending.HELD_NO_DEFAULT
    assert svc.smtp_calls == 0


def test_decision_failure_is_held_not_ses(monkeypatch):
    def boom(**k):
        raise RuntimeError("db down")
    monkeypatch.setattr(sending, "decide", boom)
    svc = _Svc()
    res = svc.send_email("s", "b", ["sup@x.com"], "x@y", company_mailbox=True)
    assert res.success is False and svc.smtp_calls == 0
    assert res.held_reason == sending.HELD_UNCHECKED and "db down" not in res.held_reason


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
    assert res.success is False and svc.smtp_calls == 0
    # plain words for the buyer; Microsoft's text goes to the log only
    assert res.held_reason == "Microsoft 365 did not accept the email; your admin has been told"


def test_send_via_graph_sends_from_the_binding_mailbox():
    calls = []

    class Client:
        def send_mime(self, mailbox, mime):
            calls.append((mailbox, mime))

    sending.send_via_graph(B(), b"MIME", client_for=lambda: Client())
    assert calls == [("procurement@cust.com", b"MIME")]


def test_send_via_graph_reuses_the_cached_reader_client(monkeypatch):
    """No Secrets Manager read and token per email: the receiving side's cached client."""
    from services.mail_m365 import receiving
    sent = []

    class Client:
        def send_mime(self, mailbox, mime):
            sent.append(mailbox)
    shared = Client()
    monkeypatch.setattr(receiving, "_default_client_for", lambda: (lambda: shared))
    sending.send_via_graph(B(), b"a")
    sending.send_via_graph(B(), b"b")
    assert sent == ["procurement@cust.com"] * 2


def test_unflagged_send_never_decides_and_uses_smtp_even_when_connected(monkeypatch):
    def boom(**k):
        raise AssertionError("decide must not run for non-supplier mail")
    monkeypatch.setattr(sending, "decide", boom)
    svc = _Svc()
    res = svc.send_email("s", "b", ["ops@x.com"], "alerts@procwise.co.uk")
    assert res.success and res.transport == "ses" and svc.smtp_calls == 2


def test_not_connected_client_is_a_clear_error(monkeypatch):
    from services.mail_m365 import receiving
    from services.style.graph_source import MailboxUnreachable
    import pytest
    monkeypatch.setattr(receiving, "_default_client_for", lambda: None)
    with pytest.raises(MailboxUnreachable, match="not connected"):
        sending.send_via_graph(B(), b"x")


# ---- I3: Graph's sendMail size limit ----

def test_oversize_mime_is_refused_before_any_request():
    import pytest
    from services.mail_m365.graph_client import MessageTooLarge

    def must_not_build():
        raise AssertionError("no client (secret read, token) for a message that cannot be sent")
    with pytest.raises(MessageTooLarge):
        sending.send_via_graph(B(), b"x" * (sending.MAX_MIME_BYTES + 1), client_for=must_not_build)


def test_mime_at_the_limit_is_sent():
    sent = []

    class Client:
        def send_mime(self, mailbox, mime):
            sent.append(len(mime))
    sending.send_via_graph(B(), b"x" * sending.MAX_MIME_BYTES, client_for=lambda: Client())
    assert sent == [sending.MAX_MIME_BYTES]


def test_oversize_send_is_held_with_the_size_reason(monkeypatch):
    from services.mail_m365 import receiving

    class Client:
        def send_mime(self, mailbox, mime):
            raise AssertionError("must not reach Graph")
    b = B()
    monkeypatch.setattr(sending, "decide", lambda **k: sending.Decision("graph", b, None))
    monkeypatch.setattr(receiving, "_default_client_for", lambda: (lambda: Client()))
    svc = _Svc()
    big = [(b"\0" * 2_400_000, "drawings.pdf")]   # base64 in the MIME takes it past 3 MB
    res = svc.send_email("s", "b", ["sup@x.com"], "x@y", big, company_mailbox=True)
    assert res.success is False and svc.smtp_calls == 0
    assert res.held_reason == ("This email is too large to send from your Microsoft 365 "
                               "mailbox (limit 3 MB including attachments)")


def test_graph_413_is_the_size_reason_too(monkeypatch):
    from services.mail_m365.graph_client import M365Client, MessageTooLarge
    from tests.services.mail_m365.certs import fake_credentials
    import pytest

    class T:
        def post_form(self, url, data):
            return 200, {"access_token": "tok", "expires_in": 3600}

        def post_body(self, url, headers, body):
            return 413, {"error": {"message": "RequestEntityTooLarge"}}
    c = M365Client(fake_credentials(), transport=T())
    with pytest.raises(MessageTooLarge) as e:
        c.send_mime("procurement@cust.com", b"small")
    assert sending.held_reason_for(e.value) == sending.HELD_TOO_LARGE


# ---- I5: a database the m365 migration has not reached ----

class _ExplodingRepo:
    def __getattr__(self, name):
        raise AssertionError(f"binding columns read ({name}) for a never-connected customer")


@pytest.mark.parametrize("integration", [{}, {"pending_nonces": ["n1"]}])
def test_never_connected_is_ses_without_touching_bindings(integration):
    assert sending.decide(repo=_ExplodingRepo(), integration=integration).kind == "ses"


def test_never_connected_is_ses_on_a_real_connection_path(monkeypatch):
    from services.mail_m365 import config
    monkeypatch.setattr(config, "read_integration", lambda c: {})
    monkeypatch.setattr(sending, "MailboxBindingRepository",
                        lambda conn: _ExplodingRepo())
    assert sending.decide(object()).kind == "ses"
