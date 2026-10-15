from datetime import datetime, timezone
from types import SimpleNamespace

from services.mail_m365 import receiving
from services.mail_m365.graph_client import Throttled

SINCE = datetime(2026, 10, 15, 8, tzinfo=timezone.utc)
MIME = (b"From: sup@supplier.com\r\nTo: procurement@cust.com\r\nSubject: Re: RFQ-1\r\n"
        b"Message-ID: <a@supplier.com>\r\nIn-Reply-To: <rfq1@cust.com>\r\n\r\nOur price is 455.")


class Client:
    def __init__(self, throttle=False):
        self.throttle = throttle
        self.fetched = []

    def list_inbox(self, mailbox, since):
        if self.throttle:
            raise Throttled(10)
        return [{"id": "m1", "receivedDateTime": "2026-10-15T09:00:00Z"}]

    def get_mime(self, mailbox, mid):
        self.fetched.append(mid)
        return MIME


def _binding(i, addr):
    from services.style.mailbox import MailboxBinding
    return MailboxBinding(i, "org", "graph", addr, "send_receive", "arn:aws:secretsmanager:x",
                          None, SINCE, "ev", None, "OK", True)


class FakeConn:
    """Non-autocommit connection that records every statement run on it."""
    autocommit = False

    def __init__(self):
        self.sql = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        conn = self

        class Cur:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def execute(self, sql, params=None): conn.sql.append((sql, params))
        return Cur()

    def commit(self): self.commits += 1
    def rollback(self): self.rollbacks += 1


class Pipeline:
    def __init__(self):
        self.calls = []

    def ingest(self, cur, binding, message):
        self.calls.append(message)
        return SimpleNamespace(artifact_ids=[1], quarantined=[], duplicates=[])


def test_source_yields_raw_mime_with_graph_ids():
    src = receiving.GraphMailSource(lambda: Client())
    msgs = list(src.iter_messages({"mailbox_address": "procurement@cust.com"}, SINCE))
    assert msgs[0].raw == MIME and msgs[0].provider_message_id == "m1"
    assert msgs[0].source_ref == "m365:procurement@cust.com"
    assert msgs[0].received_at == datetime(2026, 10, 15, 9, tzinfo=timezone.utc)


def test_replies_are_parsed_for_the_watcher():
    class Repo:
        def list_org(self): return [_binding(1, "procurement@cust.com")]
    got = receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: Client())
    assert len(got) == 1 and "455" in (got[0].body or "")


def test_one_failing_mailbox_does_not_stop_replies():
    class Repo:
        def list_org(self): return [_binding(2, "b@cust.com"), _binding(1, "a@cust.com")]
    clients = {"b@cust.com": Client(throttle=True), "a@cust.com": Client()}

    class Router:
        def list_inbox(self, mailbox, since): return clients[mailbox].list_inbox(mailbox, since)
        def get_mime(self, mailbox, mid): return clients[mailbox].get_mime(mailbox, mid)
    assert len(receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: Router())) == 1


def test_fetch_replies_never_raises_into_the_watcher():
    class Repo:
        def list_org(self): raise RuntimeError("db down")
    assert receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: Client()) == []


def test_fetch_replies_does_nothing_when_m365_is_not_connected(monkeypatch):
    monkeypatch.setattr(receiving, "_default_client_for", lambda: None)

    class Repo:
        def list_org(self): return [_binding(1, "a@cust.com")]
    assert receiving.fetch_replies(SINCE, repo=Repo()) == []


def test_throttled_binding_keeps_its_cursor():          # Review Focus 5
    src = receiving.GraphMailSource(lambda: Client(throttle=True))
    pipeline, conn = Pipeline(), FakeConn()
    binding = {"binding_id": 1, "mailbox_address": "a@cust.com", "last_sync_at": SINCE}
    out = receiving.poll_one(conn, binding, src, pipeline)
    assert out == {"throttled": 1}
    assert pipeline.calls == []
    assert not [s for s, _ in conn.sql if "last_sync_at" in s], "cursor must not move"

    # Next pass, Graph answers: the message that was waiting is ingested and the cursor moves.
    out = receiving.poll_one(conn, binding, receiving.GraphMailSource(lambda: Client()), pipeline)
    assert out["messages"] == 1 and len(pipeline.calls) == 1
    assert pipeline.calls[0].provider_message_id == "m1"
    moved = [p for s, p in conn.sql if "SET last_sync_at" in s]
    assert moved and moved[0][0] == datetime(2026, 10, 15, 9, tzinfo=timezone.utc)


def test_a_throttled_binding_does_not_stop_the_others(monkeypatch):
    bindings = [{"binding_id": 1, "mailbox_address": "b@cust.com", "last_sync_at": SINCE},
                {"binding_id": 2, "mailbox_address": "a@cust.com", "last_sync_at": SINCE}]
    monkeypatch.setattr("services.mail_intake.poll.active_bindings",
                        lambda cur, provider, roles: bindings)
    clients = {"b@cust.com": Client(throttle=True), "a@cust.com": Client()}

    class Router:
        def list_inbox(self, mailbox, since): return clients[mailbox].list_inbox(mailbox, since)
        def get_mime(self, mailbox, mid): return clients[mailbox].get_mime(mailbox, mid)
    monkeypatch.setattr(receiving, "_default_client_for", lambda: (lambda: Router()))
    pipeline, conn = Pipeline(), FakeConn()
    out = receiving.poll_m365(conn, pipeline)
    assert out["throttled"] == 1 and out["messages"] == 1
    assert [m.source_ref for m in pipeline.calls] == ["m365:a@cust.com"]


def test_a_crashing_binding_does_not_stop_the_others(monkeypatch):
    bindings = [{"binding_id": 1, "mailbox_address": "b@cust.com", "last_sync_at": SINCE},
                {"binding_id": 2, "mailbox_address": "a@cust.com", "last_sync_at": SINCE}]
    monkeypatch.setattr("services.mail_intake.poll.active_bindings",
                        lambda cur, provider, roles: bindings)

    class Router:
        def list_inbox(self, mailbox, since):
            if mailbox == "b@cust.com":
                raise ValueError("boom")
            return [{"id": "m1", "receivedDateTime": "2026-10-15T09:00:00Z"}]
        def get_mime(self, mailbox, mid): return MIME
    monkeypatch.setattr(receiving, "_default_client_for", lambda: (lambda: Router()))
    out = receiving.poll_m365(FakeConn(), Pipeline())
    assert out["binding_errors"] == 1 and out["messages"] == 1


def test_active_bindings_defaults_to_intake_and_accepts_roles():
    from services.mail_intake.poll import active_bindings
    seen = []

    class Cur:
        def execute(self, sql, params): seen.append(params)
        def fetchall(self): return []
    active_bindings(Cur())
    active_bindings(Cur(), provider="graph", roles=("send_receive",))
    assert seen == [(["intake"], "relay"), (["send_receive"], "graph")]
