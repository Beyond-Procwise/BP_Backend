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


class HealthRepo:
    """Bindings by id with their health; records every record_health call."""

    def __init__(self, **states):
        from services.style.mailbox import MailboxBinding
        self.rows = {}
        for i, (addr, (state, active)) in enumerate(states.items(), start=1):
            self.rows[i] = MailboxBinding(i, "org", "graph", f"{addr}@cust.com", "send_receive",
                                          "arn:aws:secretsmanager:x", None, SINCE, "ev", None,
                                          state, active)
        self.recorded = []

    def get(self, binding_id):
        return self.rows.get(binding_id)

    def record_health(self, binding_id, state):
        self.recorded.append((binding_id, state))


def _poll_with(monkeypatch, repo, throttle=()):
    bindings = [{"binding_id": i, "mailbox_address": b.mailbox_address, "last_sync_at": SINCE}
                for i, b in repo.rows.items()]
    monkeypatch.setattr("services.mail_intake.poll.active_bindings",
                        lambda cur, provider, roles: bindings)

    class Router:
        def list_inbox(self, mailbox, since):
            return Client(throttle=mailbox in throttle).list_inbox(mailbox, since)
        def get_mime(self, mailbox, mid): return MIME
    monkeypatch.setattr(receiving, "_default_client_for", lambda: (lambda: Router()))
    conn = FakeConn()
    receiving.poll_m365(conn, Pipeline(), repo=repo)
    return conn


def test_a_successful_poll_lifts_degraded_to_ok(monkeypatch):
    """I6: one 429/5xx in the hourly probe must not hold supplier email for an hour."""
    repo = HealthRepo(a=("DEGRADED", True), b=("OK", True))
    conn = _poll_with(monkeypatch, repo)
    assert repo.recorded == [(1, "OK")]          # only the degraded one, and only to OK
    assert conn.commits >= 1


def test_a_poll_never_lifts_revoked_or_inactive(monkeypatch):
    repo = HealthRepo(a=("REVOKED", False), b=("DEGRADED", False), c=("REVOKED", True))
    _poll_with(monkeypatch, repo)
    assert repo.recorded == []


def test_a_throttled_poll_does_not_lift_degraded(monkeypatch):
    repo = HealthRepo(a=("DEGRADED", True))
    _poll_with(monkeypatch, repo, throttle=("a@cust.com",))
    assert repo.recorded == []


def test_active_bindings_defaults_to_intake_and_accepts_roles():
    from services.mail_intake.poll import active_bindings
    seen = []

    class Cur:
        def execute(self, sql, params): seen.append(params)
        def fetchall(self): return []
    active_bindings(Cur())
    active_bindings(Cur(), provider="graph", roles=("send_receive",))
    assert seen == [(["intake"], "relay"), (["send_receive"], "graph")]


# --- fix round 1 -------------------------------------------------------------------------

import asyncio  # noqa: E402

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_caches():
    receiving._reset_caches()
    yield
    receiving._reset_caches()


def test_second_pass_downloads_nothing_but_still_returns_the_reply():
    client = Client()

    class Repo:
        def list_org(self): return [_binding(1, "procurement@cust.com")]
    first = receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: client)
    second = receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: client)
    assert client.fetched == ["m1"], "full MIME is downloaded once per message"
    assert len(first) == len(second) == 1 and "455" in second[0].body
    # each pass (each workflow) gets its own object, so one cannot alter another's
    assert first[0] is not second[0]


def test_only_graph_send_receive_bindings_are_read():
    from services.style.mailbox import MailboxBinding
    other = MailboxBinding(3, "org", "relay", "x@cust.com", "send_receive", "arn:aws:secretsmanager:x",
                           None, SINCE, "ev", None, "OK", True)

    class Repo:
        def list_org(self): return [other]
    client = Client()
    assert receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: client) == []
    assert client.fetched == []


def test_client_is_built_once_within_the_ttl(monkeypatch):
    import services.mail_m365.admin as admin
    import services.mail_m365.config as config
    import services.db as db
    built = []
    settings = config.M365Settings("cid", "arn:aws:secretsmanager:s", "https://x", "arn:k", 120)
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(config, "read_integration",
                        lambda c: {"status": "connected", "tenant_id": "t1"})

    class Ctx:
        def __enter__(self): return object()
        def __exit__(self, *a): return False
    monkeypatch.setattr(db, "get_conn", lambda: Ctx())
    monkeypatch.setattr(admin, "_default_factory",
                        lambda s: (lambda tenant: built.append(tenant) or Client()))
    a = receiving._default_client_for()
    b = receiving._default_client_for()
    assert built == ["t1"] and a() is b()


def test_client_is_rebuilt_after_the_ttl(monkeypatch):
    import services.mail_m365.admin as admin
    import services.mail_m365.config as config
    import services.db as db
    built = []
    settings = config.M365Settings("cid", "arn:aws:secretsmanager:s", "https://x", "arn:k", 120)
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    monkeypatch.setattr(config, "read_integration",
                        lambda c: {"status": "connected", "tenant_id": "t1"})

    class Ctx:
        def __enter__(self): return object()
        def __exit__(self, *a): return False
    monkeypatch.setattr(db, "get_conn", lambda: Ctx())
    monkeypatch.setattr(admin, "_default_factory",
                        lambda s: (lambda tenant: built.append(tenant) or Client()))
    receiving._default_client_for()
    receiving._client_cache["c"]["at"] -= receiving.CLIENT_TTL_SECONDS + 1
    receiving._default_client_for()
    assert built == ["t1", "t1"]


def _bare_watcher(fetcher=None):
    from services.email_watcher import EmailWatcher, EmailWatcherConfig
    w = EmailWatcher.__new__(EmailWatcher)
    w.config = EmailWatcherConfig()
    w._fetcher = fetcher
    return w


def test_watcher_appends_graph_replies_and_keeps_imap(monkeypatch):
    from services.email_watcher import parse_email_bytes
    imap = parse_email_bytes(MIME)
    graph = parse_email_bytes(MIME.replace(b"455", b"999"))
    monkeypatch.setattr(receiving, "fetch_replies", lambda since, **kw: [graph])
    got = asyncio.run(_bare_watcher(lambda since: [imap])._fetch_emails(SINCE))
    assert got == [imap, graph]


def test_watcher_keeps_imap_results_when_graph_raises(monkeypatch):
    from services.email_watcher import parse_email_bytes
    imap = parse_email_bytes(MIME)

    def boom(since, **kw): raise RuntimeError("graph down")
    monkeypatch.setattr(receiving, "fetch_replies", boom)
    assert asyncio.run(_bare_watcher(lambda since: [imap])._fetch_emails(SINCE)) == [imap]


def test_not_configured_warning_only_without_imap_and_without_m365(monkeypatch, caplog):
    monkeypatch.setattr(receiving, "fetch_replies", lambda since, **kw: [])
    monkeypatch.setattr(receiving, "has_bindings", lambda: True)
    with caplog.at_level("WARNING"):
        asyncio.run(_bare_watcher()._fetch_emails(SINCE))
    assert "not configured" not in caplog.text
    monkeypatch.setattr(receiving, "has_bindings", lambda: False)
    with caplog.at_level("WARNING"):
        asyncio.run(_bare_watcher()._fetch_emails(SINCE))
    assert "not configured" in caplog.text


def _scheduler(monkeypatch):
    from services import backend_scheduler as bs
    s = bs.BackendScheduler.__new__(bs.BackendScheduler)
    s._jobs = {}
    registered = []
    monkeypatch.setattr(bs.BackendScheduler, "register_job",
                        lambda self, name, fn, **kw: registered.append((name, kw)))
    monkeypatch.setattr(bs.BackendScheduler, "_mail_intake_tables_present", staticmethod(lambda: True))
    return s, registered


def test_scheduler_skips_m365_job_when_switched_off(monkeypatch):
    s, registered = _scheduler(monkeypatch)
    monkeypatch.setenv("M365_POLL", "off")
    s._register_m365_poll_job()
    assert registered == []


def test_scheduler_skips_m365_job_without_settings(monkeypatch):
    import src.services.mail_m365.config as config
    s, registered = _scheduler(monkeypatch)
    monkeypatch.delenv("M365_POLL", raising=False)
    monkeypatch.setattr(config, "load_settings", lambda: None)
    s._register_m365_poll_job()
    assert registered == []


def test_scheduler_registers_m365_job_with_poll_seconds(monkeypatch):
    import src.services.mail_m365.config as config
    s, registered = _scheduler(monkeypatch)
    monkeypatch.delenv("M365_POLL", raising=False)
    monkeypatch.setattr(config, "load_settings", lambda: config.M365Settings(
        "cid", "arn:aws:secretsmanager:s", "https://x", "arn:k", 90))
    s._register_m365_poll_job()
    assert registered[0][0] == "mail-m365-poll" and registered[0][1]["interval"].total_seconds() == 90


def test_m365_job_runs_the_shared_handover_steps(monkeypatch):
    """Attachments must reach extraction even when MAIL_INTAKE_POLL=off."""
    import contextlib
    from services import backend_scheduler as bs
    import src.services.mail_m365.receiving as rcv
    import src.services.mail_intake.store as store_mod
    import src.services.mail_intake.pipeline as pipe_mod
    import src.services.sell_side._db as sdb
    import config.settings as cs
    s, _ = _scheduler(monkeypatch)
    calls = []
    monkeypatch.setattr(bs.BackendScheduler, "_run_mail_handover_steps",
                        staticmethod(lambda store: calls.append(("handover", store))))
    monkeypatch.setattr(rcv, "poll_m365", lambda conn, pipeline: calls.append("poll") or {"bindings": 0})
    monkeypatch.setattr(store_mod, "ObjectStore", lambda bucket: ("store", bucket))
    monkeypatch.setattr(pipe_mod, "IngestPipeline", lambda *a, **k: object())
    monkeypatch.setattr(sdb, "transactional_conn", contextlib.nullcontext)
    monkeypatch.setattr(cs, "Settings", lambda: SimpleNamespace(s3_bucket_name="b"))
    s._run_m365_poll()
    assert calls == ["poll", ("handover", ("store", "b"))]
