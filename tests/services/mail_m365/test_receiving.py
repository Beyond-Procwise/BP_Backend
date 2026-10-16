from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from services.mail_m365 import receiving
from services.mail_m365.graph_client import Throttled

SINCE = datetime(2026, 10, 15, 8, tzinfo=timezone.utc)
MIME = (b"From: sup@supplier.com\r\nTo: procurement@cust.com\r\nSubject: Re: RFQ-1\r\n"
        b"Message-ID: <a@supplier.com>\r\nIn-Reply-To: <rfq1@cust.com>\r\n\r\nOur price is 455.")


@pytest.fixture(autouse=True)
def _whole_inbox_by_default(monkeypatch):
    """The older tests here are about plumbing, not scope: read the whole inbox."""
    monkeypatch.setattr(receiving, "_read_scope", lambda: "whole_inbox")


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


# ---- Task 8: read scope -------------------------------------------------------------------

class ScopedClient:
    """supplier, a reply to a mail we sent, and an unrelated colleague."""
    def __init__(self):
        self.fetched, self.header_calls = [], []

    def list_inbox(self, mailbox, since):
        return [{"id": "sup", "receivedDateTime": "2026-10-15T09:00:00Z", "from": "sup@supplier.com"},
                {"id": "rep", "receivedDateTime": "2026-10-15T09:01:00Z", "from": "someone@gmail.com"},
                {"id": "col", "receivedDateTime": "2026-10-15T09:02:00Z", "from": "colleague@cust.com"}]

    def get_headers(self, mailbox, mid):
        self.header_calls.append(mid)
        if mid == "rep":
            return {"in-reply-to": "<ours-1@cust.com>", "references": "<old@x> <ours-1@cust.com>"}
        return {"in-reply-to": "<unrelated@elsewhere>"}

    def get_mime(self, mailbox, mid):
        self.fetched.append(mid)
        return MIME


def _scope(monkeypatch, scope):
    monkeypatch.setattr(receiving, "_read_scope", scope if callable(scope) else (lambda: scope))
    monkeypatch.setattr(receiving, "_default_checks", lambda: (
        lambda addr: addr == "sup@supplier.com",
        lambda ids: any(i in ("<ours-1@cust.com>", "ours-1@cust.com") for i in ids)))


def _ids(src_client):
    src = receiving.GraphMailSource(lambda: src_client)
    list(src.iter_messages({"mailbox_address": "procurement@cust.com"}, SINCE))
    return src_client.fetched


def test_replies_and_suppliers_never_downloads_an_unrelated_colleague(monkeypatch):
    _scope(monkeypatch, "replies_and_suppliers")
    c = ScopedClient()
    assert _ids(c) == ["sup", "rep"]
    assert "sup" not in c.header_calls, "a known supplier needs no header lookup"


def test_whole_inbox_downloads_everything_without_looking_at_headers(monkeypatch):
    _scope(monkeypatch, "whole_inbox")
    c = ScopedClient()
    assert _ids(c) == ["sup", "rep", "col"] and c.header_calls == []


def test_unreadable_policy_reads_the_narrower_scope(monkeypatch):
    from services.mail_policy import PolicyUnavailable
    monkeypatch.undo()
    monkeypatch.setattr("services.mail_policy.load",
                        lambda conn=None: (_ for _ in ()).throw(PolicyUnavailable("down")))
    assert receiving._read_scope() == "replies_and_suppliers"
    _scope(monkeypatch, receiving._read_scope)
    assert _ids(ScopedClient()) == ["sup", "rep"]


def test_the_watcher_path_applies_the_same_scope(monkeypatch):
    _scope(monkeypatch, "replies_and_suppliers")
    c = ScopedClient()

    class Repo:
        def list_org(self): return [_binding(1, "procurement@cust.com")]
    got = receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: c)
    assert c.fetched == ["sup", "rep"] and len(got) == 2
    receiving.fetch_replies(SINCE, repo=Repo(), client_for=lambda: c)
    assert c.header_calls.count("col") == 1, "an out-of-scope message is judged once, not every 30 s"


def test_in_scope_a_failed_lookup_leaves_the_message_unread():
    class Boom(ScopedClient):
        def get_headers(self, mailbox, mid): raise RuntimeError("graph down")
    item = {"id": "x", "from": "a@b.com"}
    assert receiving.in_scope(item, policy=SimpleNamespace(read_scope="replies_and_suppliers"),
                              client=Boom(), mailbox="m@x", is_supplier=lambda a: False,
                              sent_ids=lambda ids: True) is False


# ---- round 1: a failed check is retried, never remembered as a no ----------------------

class _Flaky(ScopedClient):
    def __init__(self, fail_headers=False):
        super().__init__()
        self.fail_headers = fail_headers

    def list_inbox(self, mailbox, since):
        return [{"id": "rep", "receivedDateTime": "2026-10-15T09:00:00Z", "from": "sup@supplier.com"}]

    def get_headers(self, mailbox, mid):
        self.header_calls.append(mid)
        if self.fail_headers:
            raise RuntimeError("graph hiccup")
        return {"in-reply-to": "<unrelated@elsewhere>"}


class _Repo:
    def list_org(self): return [_binding(1, "procurement@cust.com")]


def test_a_failed_supplier_lookup_is_retried_next_pass(monkeypatch):
    monkeypatch.setattr(receiving, "_read_scope", lambda: "replies_and_suppliers")
    state = {"db_down": True}

    def is_supplier(addr):
        if state["db_down"]:
            raise RuntimeError("db down")
        return addr == "sup@supplier.com"
    monkeypatch.setattr(receiving, "_default_checks", lambda: (is_supplier, lambda ids: False))
    c = _Flaky(fail_headers=True)
    assert receiving.fetch_replies(SINCE, repo=_Repo(), client_for=lambda: c) == [] and c.fetched == []
    state["db_down"] = False
    assert len(receiving.fetch_replies(SINCE, repo=_Repo(), client_for=lambda: c)) == 1
    assert c.fetched == ["rep"]


def test_a_failed_header_fetch_is_retried_next_pass(monkeypatch):
    monkeypatch.setattr(receiving, "_read_scope", lambda: "replies_and_suppliers")
    monkeypatch.setattr(receiving, "_default_checks", lambda: (
        lambda a: False, lambda ids: "ours-1@cust.com" in ids))
    c = _Flaky(fail_headers=True)
    assert receiving.fetch_replies(SINCE, repo=_Repo(), client_for=lambda: c) == [] and c.fetched == []
    c.fail_headers = False
    c.get_headers = lambda m, i: {"in-reply-to": "<ours-1@cust.com>"}
    assert len(receiving.fetch_replies(SINCE, repo=_Repo(), client_for=lambda: c)) == 1


def test_a_definite_no_is_still_remembered(monkeypatch):
    monkeypatch.setattr(receiving, "_read_scope", lambda: "replies_and_suppliers")
    monkeypatch.setattr(receiving, "_default_checks", lambda: (lambda a: False, lambda ids: False))
    c = _Flaky()
    for _ in range(3):
        assert receiving.fetch_replies(SINCE, repo=_Repo(), client_for=lambda: c) == []
    assert c.header_calls == ["rep"] and c.fetched == []


# ---- final review F1: an unchecked message holds the intake cursor --------------------------

class _IntakeFlaky(ScopedClient):
    """09:00 unrelated colleague (definite no), 09:01 a supplier whose lookup may fail,
    09:02 a reply to mail we sent (always checkable)."""
    def __init__(self):
        super().__init__()

    def list_inbox(self, mailbox, since):
        items = [{"id": "col", "receivedDateTime": "2026-10-15T09:00:00Z", "from": "colleague@cust.com"},
                 {"id": "sup", "receivedDateTime": "2026-10-15T09:01:00Z", "from": "sup@supplier.com"},
                 {"id": "rep", "receivedDateTime": "2026-10-15T09:02:00Z", "from": "someone@gmail.com"}]
        stamp = since.strftime("%Y-%m-%dT%H:%M:%SZ")
        return [i for i in items if i["receivedDateTime"] >= stamp]


def test_an_unchecked_message_is_not_lost_by_the_intake_cursor(monkeypatch):
    monkeypatch.setattr(receiving, "_read_scope", lambda: "replies_and_suppliers")
    state = {"db_down": True}

    def is_supplier(addr):
        if state["db_down"] and addr == "sup@supplier.com":
            raise RuntimeError("db hiccup")
        return addr == "sup@supplier.com"
    monkeypatch.setattr(receiving, "_default_checks", lambda: (
        is_supplier, lambda ids: any(i in ("<ours-1@cust.com>", "ours-1@cust.com") for i in ids)))
    c = _IntakeFlaky()
    binding = {"binding_id": 1, "mailbox_address": "procurement@cust.com", "last_sync_at": SINCE}
    pipeline, conn = Pipeline(), FakeConn()

    receiving.poll_one(conn, binding, receiving.GraphMailSource(lambda: c), pipeline)
    assert "sup" not in c.fetched, "an unchecked message is never downloaded"
    assert "rep" not in c.fetched and pipeline.calls == [], \
        "nothing after the unchecked message is taken, so the cursor cannot pass it"
    assert not [s for s, _ in conn.sql if "SET last_sync_at" in s]

    state["db_down"] = False                   # the lookup works again on the next pass
    receiving.poll_one(conn, binding, receiving.GraphMailSource(lambda: c), pipeline)
    assert [m.provider_message_id for m in pipeline.calls] == ["sup", "rep"]
    assert c.fetched == ["sup", "rep"]


def test_a_skipped_message_is_logged_without_sender_or_subject(monkeypatch, caplog):
    import logging
    _scope(monkeypatch, "replies_and_suppliers")
    with caplog.at_level(logging.INFO, logger=receiving.log.name):
        _ids(ScopedClient())
    lines = [r.getMessage() for r in caplog.records if "skipped out-of-scope message" in r.getMessage()]
    assert lines == ["m365 intake skipped out-of-scope message col (binding None)"]
    assert "colleague@cust.com" not in lines[0]
