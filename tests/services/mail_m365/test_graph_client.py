import base64
from datetime import datetime, timezone

import pytest

from services.mail_m365.graph_client import M365Client, Throttled
from tests.services.mail_m365.certs import fake_credentials
from services.style.graph_source import (MailboxAccessDenied,
                                         MailboxUnreachable, RequestsTransport)


class Stub:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []
        self.token_posts = 0

    def post_form(self, url, data):
        self.token_posts += 1
        return 200, {"access_token": "tok", "expires_in": 3600}

    def _match(self, method, url):
        self.calls.append((method, url))
        for (m, frag), resp in self.routes.items():
            if m == method and frag in url:
                return resp
        return 404, {}

    def get(self, url, headers):
        return self._match("GET", url)

    def get_bytes(self, url, headers):
        return self._match("GETB", url)

    def post_body(self, url, headers, body):
        self.last_body = body
        self.last_headers = headers
        return self._match("POST", url)


def client(routes, clock=lambda: 1000.0):
    return M365Client(fake_credentials(), transport=Stub(routes), clock=clock)


def test_token_is_cached_until_near_expiry():
    now = [1000.0]
    c = client({}, clock=lambda: now[0])
    c.token(); c.token()
    assert c.transport.token_posts == 1
    now[0] += 3600 - 30
    c.token()
    assert c.transport.token_posts == 2


def test_can_read_allowed_and_denied():
    c = client({("GET", "/users/ok@x/mailFolders/inbox"): (200, {"id": "inbox"}),
                ("GET", "/users/no@x/mailFolders/inbox"): (403, {})})
    assert c.can_read("ok@x")[0] is True
    assert c.can_read("no@x")[0] is False


def test_can_read_401_is_our_token_not_a_denial():
    c = client({("GET", "/mailFolders/inbox"): (401, {})})
    with pytest.raises(MailboxUnreachable, match="sign-in"):
        c.can_read("m@x")


def test_can_read_inconclusive_raises():
    c = client({("GET", "/mailFolders/inbox"): (500, {})})
    with pytest.raises(MailboxUnreachable):
        c.can_read("m@x")


def test_probe_control_success_is_the_failure_case():
    c = client({("GET", "/users/ctl@x/messages"): (200, {"value": []})})
    assert c.probe_control("ctl@x")[0] is True
    c = client({("GET", "/users/ctl@x/messages"): (403, {})})
    assert c.probe_control("ctl@x")[0] is False
    c = client({("GET", "/users/ctl@x/messages"): (503, {})})
    with pytest.raises(MailboxUnreachable):
        c.probe_control("ctl@x")


@pytest.mark.parametrize("status", [401, 404])
def test_probe_control_only_403_counts_as_denied(status):
    c = client({("GET", "/users/ctl@x/messages"): (status, {})})
    with pytest.raises(MailboxUnreachable):
        c.probe_control("ctl@x")


class PagedInbox(Stub):
    """An Inbox of `n` messages served $top/$skip at a time. Every page also carries a
    hostile nextLink, which must never be requested."""

    def __init__(self, n):
        super().__init__({})
        self.msgs = [{"id": f"m{i:03d}", "receivedDateTime": f"2026-10-15T{i // 60:02d}:{i % 60:02d}:00Z"}
                     for i in range(n)]

    def get(self, url, headers):
        self.calls.append(("GET", url))
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(url).query)
        top, skip = int(q["$top"][0]), int(q["$skip"][0])
        page = self.msgs[skip:skip + top]
        body = {"value": page}
        if skip + top < len(self.msgs):
            body["@odata.nextLink"] = "https://evil.example/users/other@x/messages?steal=1"
        return 200, body


def test_list_inbox_pages_with_its_own_urls_and_never_follows_next_link():
    """I4: 120 messages over three pages of 50; every request is one we built."""
    c = M365Client(fake_credentials(), transport=PagedInbox(120))
    got = c.list_inbox("m@x", datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert [m["id"] for m in got] == [f"m{i:03d}" for i in range(120)]
    urls = [u for _, u in c.transport.calls]
    assert len(urls) == 3
    assert all(u.startswith("https://graph.microsoft.com/v1.0/users/m@x/mailFolders/inbox/messages?")
               for u in urls)
    assert ["$skip=0" in urls[0], "$skip=50" in urls[1], "$skip=100" in urls[2]] == [True] * 3
    assert all("$filter=receivedDateTime ge 2026-10-15T00:00:00Z" in u
               and "$orderby=receivedDateTime asc" in u for u in urls)
    assert not any("evil" in u or "steal" in u for u in urls)


def test_list_inbox_exact_multiple_of_page_size_stops_on_the_empty_page():
    c = M365Client(fake_credentials(), transport=PagedInbox(100))
    assert len(c.list_inbox("m@x", datetime(2026, 10, 15, tzinfo=timezone.utc))) == 100
    assert len(c.transport.calls) == 3


def test_list_inbox_sorts_oldest_first():
    c = client({("GET", "/users/m@x/mailFolders/inbox/messages"): (200, {"value": [
        {"id": "b", "receivedDateTime": "2026-10-15T10:00:00Z"},
        {"id": "a", "receivedDateTime": "2026-10-15T09:00:00Z"}]})})
    got = c.list_inbox("m@x", datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert [m["id"] for m in got] == ["a", "b"]


def test_list_inbox_filter_is_utc_for_aware_and_naive_since():
    from datetime import timedelta
    c = client({("GET", "/messages"): (200, {"value": []})})
    c.list_inbox("m@x", datetime(2026, 10, 15, 10, 0, tzinfo=timezone(timedelta(hours=1))))
    assert "receivedDateTime ge 2026-10-15T09:00:00Z" in c.transport.calls[-1][1]
    c.list_inbox("m@x", datetime(2026, 10, 15, 10, 0))
    assert "receivedDateTime ge 2026-10-15T10:00:00Z" in c.transport.calls[-1][1]


def test_list_inbox_429_raises_throttled():
    c = client({("GET", "/messages"): (429, {"retry_after": 7})})
    with pytest.raises(Throttled) as ei:
        c.list_inbox("m@x", datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert ei.value.retry_after == 7
    assert isinstance(ei.value, MailboxUnreachable)


def test_get_mime_returns_bytes_and_maps_errors():
    c = client({("GETB", "/users/m@x/messages/abc/$value"): (200, b"raw")})
    assert c.get_mime("m@x", "abc") == b"raw"
    c = client({("GETB", "/$value"): (403, b"")})
    with pytest.raises(MailboxAccessDenied):
        c.get_mime("m@x", "abc")
    c = client({("GETB", "/$value"): (429, b"")})
    with pytest.raises(Throttled):
        c.get_mime("m@x", "abc")


def test_send_mime_posts_base64_text_plain():
    c = client({("POST", "/users/m@x/sendMail"): (202, {})})
    c.send_mime("m@x", b"Subject: hi\r\n\r\nbody")
    assert c.transport.last_headers["Content-Type"] == "text/plain"
    assert base64.b64decode(c.transport.last_body) == b"Subject: hi\r\n\r\nbody"


def test_send_denied_raises_access_denied():
    c = client({("POST", "/sendMail"): (403, {})})
    with pytest.raises(MailboxAccessDenied):
        c.send_mime("m@x", b"x")


def test_send_throttled_and_other_failures():
    with pytest.raises(Throttled):
        client({("POST", "/sendMail"): (429, {})}).send_mime("m@x", b"x")
    with pytest.raises(MailboxUnreachable):
        client({("POST", "/sendMail"): (500, {})}).send_mime("m@x", b"x")


@pytest.mark.parametrize("path", ["/users/m@x/messages/1/move",
                                  "/me/messages", "/users/m@x/mailFolders/inbox/messages/1/forward",
                                  "/users/m@x/messages/1/attachments",
                                  "/users/m@x/messages/1/../move"])
def test_paths_outside_the_allowlist_are_refused(path):
    c = client({})
    for method in ("GET", "POST", "PATCH", "DELETE"):
        with pytest.raises(ValueError):
            c._url(path, method=method)


def test_send_path_is_post_only_and_reads_are_get_only():
    c = client({})
    with pytest.raises(ValueError):
        c._url("/users/m@x/sendMail", method="GET")
    with pytest.raises(ValueError):
        c._url("/users/m@x/mailFolders/inbox/messages", method="POST")


def test_no_delete_or_patch_method_exists():
    assert not any(hasattr(M365Client, n) for n in ("delete", "patch", "move", "mark_read"))


# -- the two transport additions ------------------------------------------------

class _Resp:
    def __init__(self, status, content=b"", js=None):
        self.status_code, self.content, self._js = status, content, js

    def json(self):
        if self._js is None:
            raise ValueError
        return self._js


def test_transport_get_bytes_and_post_body_use_mailbox_egress(monkeypatch):
    from src.services import egress
    seen = {}

    def fake_get(url, *, purpose, **kw):
        seen["get"] = (url, purpose, kw)
        return _Resp(200, b"raw")

    def fake_post(url, *, purpose, **kw):
        seen["post"] = (url, purpose, kw)
        return _Resp(202)

    monkeypatch.setattr(egress, "get", fake_get)
    monkeypatch.setattr(egress, "post", fake_post)
    t = RequestsTransport()
    assert t.get_bytes("https://graph.microsoft.com/x", {"A": "b"}) == (200, b"raw")
    assert seen["get"][1] == egress.Purpose.MAILBOX
    assert t.post_body("https://graph.microsoft.com/y", {"A": "b"}, b"zz") == (202, {})
    assert seen["post"][1] == egress.Purpose.MAILBOX
    assert seen["post"][2]["data"] == b"zz"


def test_transport_wraps_refused_and_failed_calls(monkeypatch):
    from src.services import egress
    monkeypatch.setattr(egress, "get", lambda *a, **k: None)   # egress refusal returns None
    monkeypatch.setattr(egress, "post", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    t = RequestsTransport()
    with pytest.raises(MailboxUnreachable):
        t.get_bytes("https://graph.microsoft.com/x", {})
    with pytest.raises(MailboxUnreachable):
        t.post_body("https://graph.microsoft.com/y", {}, b"")


@pytest.mark.parametrize("host", ["graph.microsoft.com", "login.microsoftonline.com"])
def test_egress_permits_the_microsoft_hosts(monkeypatch, host):
    from src.services import egress
    monkeypatch.setattr(egress, "_resolve", lambda h: ["20.190.151.1"])
    ok, why = egress.check_destination(f"https://{host}/x")
    assert ok, why
    assert egress._evaluate(egress.Purpose.MAILBOX, host).permitted is True


def test_list_inbox_selects_the_sender_and_returns_a_bare_address():
    c = client({("GET", "/mailFolders/inbox/messages"): (200, {"value": [
        {"id": "a", "receivedDateTime": "2026-10-15T09:00:00Z",
         "from": {"emailAddress": {"address": " Sup@Supplier.COM ", "name": "Sup"}}},
        {"id": "b", "receivedDateTime": "2026-10-15T09:01:00Z"}]})})
    got = c.list_inbox("m@x", datetime(2026, 10, 15, tzinfo=timezone.utc))
    assert "$select=id,receivedDateTime,from" in c.transport.calls[0][1]
    assert got[0]["from"] == "sup@supplier.com" and got[1]["from"] == ""


def test_get_headers_lowercases_names_and_quotes_a_hostile_id():
    hdrs = {"internetMessageHeaders": [{"name": "In-Reply-To", "value": "<a@x>"},
                                       {"name": "References", "value": "<b@x> <a@x>"}]}
    c = client({("GET", "internetMessageHeaders"): (200, hdrs)})
    got = c.get_headers("m@x", "AAA/users/other@x/messages?x=1#")
    with pytest.raises(ValueError):          # a dot-dot id is refused outright by the allowlist
        c.get_headers("m@x", "AAA/../../users/other@x")
    assert got == {"in-reply-to": "<a@x>", "references": "<b@x> <a@x>"}
    url = c.transport.calls[0][1]
    assert url.startswith("https://graph.microsoft.com/v1.0/users/m@x/messages/AAA%2Fusers%2Fother%40x%2Fmessages%3Fx%3D1%23?")
    assert url.endswith("?$select=internetMessageHeaders") and "other@x/messages" not in url and "x=1" not in url.split("?")[0]


def test_get_headers_maps_errors_and_missing_headers():
    assert client({("GET", "internetMessageHeaders"): (200, {})}).get_headers("m@x", "1") == {}
    with pytest.raises(Throttled):
        client({("GET", "internetMessageHeaders"): (429, {})}).get_headers("m@x", "1")
    with pytest.raises(MailboxAccessDenied):
        client({("GET", "internetMessageHeaders"): (403, {})}).get_headers("m@x", "1")
    with pytest.raises(MailboxUnreachable):
        client({("GET", "internetMessageHeaders"): (500, {})}).get_headers("m@x", "1")
