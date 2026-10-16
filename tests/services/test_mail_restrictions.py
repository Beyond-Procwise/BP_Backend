"""The restrictions the Email page draws, worked out on the server from live state."""
import re
from datetime import datetime, timedelta, timezone

import jwt
import pytest

from services import mail_policy as mp
from services import mail_restrictions as mr
from services.mail_m365 import graph_client
from services.mail_providers import m365 as m365_mod
from services.mail_providers.m365 import M365Provider
from services.style.graph_source import MailboxUnreachable

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=timezone.utc)
IDS = ["no_mail_permissions", "mailbox_scope", "actions_limited", "access_level",
       "internal_approval", "external_suppliers_only", "read_scope", "on_disconnect"]


def _policy(**over):
    v = dict(version=3, internal_send="with_user_approval", company_domains=("cust.com",),
             external_recipients="known_suppliers_only", read_scope="replies_and_suppliers",
             on_disconnect="hold", access_level="read_send", notices_from_mailbox="off",
             poll_seconds=120, restriction_group="Procwise Mailboxes", history=())
    v.update(over)
    return mp.EmailPolicy(**v)


def _mailbox(address="procurement@cust.com", *, age=timedelta(minutes=10), state="OK",
             withdrawn=False, active=True):
    return {"binding_id": 1, "address": address, "active": active, "is_default": True,
            "health_state": state, "verified_at": (NOW - age) if age is not None else None,
            "last_test_send_at": NOW - timedelta(days=1), "withdrawn": withdrawn}


class FakeProvider:
    def __init__(self, *, proof=None, mailboxes=None, prove_raises=None):
        self.proof = proof if proof is not None else [
            {"restriction": "no_mail_permissions", "check": "Mail permissions", "answer": "none",
             "result": "pass", "at": NOW.isoformat()}]
        self.mailboxes = [_mailbox()] if mailboxes is None else mailboxes
        self.prove_raises = prove_raises

    def prove(self, conn):
        if self.prove_raises:
            raise self.prove_raises
        return self.proof

    def status(self, conn):
        return {"connected": True, "mailboxes": self.mailboxes}


def _by_id(items):
    return {i["id"]: i for i in items}


def _build(provider=None, policy="default"):
    return mr.build(None, provider=FakeProvider() if provider is None else provider,
                    policy=_policy() if policy == "default" else policy, now=NOW)


# ---- shape and order ----

def test_eight_restrictions_in_the_spec_order():
    items = _build()
    assert [i["id"] for i in items] == IDS
    for i in items:
        assert set(i) >= {"id", "title", "explanation", "guarantee", "status",
                          "last_checked_at", "evidence"}
        for e in i["evidence"]:
            assert set(e) >= {"check", "answer", "result", "at"}
    g = {i["id"]: i["guarantee"] for i in items}
    assert g == {"no_mail_permissions": "provider", "mailbox_scope": "provider",
                 "actions_limited": "procwise", "access_level": "policy",
                 "internal_approval": "policy", "external_suppliers_only": "policy",
                 "read_scope": "policy", "on_disconnect": "policy"}


def test_titles_and_explanations_are_the_mock_up_words():
    r = _by_id(_build())
    assert r["no_mail_permissions"]["title"] == "Procwise has no mail permissions in your Microsoft account"
    assert r["no_mail_permissions"]["explanation"] == (
        "Microsoft shows “Procwise Mail” with sign-in only. Mail access comes solely from the "
        "restriction below.")
    assert r["mailbox_scope"]["title"] == "Only the mailboxes in “Procwise Mailboxes” can be reached"
    assert r["actions_limited"]["title"] == "Read and send only — never move, delete or mark as read"
    assert r["internal_approval"]["title"] == "Emails to colleagues need the user’s approval"
    assert r["internal_approval"]["explanation"] == (
        "Forwards, internal replies and Procwise’s own notices wait until the person who wrote "
        "them approves.")
    assert r["external_suppliers_only"]["title"] == "Emails outside the company go only to known suppliers"
    assert r["external_suppliers_only"]["explanation"] == \
        "Anything else is held with the reason and shown to the sender."
    assert r["read_scope"]["title"] == "Only replies and supplier mail are read"
    assert r["read_scope"]["explanation"] == (
        "Replies to emails Procwise sent, and mail from known suppliers. Other mail in the "
        "shared inbox is not opened.")
    assert r["on_disconnect"]["title"] == "If the connection breaks, emails are held"
    assert r["on_disconnect"]["explanation"] == \
        "They never go out from the Procwise address instead. You are alerted."


def test_the_group_name_comes_from_the_policy():
    r = _by_id(_build(policy=_policy(restriction_group="Buyers Mail")))
    assert r["mailbox_scope"]["title"] == "Only the mailboxes in “Buyers Mail” can be reached"


# ---- no_mail_permissions, through the real Microsoft provider ----

def _token(roles):
    claims = {"aud": "https://graph.microsoft.com", "tid": "t"}
    if roles is not None:
        claims["roles"] = roles
    return jwt.encode(claims, "irrelevant-secret-of-thirty-two-bytes!", algorithm="HS256")


class FakeClient:
    def __init__(self, token=None, raises=None):
        self._token, self._raises = token, raises

    def token(self):
        if self._raises:
            raise self._raises
        return self._token


@pytest.fixture(autouse=True)
def _fresh_cache():
    m365_mod._reset_proof_cache()
    yield
    m365_mod._reset_proof_cache()


@pytest.fixture
def connected(monkeypatch):
    monkeypatch.setattr(m365_mod, "read_integration",
                        lambda conn: {"status": "connected", "tenant_id": "tenant-1",
                                      "provider": "m365"})
    monkeypatch.setattr(m365_mod, "load_settings", lambda: object())


def _provider_with(client):
    return FakeProvider(proof=M365Provider(client_factory=lambda tenant: client).prove(object()))


def test_mail_read_granted_is_failed(connected):
    r = _by_id(_build(_provider_with(FakeClient(_token(["Mail.Read", "User.Read.All"])))))
    item = r["no_mail_permissions"]
    assert item["status"] == "failed"
    assert "Mail.Read" in item["evidence"][0]["answer"]
    assert "User.Read.All" not in item["evidence"][0]["answer"]


def test_no_mail_roles_is_in_force(connected):
    for roles in ([], None, ["User.Read.All"]):
        r = _by_id(_build(_provider_with(FakeClient(_token(roles)))))
        assert r["no_mail_permissions"]["status"] == "in_force", roles


def test_token_failure_is_not_proven(connected):
    r = _by_id(_build(_provider_with(FakeClient(raises=MailboxUnreachable("down")))))
    assert r["no_mail_permissions"]["status"] == "not_proven"


def test_unreadable_token_is_not_proven(connected):
    r = _by_id(_build(_provider_with(FakeClient("not-a-jwt"))))
    assert r["no_mail_permissions"]["status"] == "not_proven"


def test_not_connected_is_not_proven(monkeypatch):
    monkeypatch.setattr(m365_mod, "read_integration", lambda conn: {"status": "disconnected"})
    monkeypatch.setattr(m365_mod, "load_settings", lambda: object())
    rows = M365Provider(client_factory=lambda t: pytest.fail("no client when not connected")).prove(object())
    r = _by_id(_build(FakeProvider(proof=rows)))
    assert r["no_mail_permissions"]["status"] == "not_proven"


def test_no_provider_at_all_is_not_proven():
    r = _by_id(mr.build(None, provider=None, policy=_policy(), now=NOW))
    assert r["no_mail_permissions"]["status"] == "not_proven"
    assert r["mailbox_scope"]["status"] == "not_proven"


def test_prove_raising_is_not_proven():
    r = _by_id(_build(FakeProvider(prove_raises=RuntimeError("boom"))))
    assert r["no_mail_permissions"]["status"] == "not_proven"


# ---- mailbox_scope ----

def _scope(mailboxes):
    return _by_id(_build(FakeProvider(mailboxes=mailboxes)))["mailbox_scope"]


def test_scope_fresh_and_healthy_is_in_force_with_one_row_per_mailbox():
    s = _scope([_mailbox("a@cust.com"), _mailbox("b@cust.com", age=timedelta(minutes=90))])
    assert s["status"] == "in_force"
    assert len(s["evidence"]) == 2
    assert "a@cust.com" in s["evidence"][0]["check"] and "b@cust.com" in s["evidence"][1]["check"]
    assert s["evidence"][1]["at"] == (NOW - timedelta(minutes=90)).isoformat()


def test_scope_withdrawn_or_revoked_is_failed():
    assert _scope([_mailbox(), _mailbox("b@cust.com", state="REVOKED", withdrawn=True,
                                        active=False)])["status"] == "failed"
    assert _scope([_mailbox(state="REVOKED")])["status"] == "failed"


def test_scope_stale_or_degraded_is_not_proven():
    assert _scope([_mailbox(age=timedelta(hours=3))])["status"] == "not_proven"
    assert _scope([_mailbox(state="DEGRADED")])["status"] == "not_proven"
    assert _scope([_mailbox(age=None)])["status"] == "not_proven"


def test_scope_failed_beats_stale():
    assert _scope([_mailbox(age=timedelta(hours=3)),
                   _mailbox("b@cust.com", state="REVOKED", withdrawn=True)])["status"] == "failed"


def test_scope_with_no_mailboxes_is_not_proven():
    assert _scope([])["status"] == "not_proven"


def test_scope_status_raising_is_not_proven():
    class P(FakeProvider):
        def status(self, conn):
            raise RuntimeError("db down")
    assert _by_id(_build(P()))["mailbox_scope"]["status"] == "not_proven"


def test_scope_accepts_naive_and_text_times():
    naive = (NOW - timedelta(minutes=5)).replace(tzinfo=None)
    m1, m2 = _mailbox(), _mailbox("b@cust.com")
    m1["verified_at"] = naive
    m2["verified_at"] = (NOW - timedelta(minutes=5)).isoformat()
    assert _scope([m1, m2])["status"] == "in_force"


# ---- actions_limited ----

def test_actions_limited_lists_the_allowed_actions_in_plain_words():
    a = _by_id(_build())["actions_limited"]
    assert a["status"] == "in_force" and a["guarantee"] == "procwise"
    assert a["evidence"][0]["answer"] == "list inbox, open message, send"


def test_a_new_graph_path_breaks_actions_limited(monkeypatch):
    patterns = {k: list(v) for k, v in graph_client.ALLOWED_PATH_PATTERNS.items()}
    patterns["DELETE"] = [re.compile(r"^/users/[^/]+/messages/[^/]+$")]
    monkeypatch.setattr(graph_client, "ALLOWED_PATH_PATTERNS", patterns)
    assert _by_id(_build())["actions_limited"]["status"] == "failed"


# ---- policy items ----

POLICY_IDS = ["access_level", "internal_approval", "external_suppliers_only", "read_scope",
              "on_disconnect"]


def test_policy_items_on_are_in_force():
    r = _by_id(_build())
    for i in POLICY_IDS:
        assert r[i]["status"] == "in_force", i
        assert r[i]["evidence"] and "version 3" in r[i]["evidence"][0]["check"]


@pytest.mark.parametrize("field,value,item,words", [
    ("external_recipients", "any", "external_suppliers_only", "Your policy allows any outside address"),
    ("read_scope", "whole_inbox", "read_scope", "Your policy lets Procwise read the whole shared inbox"),
    ("on_disconnect", "procwise_address", "on_disconnect",
     "Your policy sends from the Procwise address if the connection breaks"),
    ("access_level", "read_send_change", "access_level",
     "Your policy lets your IT grant read, send and change"),
    ("internal_send", "off", "internal_approval",
     "Your policy switches emails to colleagues off, so none are sent"),
])
def test_the_wider_choice_is_shown_not_failed(field, value, item, words):
    r = _by_id(_build(policy=_policy(**{field: value})))
    assert r[item]["status"] == "in_force"
    assert words in r[item]["explanation"]


def test_unreadable_policy_makes_every_policy_item_not_proven():
    r = _by_id(_build(policy=None))
    for i in POLICY_IDS:
        assert r[i]["status"] == "not_proven", i
        assert r[i]["explanation"] == "Email policy could not be read"
    assert r["actions_limited"]["status"] == "in_force"
    assert r["mailbox_scope"]["title"] == "Only the mailboxes in your restriction group can be reached"


# ---- evidence report ----

HOSTILE = "<script>alert(1)</script>@evil.com"


def _html(**over):
    mbx = [_mailbox(HOSTILE)]
    items = _build(FakeProvider(mailboxes=mbx))
    hist = ({"version": 3, "changed_by": "<img src=x onerror=alert(1)>", "changed_at": NOW.isoformat(),
             "field": "read_scope", "before": "whole_inbox", "after": "<b>x</b>"},)
    kw = dict(restrictions=items, policy=_policy(history=hist), mailboxes=mbx, history=list(hist),
              generated_at=NOW, generated_by="<i>admin</i>@cust.com")
    kw.update(over)
    return mr.evidence_html(kw.pop("restrictions"), kw.pop("policy"), kw.pop("mailboxes"),
                            kw.pop("history"), **kw)


def test_evidence_html_escapes_everything():
    html = _html()
    assert "<script>" not in html and "<img" not in html and "<b>x</b>" not in html \
        and "<i>admin" not in html
    assert "&lt;script&gt;" in html


def test_evidence_html_has_no_outside_resources():
    html = _html()
    assert "http://" not in html and "https://" not in html and "src=" not in html.replace("src=x", "")


def test_evidence_html_is_dated_and_names_who_made_it():
    html = _html()
    assert "16 Oct 2026" in html
    for i in _build():
        assert i["title"].replace("“", "").split()[0] in html


def test_evidence_html_without_policy_says_so():
    assert "Email policy could not be read" in _html(policy=None, history=[])


def test_evidence_pdf_bytes():
    from services.report_export import render_pdf
    pdf = render_pdf(_html())
    assert pdf[:4] == b"%PDF"


# ---- the Microsoft check is cached briefly, failures only very briefly ----

class CountingClient:
    def __init__(self, token=None, raises=None):
        self.calls, self._token, self._raises = 0, token, raises

    def token(self):
        self.calls += 1
        if self._raises:
            raise self._raises
        return self._token


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def test_two_reads_within_the_window_ask_microsoft_once(connected):
    client, clock = CountingClient(_token([])), Clock()
    first = M365Provider(client_factory=lambda t: client, clock=clock).prove(object())
    clock.t += m365_mod.PROOF_TTL_SECONDS - 1
    # a fresh provider, as each request builds one: the cache must outlive it
    second = M365Provider(client_factory=lambda t: client, clock=clock).prove(object())
    assert client.calls == 1
    assert second == first          # the true time of the actual check, not "now"


def test_after_the_window_microsoft_is_asked_again(connected):
    client, clock = CountingClient(_token([])), Clock()
    M365Provider(client_factory=lambda t: client, clock=clock).prove(object())
    clock.t += m365_mod.PROOF_TTL_SECONDS + 1
    M365Provider(client_factory=lambda t: client, clock=clock).prove(object())
    assert client.calls == 2


def test_a_failed_check_is_not_cached_long(connected):
    client, clock = CountingClient(raises=MailboxUnreachable("slow")), Clock()
    rows = M365Provider(client_factory=lambda t: client, clock=clock).prove(object())
    assert rows[0]["result"] == "unknown"
    clock.t += m365_mod.UNKNOWN_TTL_SECONDS + 1
    M365Provider(client_factory=lambda t: client, clock=clock).prove(object())
    assert client.calls == 2
    assert m365_mod.UNKNOWN_TTL_SECONDS <= 10


def test_the_cache_is_per_tenant(monkeypatch):
    tenant = {"id": "tenant-1"}
    monkeypatch.setattr(m365_mod, "read_integration",
                        lambda conn: {"status": "connected", "tenant_id": tenant["id"]})
    monkeypatch.setattr(m365_mod, "load_settings", lambda: object())
    client, clock = CountingClient(_token([])), Clock()
    M365Provider(client_factory=lambda t: client, clock=clock).prove(object())
    tenant["id"] = "tenant-2"
    M365Provider(client_factory=lambda t: client, clock=clock).prove(object())
    assert client.calls == 2


def test_prove_without_a_connection_reads_the_integration_on_its_own(monkeypatch):
    """The router calls prove(None) after releasing its connection; the provider opens a
    short one only to read the integration, and has closed it before Microsoft is asked."""
    events = []

    class Own:
        def __enter__(self):
            events.append("open"); return self
        def __exit__(self, *a):
            events.append("close"); return False

    import services.db as db
    monkeypatch.setattr(db, "get_conn", lambda: Own())
    monkeypatch.setattr(m365_mod, "read_integration",
                        lambda conn: {"status": "connected", "tenant_id": "t"})
    monkeypatch.setattr(m365_mod, "load_settings", lambda: object())

    class C:
        def token(self):
            events.append("microsoft"); return _token([])
    M365Provider(client_factory=lambda t: C()).prove(None)
    assert events == ["open", "close", "microsoft"]


# ---- the words match the status ----

def test_permissions_explanation_follows_the_status(connected):
    ok = _by_id(_build(_provider_with(FakeClient(_token([])))))["no_mail_permissions"]
    assert "sign-in only" in ok["explanation"]
    m365_mod._reset_proof_cache()   # each case is a fresh answer from Microsoft
    bad = _by_id(_build(_provider_with(FakeClient(_token(["Mail.Read"])))))["no_mail_permissions"]
    assert bad["status"] == "failed"
    assert "sign-in only" not in bad["explanation"]
    assert "Microsoft reports" in bad["explanation"] and "mail permissions" in bad["explanation"]
    m365_mod._reset_proof_cache()
    unk = _by_id(_build(_provider_with(FakeClient(raises=MailboxUnreachable("x")))))["no_mail_permissions"]
    assert unk["status"] == "not_proven"
    assert "sign-in only" not in unk["explanation"]
    assert "could not confirm" in unk["explanation"]


def test_scope_explanation_follows_the_status():
    ok = _scope([_mailbox()])
    assert "succeeded" in ok["explanation"]
    bad = _scope([_mailbox(state="REVOKED", withdrawn=True)])
    assert bad["status"] == "failed"
    assert "succeeded" not in bad["explanation"] and "refused by Microsoft" not in bad["explanation"]
    assert "Microsoft reports" in bad["explanation"]
    stale = _scope([_mailbox(age=timedelta(hours=3))])
    assert stale["status"] == "not_proven"
    assert "succeeded" not in stale["explanation"] and "could not confirm" in stale["explanation"]
    none = _scope([])
    assert "succeeded" not in none["explanation"] and "could not confirm" in none["explanation"]


def test_actions_explanation_follows_the_status(monkeypatch):
    patterns = {k: list(v) for k, v in graph_client.ALLOWED_PATH_PATTERNS.items()}
    patterns["DELETE"] = [re.compile(r"^/users/[^/]+/messages/[^/]+$")]
    monkeypatch.setattr(graph_client, "ALLOWED_PATH_PATTERNS", patterns)
    a = _by_id(_build())["actions_limited"]
    assert a["status"] == "failed" and "refuses every other action" not in a["explanation"]


# ---- one read of the mailboxes feeds both the line and the table ----

def test_build_uses_the_mailboxes_it_is_given():
    class P(FakeProvider):
        def status(self, conn):
            pytest.fail("mailboxes were given; they must not be read again")
    items = mr.build(None, provider=P(), policy=_policy(), now=NOW,
                     mailboxes=[_mailbox("given@cust.com")], proof=FakeProvider().proof)
    s = _by_id(items)["mailbox_scope"]
    assert "given@cust.com" in s["evidence"][0]["check"]


def test_build_with_unreadable_mailboxes_is_not_proven():
    items = mr.build(None, provider=FakeProvider(), policy=_policy(), now=NOW, mailboxes=None,
                     proof=FakeProvider().proof)
    assert _by_id(items)["mailbox_scope"]["status"] == "not_proven"
