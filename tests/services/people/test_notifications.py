from datetime import datetime, timezone
from decimal import Decimal

import pytest

from services.people.events import CATALOGUE, DomainEvent, UnknownEventType
from services.people.hierarchy import HierarchyService
from services.people.models import EntityRef
from services.people.notifications import (MAX_ATTEMPTS, InMemoryNotificationRepo, NotificationService, drain)
from services.people.resolver import RecipientResolver, StaticRules
from tests.services.people.test_people import NOW, org


def build(prefs=None, channels=None, now=NOW, store=None):
    s = store or org()
    res = RecipientResolver(s, HierarchyService(s, now=lambda: NOW), StaticRules(), lambda p, e: True)
    repo = InMemoryNotificationRepo()
    svc = NotificationService(
        res, repo,
        entity_for=lambda e: EntityRef(e.entity_type, e.entity_id, amount=Decimal(e.data.get("amount", 0))),
        person_for=lambda t, sub: s.get(t, sub),
        prefs_for=lambda t, sub: (prefs or {}).get(sub, {}),
        channels=channels, now=lambda: now)
    return svc, repo


def ev(type="requisition.submitted", amount=60000, **kw):
    return DomainEvent(type=type, actor_id="buyer", entity_type="requisition", entity_id="R1",
                       summary="Jane submitted R1", data={"amount": amount}, **kw)


def test_unknown_event_type_rejected():
    with pytest.raises(UnknownEventType):
        DomainEvent(type="nope", actor_id="a", entity_type="x", entity_id="1", summary="s")


def test_acceptance_1_manager_and_approver_notified_actor_not():
    svc, repo = build()
    assert svc.handle(ev()) == 2
    by = {r["user"]: r for r in repo.rows}
    assert set(by) == {"cm", "cpo"}                       # manager, and unlimited approver for 60k
    assert "manager" in by["cm"]["detail"] and by["cpo"]["reason"] == "approver"
    assert all(r["link"].startswith("/") and "api" not in r["link"] for r in repo.rows)


def test_idempotent_replay():
    svc, repo = build()
    e = ev()
    assert svc.handle(e) == 2
    assert svc.handle(e) == 0
    assert len(repo.rows) == 2


def test_muted_event_skipped_but_mandatory_not():
    prefs = {"cm": {"muted_events": ["requisition.submitted", "approval.requested"]}}
    svc, repo = build(prefs)
    svc.handle(ev())
    assert {r["user"] for r in repo.rows} == {"cpo"}
    svc.handle(ev("approval.requested", amount=100))       # cm is the approver for 100; mandatory
    assert "cm" in {r["user"] for r in repo.rows}


def test_channel_dispatch_and_quiet_hours():
    sent = []
    ch = {"email": lambda person, msg: sent.append((person.subject, msg["title"]))}
    prefs = {"cm": {"channels": ["email"], "quiet_hours": ["22:00", "07:00"]}}
    svc, repo = build(prefs, ch, now=datetime(2026, 10, 21, 12, tzinfo=timezone.utc))
    svc.handle(ev())
    assert sent == [("cm", "Jane submitted R1")]
    svc, repo = build(prefs, ch, now=datetime(2026, 10, 21, 23, tzinfo=timezone.utc))
    sent.clear(); svc.handle(ev())
    assert sent == [] and [r["channels"]["email"] for r in repo.rows if r["user"] == "cm"] == ["deferred_quiet_hours"]


def test_quiet_hours_do_not_hold_mandatory():
    sent = []
    prefs = {"cm": {"channels": ["email"], "quiet_hours": ["00:00", "23:59"]}}
    svc, _ = build(prefs, {"email": lambda p, m: sent.append(p.subject)})
    svc.handle(ev("approval.requested", amount=100))
    assert sent == ["cm"]


def test_channel_failure_raises_but_keeps_rows_then_retry_does_not_duplicate_rows():
    calls = {"n": 0}
    def flaky(person, msg):
        calls["n"] += 1
        raise ConnectionError("smtp down")
    svc, repo = build({"cm": {"channels": ["email"]}}, {"email": flaky})
    e = ev()
    with pytest.raises(RuntimeError):
        svc.handle(e)
    assert len(repo.rows) == 2
    assert [r["channels"]["email"] for r in repo.rows if r["user"] == "cm"] == ["failed"]
    with pytest.raises(RuntimeError):
        svc.handle(e)                                      # retry really re-attempts the channel
    assert calls["n"] == 2 and len(repo.rows) == 2


def test_retry_after_channel_recovers_sends_once_and_replay_does_not_resend():
    state = {"up": False, "sent": []}
    def email(person, msg):
        if not state["up"]:
            raise ConnectionError("down")
        state["sent"].append(person.subject)
    svc, repo = build({"cm": {"channels": ["email"]}}, {"email": email})
    e = ev()
    with pytest.raises(RuntimeError):
        svc.handle(e)
    state["up"] = True
    assert svc.handle(e) == 0 and state["sent"] == ["cm"]
    svc.handle(e)
    assert state["sent"] == ["cm"]


class FakeOutbox:
    def __init__(self, events):
        self.events, self.state = events, {}
    def claim(self, limit): return [e for e in self.events if e.event_id not in self.state][:limit]
    def done(self, i): self.state[i] = "done"
    def failed(self, i, err, give_up):
        for e in self.events:
            if e.event_id == i: e.attempts += 1
        if give_up: self.state[i] = "gave_up"


def test_drain_retries_then_gives_up():
    def boom(person, msg): raise ConnectionError("down")
    svc, _ = build({"cm": {"channels": ["email"]}}, {"email": boom})
    ob = FakeOutbox([ev()])
    results = [drain(ob, svc) for _ in range(MAX_ATTEMPTS)]
    assert [r["retry"] for r in results] == [1] * (MAX_ATTEMPTS - 1) + [0]
    assert results[-1]["gave_up"] == 1 and ob.state[ob.events[0].event_id] == "gave_up"


def test_drain_success():
    svc, repo = build()
    ob = FakeOutbox([ev(), ev(amount=100)])
    assert drain(ob, svc) == {"done": 2, "retry": 0, "gave_up": 0}


def test_catalogue_covers_prompt_minimum():
    for t in ["requisition.created", "approval.requested", "po.issued", "invoice.mismatch", "supplier.risk_flagged",
              "contract.expiring", "rfq.response_received", "agent.recommendation_ready", "comment.mentioned",
              "message.received"]:
        assert t in CATALOGUE


# --- email channel ---------------------------------------------------------
from services.people.email_channel import compose, make_email_channel


def _email_svc(sent, addresses, accept=True, prefs=None):
    chan = make_email_channel(lambda p: addresses.get(p.subject),
                              lambda to, subj, body: (sent.append((to, subj, body)) or accept), "https://app.example")
    p = prefs if prefs is not None else {"cm": {"channels": ["email"]}, "cpo": {"channels": ["email"]}}
    return build(p, channels={"email": chan})


def test_email_sent_with_value_free_text_and_link():
    sent = []
    svc, repo = _email_svc(sent, {"cm": "cm@x.com", "cpo": "cpo@x.com"})
    svc.handle(ev(amount=60000))
    assert {s[0] for s in sent} == {"cm@x.com", "cpo@x.com"}
    assert all("60000" not in s[2] and "https://app.example/requisition/R1" in s[2] for s in sent)
    assert all(r["channels"]["email"] == "sent" for r in repo.rows)


def test_email_not_sent_twice_on_replay():
    sent = []
    svc, _ = _email_svc(sent, {"cm": "cm@x.com", "cpo": "cpo@x.com"})
    e = ev()
    svc.handle(e); svc.handle(e)
    assert len(sent) == 2


def test_missing_address_is_skipped_not_retried():
    sent = []
    svc, repo = _email_svc(sent, {"cm": "cm@x.com"})          # cpo has no address
    svc.handle(ev())                                           # must not raise
    st = {r["user"]: r["channels"]["email"] for r in repo.rows}
    assert st == {"cm": "sent", "cpo": "skipped_no_address"} and len(sent) == 1


def test_mail_service_refusal_marks_failed_and_raises_for_retry():
    svc, repo = _email_svc([], {"cm": "cm@x.com", "cpo": "cpo@x.com"}, accept=False)
    with pytest.raises(RuntimeError):
        svc.handle(ev())
    assert all(r["channels"]["email"] == "failed" for r in repo.rows)


def test_no_email_prefs_no_email():
    sent = []
    svc, _ = _email_svc(sent, {"cm": "cm@x.com"}, prefs={})
    svc.handle(ev())
    assert sent == []


def test_compose_omits_link_without_base_url():
    assert "Open it" not in compose({"title": "t", "link": "/x/1"})[1]


def test_body_comes_from_the_event_when_it_says_what_it_is_about():
    """A message event carries a preview and context in data['body']; the bell shows that, not
    the title twice (live, 2026-10-22: body == title for every message notification)."""
    svc, repo = build()
    e = DomainEvent(type="requisition.submitted", actor_id="buyer", entity_type="requisition", entity_id="R1",
                    summary="Jane submitted R1", data={"amount": 60000, "body": "For the Q4 laptops\nAbout requisition R1"})
    svc.handle(e)
    assert {r["body"] for r in repo.rows} == {"For the Q4 laptops\nAbout requisition R1"}
    svc, repo = build()
    svc.handle(ev())
    assert {r["body"] for r in repo.rows} == {"Jane submitted R1"}
