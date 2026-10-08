"""The email-learning HTTP surface: who may do what, who is recorded, and what never leaves. Real Postgres behind it."""

import json
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api.routers.email_learning as mod
from src.services import actions
from tests.email_evals.test_learning import db, engine  # noqa: F401  (fixtures)
from tests.email_evals.test_queues import clean, cls, dq, evalc, exemplar, review, rule  # noqa: F401

NICK = "sub-nick"


class Gate:
    """A stand-in for guardrail.authorize that records every question and answers from a table."""

    def __init__(self):
        self.asked, self.deny = [], set()

    def __call__(self, action, action_class, principal, context=None, policy_engine=None):
        self.asked.append((action, action_class, dict(context or {})))
        return SimpleNamespace(allowed=action not in self.deny)


@pytest.fixture
def api(clean, engine, monkeypatch):
    gate = Gate()
    monkeypatch.setattr(mod.guardrail, "authorize", gate)
    monkeypatch.setattr(mod.rbac, "policy_engine", lambda: engine)

    @contextmanager
    def store(agent_nick):
        yield clean
    monkeypatch.setattr(mod, "_store", store)
    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[mod.require_user] = lambda: SimpleNamespace(subject=NICK)
    app.dependency_overrides[mod.get_agent_nick] = lambda: object()
    client = TestClient(app)
    client.gate, client.app_, client.db = gate, app, clean
    return client


def row(db, sql, *p):
    with db.cursor() as cur:
        cur.execute(sql, p)
        return cur.fetchone()


# --- the vocabulary -----------------------------------------------------------------------------------------------------------

def test_the_new_actions_are_in_the_closed_vocabulary_with_the_right_classes():
    assert actions.action_class("email.draft.read") == "read"
    assert actions.action_class("email.learning.read") == "read"
    assert actions.action_class("email.learning.decide") == "write"
    assert actions.action_class("exemplar.approve") == "configure"


def test_the_router_is_registered_with_the_authenticated_routers():
    text = (Path(mod.__file__).resolve().parents[1] / "main.py").read_text()
    assert "from api.routers import email_learning as email_learning_router" in text
    assert re.search(r"_AUTHENTICATED_ROUTERS\s*=\s*\[(?:.|\n)*?email_learning_router\.router", text)


# --- every route asks the right question ----------------------------------------------------------------------------------------

READS = [("/queues", "email.learning.read"), ("/data-quality", "email.learning.read"), ("/review-items", "email.learning.read"),
         ("/style-rules", "email.learning.read"), ("/eval-candidates", "email.learning.read"),
         ("/classifier-examples", "email.learning.read"), ("/exemplars", "email.learning.read"), ("/metrics", "email.learning.read")]


@pytest.mark.parametrize("path,action", READS)
def test_each_listing_asks_its_gate_exactly_once_and_denial_is_a_403(api, path, action):
    assert api.get("/email-learning" + path).status_code == 200
    assert [a[0] for a in api.gate.asked] == [action]
    api.gate.asked.clear(); api.gate.deny.add(action)
    r = api.get("/email-learning" + path)
    assert r.status_code == 403 and "permitted" in r.json()["detail"]


def test_the_exemplar_text_needs_the_configure_gate_that_listing_does_not(api):
    e = exemplar(api.db, text="Dear Alex, thank you.")
    api.gate.deny.add("exemplar.approve")
    assert api.get("/email-learning/exemplars").status_code == 200
    assert api.get(f"/email-learning/exemplars/{e}").status_code == 403
    api.gate.deny.clear()
    got = api.get(f"/email-learning/exemplars/{e}")
    assert got.status_code == 200 and got.json()["text"] == "Dear Alex, thank you."
    assert ("exemplar.approve", "configure") == api.gate.asked[-1][:2]
    assert api.get("/email-learning/exemplars/99999").status_code == 404


@pytest.mark.parametrize("path,make,gate", [
    ("/data-quality/{id}/decision", dq, ("email.learning.decide", "write")),
    ("/review-items/{id}/decision", review, ("email.learning.decide", "write")),
    ("/eval-candidates/{id}/decision", evalc, ("email.learning.decide", "write")),
    ("/classifier-examples/{id}/decision", cls, ("email.learning.decide", "write")),
    ("/exemplars/{id}/decision", exemplar, ("exemplar.approve", "configure")),
])
def test_each_decision_asks_its_gate_and_a_denial_changes_nothing(api, path, make, gate):
    item = make(api.db) if make is not exemplar else make(api.db, author="other")
    body = {"action": {"/data-quality": "resolve", "/review-items": "accept", "/eval-candidates": "export", "/classifier-examples": "export",
                       "/exemplars": "approve"}[next(k for k in ("/data-quality", "/review-items", "/eval-candidates", "/classifier-examples", "/exemplars") if path.startswith(k))]}
    api.gate.deny.add(gate[0])
    assert api.post("/email-learning" + path.format(id=item), json=body).status_code == 403
    assert api.gate.asked[-1][:2] == gate
    api.gate.deny.clear()
    r = api.post("/email-learning" + path.format(id=item), json=body)
    assert r.status_code == 200 and r.json() == {"ok": True}, r.text


# --- who is recorded ----------------------------------------------------------------------------------------------------------------

def test_the_person_recorded_is_the_authenticated_one_whatever_the_body_claims(api):
    item = dq(api.db)
    r = api.post(f"/email-learning/data-quality/{item}/decision", json={"action": "resolve", "note": "fixed", "by": "mallory", "resolved_by": "mallory", "subject": "mallory"})
    assert r.status_code == 200
    assert row(api.db, "SELECT resolved_by FROM email_agent.bp_dq_item WHERE dq_id = %s", item)[0] == NICK


def test_a_blank_person_is_refused_before_anything_is_asked_or_changed(api):
    api.app_.dependency_overrides[mod.require_user] = lambda: SimpleNamespace(subject="  ")
    item = dq(api.db)
    assert api.post(f"/email-learning/data-quality/{item}/decision", json={"action": "resolve"}).status_code == 401
    assert api.get("/email-learning/queues").status_code == 401
    assert api.gate.asked == [] and row(api.db, "SELECT status FROM email_agent.bp_dq_item WHERE dq_id = %s", item)[0] == "open"


def test_a_person_cannot_decide_somebody_elses_style_rule_or_see_it(api):
    theirs = rule(api.db, "someone-else", key="k9")
    assert api.post(f"/email-learning/style-rules/{theirs}/decision", json={"action": "approve"}).status_code == 404
    assert api.get("/email-learning/style-rules").json() == {"items": []}
    mine = rule(api.db, NICK)
    assert api.post(f"/email-learning/style-rules/{mine}/decision", json={"action": "edit", "text": "Be brief."}).status_code == 200
    assert [r["text"] for r in api.get("/email-learning/style-rules").json()["items"]] == ["Be brief."]


def test_an_exemplar_cannot_be_approved_by_its_author_through_the_api(api):
    own = exemplar(api.db, author=NICK)
    assert api.post(f"/email-learning/exemplars/{own}/decision", json={"action": "approve"}).status_code == 422
    assert row(api.db, "SELECT status FROM email_agent.bp_exemplar_candidate WHERE exemplar_id = %s", own)[0] == "candidate"
    other = exemplar(api.db, author="someone-else")
    assert api.post(f"/email-learning/exemplars/{other}/decision", json={"action": "approve"}).status_code == 200
    status, by, after = row(api.db, "SELECT status, approved_by, review_after FROM email_agent.bp_exemplar_candidate WHERE exemplar_id = %s", other)
    assert (status, by) == ("approved", NICK) and after is not None                                     # re-review is scheduled


def test_approving_an_exemplar_needs_the_review_period_to_be_configured(api, monkeypatch):
    monkeypatch.setattr(mod.rbac, "policy_engine", lambda: None)
    other = exemplar(api.db, author="someone-else")
    r = api.post(f"/email-learning/exemplars/{other}/decision", json={"action": "approve"})
    assert r.status_code == 503 and "not configured" in r.json()["detail"]


@pytest.mark.parametrize("path", ["/data-quality/{id}/decision", "/review-items/{id}/decision", "/exemplars/{id}/decision"])
def test_a_nonsense_action_is_refused_and_changes_nothing(api, path):
    make = {"/data-quality": dq, "/review-items": review, "/exemplars": exemplar}[path[:path.index("/{")]]
    item = make(api.db)
    assert api.post("/email-learning" + path.format(id=item), json={"action": "obliterate"}).status_code == 422
    assert api.post("/email-learning" + path.format(id=999999), json={"action": "resolve"}).status_code in (404, 422)


# --- what comes back ----------------------------------------------------------------------------------------------------------------------

def test_no_listing_returns_raw_text_or_an_internal_name(api):
    dq(api.db); review(api.db); rule(api.db, NICK); exemplar(api.db, text="SECRET EXEMPLAR TEXT"); evalc(api.db); cls(api.db)
    blob = ""
    for p in ("/queues", "/data-quality", "/review-items", "/style-rules", "/exemplars", "/eval-candidates", "/classifier-examples", "/metrics"):
        r = api.get("/email-learning" + p)
        assert r.status_code == 200, p
        blob += r.text
    for bad in ("SECRET EXEMPLAR TEXT", "RAW DRAFT TEXT", "supplier_response", "email_agent", "bp_dq_item", "bp_draft", "\"table\"", "\"column\""):
        assert bad not in blob, bad


def test_the_queue_counts_reflect_the_caller_only_for_style_rules(api):
    rule(api.db, NICK); rule(api.db, "someone-else", key="k2"); dq(api.db)
    waiting = api.get("/email-learning/queues").json()["waiting"]
    assert waiting["style_rules"] == 1 and waiting["data_quality"] == 1


def test_an_unknown_status_or_bucket_is_a_400_not_an_empty_list(api):
    assert api.get("/email-learning/data-quality?status=approved").status_code == 400
    assert api.get("/email-learning/metrics?bucket=fortnight").status_code == 400
    assert api.get("/email-learning/data-quality?status=resolved").json() == {"items": []}


def test_metrics_report_by_family_over_time(api):
    from tests.email_evals.test_learning import sent
    sent(api.db, family="negotiation_counter")
    body = api.get("/email-learning/metrics?bucket=month&days=30").json()
    assert body["bucket"] == "month" and body["days"] == 30
    (r,) = [x for x in body["rows"] if x["family_id"] == "negotiation_counter"]
    assert r["drafts"] == 1 and r["sent"] == 1 and r["mean_edit_distance"] == 0.0
    assert api.get("/email-learning/metrics?days=99999").json()["days"] == 3650                 # clamped, not honoured
