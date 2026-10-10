"""The stage-5 suggestion endpoints through the WHOLE app (no database).

User ruling Q1 (2026-10-10): the 2xx bodies of exactly GET /agent-policies/suggestions and
GET /agent-policies/suggestions/{id}/draft pass the output scrubber untouched, on the same terms
as the rulings of 2026-10-08 and Q5. A card names the customer's own policy and its inputs, and
the draft IS the customer's policy form: its situation, excerpt and document name. Any error
answer on those paths, any other method or shape, and POST …/act are still scrubbed.
"""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import learning as L
from tests.agent_policy.fixtures import FORM_EXAMPLE

HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@example.test",
       "X-User-Groups": json.dumps(["PROCWISE_BUYER"])}
DATED = "From 01/04/2026 the agent is about to issue a refund or credit above $500."


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


def _card(**over):
    base = {"id": 12, "policyKey": "FIN-0012", "policyName": "Refunds from 01/04/2026 (IT_SEC_POLICY_V3.pdf)",
            "status": "open", "kind": "threshold",
            "change": "Raise the limit on Refund amount from 500 USD to 800 USD?",
            "evidence": "30 of 30 requests between 500 USD and 800 USD were approved, "
                        "by 3 different people over 40 days.",
            "stats": {"decisions": 30, "approved": 30, "approvers": 3, "days": 40, "medianSeconds": 60.0,
                      "wilsonLower": 0.8865},
            "suggestionConfidence": "High", "canDraft": True, "updatedAt": "2026-10-10T00:00:00+00:00"}
    base.update(over)
    return base


def _draft():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["situation"] = DATED
    form["source"]["document"] = "IT_SEC_POLICY_V3.pdf"
    form["source"]["excerpt"] = "From 01/04/2026, refunds above $500 need approval (deal DL/2024/001)."
    return {"policyKey": "FIN-0012", "baseVersion": 3, "form": form,
            "changeNote": "Suggested by learning: Raise the limit on Refund amount from 500 USD to 800 USD?",
            "checkWording": ["messageForPerson"], "changedWording": ["situation", "messageForAgent"],
            "suggestionId": 12}


@pytest.fixture
def state():
    return {"cards": [_card()], "draft": _draft(), "role": "Buyer", "audits": []}


@pytest.fixture
def client(monkeypatch, state):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: state["role"])
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: state["audits"].append(kw))
    monkeypatch.setattr(L, "list_cards", lambda conn, status="open": state["cards"])
    monkeypatch.setattr(L, "draft", lambda conn, sid: state["draft"])
    monkeypatch.setattr(L, "act", lambda conn, sid, verb, **kw: {**state["cards"][0], "status": "dismissed"})
    return TestClient(app)


def _refuse(code, message):
    def _raise(*a, **k):
        raise L.Refused(code, message)
    return _raise


# ------------------------------------------------------------------ ruling Q1: the two reads
def test_the_queue_passes_the_customers_own_wording_through(client, state):
    r = client.get("/agent-policies/suggestions", headers=HDR)
    assert r.status_code == 200, r.text
    assert r.json() == {"suggestions": state["cards"]}, r.text
    assert "01/04/2026" in r.text and "[withheld]" not in r.text
    assert client.get("/agent-policies/suggestions?status=all", headers=HDR).json() == {"suggestions": state["cards"]}


def test_the_draft_passes_the_customers_own_policy_form_through(client, state):
    r = client.get("/agent-policies/suggestions/12/draft", headers=HDR)
    assert r.status_code == 200, r.text
    assert r.json() == state["draft"], r.text
    assert r.json()["form"]["situation"] == DATED and "[withheld]" not in r.text


def test_exactly_the_two_suggestion_reads_are_exempt():
    from api import main as M
    for path in ["/agent-policies/suggestions", "/agent-policies/suggestions/12/draft",
                 "/agent-policies/suggestions/123456789012345678/draft"]:
        assert M._agent_policy_screen_exempt("GET", path, 200), path
        for status in (404, 409, 422, 500):
            assert not M._agent_policy_screen_exempt("GET", path, status), (path, status)   # errors still scrubbed
        assert not M._agent_policy_screen_exempt("GET", path + "/", 200), path              # trailing slash
        assert not M._agent_policy_screen_exempt("POST", path, 200), path                   # another method
    for method in ("POST", "GET"):
        for status in (200, 201):
            assert not M._agent_policy_screen_exempt(method, "/agent-policies/suggestions/12/act", status)
    for path in ["/agent-policies/suggestions/12", "/agent-policies/suggestions/abc/draft",
                 "/agent-policies/suggestions/1234567890123456789/draft", "/agent-policies/suggestions/12/draft/x",
                 "/agent-policies/suggestions/-1/draft", "/agent-policies/suggestionsx", "/agent-policies/suggestions/draft"]:
        assert not M._agent_policy_screen_exempt("GET", path, 200), path


# ------------------------------------------------------------------ still scrubbed
def test_a_409_from_the_draft_is_still_scrubbed(client, monkeypatch):
    monkeypatch.setattr(L, "draft", _refuse("policy_changed", "See proc.bp_agent_policy_suggestion for 01/04/2026."))
    r = client.get("/agent-policies/suggestions/12/draft", headers=HDR)
    assert r.status_code == 409
    assert "bp_agent_policy_suggestion" not in r.text and "01/04/2026" not in r.text


def test_a_422_on_the_queue_is_still_scrubbed(client):
    r = client.get("/agent-policies/suggestions?status=proc.bp_decision", headers=HDR)
    assert r.status_code == 422 and "bp_decision" not in r.text and "[withheld]" in r.text


def test_a_failure_on_an_exempt_path_says_nothing_about_the_internals(client, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("relation proc.bp_agent_policy_suggestion does not exist")
    monkeypatch.setattr(L, "list_cards", _boom)
    monkeypatch.setattr(L, "draft", _boom)
    quiet = TestClient(client.app, raise_server_exceptions=False)
    for path in ("/agent-policies/suggestions", "/agent-policies/suggestions/12/draft"):
        r = quiet.get(path, headers=HDR)
        assert r.status_code == 500 and "bp_agent_policy_suggestion" not in r.text, path


def test_the_act_answer_is_still_scrubbed(client, state, monkeypatch):
    r = client.post("/agent-policies/suggestions/12/act", json={"verb": "dismiss"}, headers=HDR)
    assert r.status_code == 201, r.text
    assert "01/04/2026" not in r.text and "[withheld]" in r.text, "POST is not part of ruling Q1"
    assert (r.json()["id"], r.json()["status"]) == (12, "dismissed"), "what the screen acts on survives"
    monkeypatch.setattr(L, "act", _refuse("not_open", "Row locked in proc.bp_agent_policy_suggestion."))
    r = client.post("/agent-policies/suggestions/12/act", json={"verb": "dismiss"}, headers=HDR)
    assert r.status_code == 409 and "bp_agent_policy_suggestion" not in r.text


# ------------------------------------------------------------------ a refusal reads as written
@pytest.mark.parametrize("code,message", [
    ("not_open", "This suggestion is no longer open."),
    ("not_confident", "There is not enough evidence yet."),
    ("policy_changed", "This policy has changed since the suggestion was made.")])
def test_a_refusal_is_a_409_whose_code_and_message_arrive_readable(client, monkeypatch, code, message):
    monkeypatch.setattr(L, "draft", _refuse(code, message))
    monkeypatch.setattr(L, "act", _refuse(code, message))
    r = client.get("/agent-policies/suggestions/12/draft", headers=HDR)
    assert r.status_code == 409 and r.json() == {"code": code, "message": message}
    r = client.post("/agent-policies/suggestions/12/act", json={"verb": "not_yet"}, headers=HDR)
    assert r.status_code == 409 and r.json() == {"code": code, "message": message}


def test_an_unknown_suggestion_is_a_404(client, monkeypatch):
    monkeypatch.setattr(L, "draft", _refuse("not_found", "No such suggestion."))
    monkeypatch.setattr(L, "act", _refuse("not_found", "No such suggestion."))
    r = client.get("/agent-policies/suggestions/12/draft", headers=HDR)
    assert r.status_code == 404 and r.json() == {"detail": "No such suggestion."}
    r = client.post("/agent-policies/suggestions/12/act", json={"verb": "not_yet"}, headers=HDR)
    assert r.status_code == 404 and r.json() == {"detail": "No such suggestion."}


# ------------------------------------------------------------------ roles, audit, shapes, order
def test_role_floors_and_the_gateway_key(client, state):
    assert client.get("/agent-policies/suggestions", headers={**HDR, "X-Gateway-Key": "nope"}).status_code == 401
    state["role"] = "Viewer"
    assert client.get("/agent-policies/suggestions", headers=HDR).status_code == 200
    assert client.get("/agent-policies/suggestions/12/draft", headers=HDR).status_code == 403
    assert client.post("/agent-policies/suggestions/12/act", json={"verb": "dismiss"}, headers=HDR).status_code == 403
    state["role"] = "Nobody"
    assert client.get("/agent-policies/suggestions", headers=HDR).status_code == 403


def test_act_passes_who_and_what_and_is_audited_before_and_after(client, state, monkeypatch):
    seen = {}
    monkeypatch.setattr(L, "act", lambda conn, sid, verb, **kw: seen.update(sid=sid, verb=verb, **kw) or _card(
        status="not_yet", canDraft=False))
    r = client.post("/agent-policies/suggestions/12/act", json={"verb": "not_yet"}, headers=HDR)
    assert r.status_code == 201, r.text
    assert (seen["sid"], seen["verb"], seen["actor"]) == (12, "not_yet", "u1") and seen["now"].tzinfo is not None
    assert [(a["action_type"], a["phase"], a["status"]) for a in state["audits"]] == [
        ("agent_policy.suggestion_act", "authorize", "allowed"), ("agent_policy.suggestion_act", "act", "done")]
    assert state["audits"][0]["details"] == {"suggestion": 12, "verb": "not_yet", "principal": "u1",
                                             "role": "Buyer", "minimum": "Buyer"}


def test_a_failed_act_is_audited_and_says_nothing_about_the_internals(client, state, monkeypatch):
    def _boom(*a, **k):
        raise RuntimeError("db down")
    monkeypatch.setattr(L, "act", _boom)
    r = client.post("/agent-policies/suggestions/12/act", json={"verb": "dismiss"}, headers=HDR)
    assert r.status_code == 500 and "db down" not in r.text
    assert (state["audits"][-1]["phase"], state["audits"][-1]["status"]) == ("act", "error")


def test_shapes(client):
    assert client.get("/agent-policies/suggestions?status=closed", headers=HDR).status_code == 422
    assert client.get("/agent-policies/suggestions/abc/draft", headers=HDR).status_code == 422
    assert client.get("/agent-policies/suggestions/0/draft", headers=HDR).status_code == 422
    for body in ({}, {"verb": "apply"}, {"verb": "draft"}, {"verb": "not_yet "}):
        assert client.post("/agent-policies/suggestions/12/act", json=body, headers=HDR).status_code == 422, body


def test_suggestion_routes_are_not_read_as_policy_keys(client, monkeypatch):
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key, **_kw: pytest.fail(f"read {key} as a policy"))
    assert client.get("/agent-policies/suggestions", headers=HDR).status_code == 200
    assert client.get("/agent-policies/suggestions/12/draft", headers=HDR).status_code == 200
    paths = [getattr(r, "path", "") for r in R.router.routes]
    for p in ("/agent-policies/suggestions", "/agent-policies/suggestions/{suggestion_id}/draft",
              "/agent-policies/suggestions/{suggestion_id}/act"):
        for swallower in ("/agent-policies/{key}", "/agent-policies/{key}/firings", "/agent-policies/{key}/versions"):
            assert paths.index(p) < paths.index(swallower), (p, swallower)
