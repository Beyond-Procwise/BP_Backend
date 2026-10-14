"""The stage-3 approval/notification/decider/firing endpoints through the WHOLE app (no database).

User ruling 2026-10-08 (Task 7 review): the 2xx bodies of exactly these GETs pass the output
scrubber untouched, like the stage-1/2 screen reads -- approvals, approvals/{id}, notifications,
deciders, {key}/firings. Realistic values the scrubber used to withhold (a dated excerpt, a file
name, a slashed deal id, a real Cognito group) now arrive intact. Anything else -- another method,
a trailing slash, a malformed id, any error answer, and POST decide -- is still scrubbed.
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import approval_views as V
from services.agent_policy import approvals as A
from services.agent_policy.enforcement import MASK

HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@example.test",
       "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


def _case(**over):
    base = {"id": 41, "policyKey": "FIN-0012", "policyVersion": 3, "status": "open",
            "actionPlain": "Issuing a refund or credit",
            "inputs": [{"field": "args.amount", "name": "Refund amount", "value": 900, "missing": False,
                        "masked": False, "unit": "USD"},
                       {"field": "args.iban", "name": "Bank account", "value": MASK, "missing": False,
                        "masked": True}],
            "agentReason": "The customer was charged twice",
            "situation": "The agent is about to issue a refund or credit above $500.",
            "excerpt": "Refunds or credits above $500 need approval from the Finance Manager.",
            "reference": "1.1", "document": "Finance Payments Policy", "level": 0,
            "levelName": "Finance Manager", "levels": ["Finance Manager", "Finance Director"],
            "respondBy": "2026-10-08T13:00:00+00:00", "onTimeout": "escalate_next",
            "options": ["approve", "reject"], "reasonRequiredOn": ["reject"], "unroutable": [],
            "requestedBy": "jane.doe@acme.example", "createdAt": "2026-10-08T09:00:00+00:00",
            "canDecide": True}
    base.update(over)
    return base


def _firing(**over):
    base = {"id": 7, "policyKey": "FIN-0012", "policyVersion": 3, "checkpoint": "tool.call.before",
            "actionName": "refund.issue", "agent": "agent_nick", "workflowId": "wf-1",
            "requestedBy": "jane.doe@acme.example", "outcome": "approve", "result": "timed_out",
            "matchedValues": {"args.amount": 900, "args.iban": MASK}, "missingInputs": [],
            "decisionId": 41, "decidedLevel": 1, "decidedBy": "system:timeout",
            "decidedAt": "2026-10-08T17:00:00+00:00", "reason": A.TIMEOUT_REASON, "durationMs": 12,
            "reversalOf": None, "createdAt": "2026-10-08T09:00:00+00:00"}
    base.update(over)
    return base


def _decider(**over):
    base = {"name": "Finance Manager", "groups": ["PROCWISE_FINANCE"], "emails": ["fm@acme.example"],
            "notes": "Month-end cover by the controller", "lastModifiedBy": "u-admin",
            "lastModifiedAt": "2026-10-08T09:00:00+00:00"}
    base.update(over)
    return base


def _note(**over):
    base = {"id": 5, "recipient": "Finance Manager",
            "message": "Issuing a refund or credit (policy FIN-0012) needs your decision; "
                       "the previous approver did not answer in time.",
            "policyKey": "FIN-0012", "decisionId": 41, "read": False, "createdAt": "2026-10-08T09:00:00+00:00"}
    base.update(over)
    return base


@pytest.fixture
def state():
    return {"case": _case(), "firing": _firing(), "decider": _decider(), "note": _note(),
            "replay": {"outcome": "ran", "resultSummary": f"Refund of 900 to {MASK} issued", "error": None,
                       "at": "2026-10-08T09:05:00+00:00"}}


@pytest.fixture
def client(monkeypatch, state):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: "Viewer")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(V, "list_cases", lambda conn, p, is_admin, status="open": [state["case"]])
    monkeypatch.setattr(V, "get_case", lambda conn, did, p, is_admin: {
        **state["case"], "history": {"decisions": [], "notes": [state["note"]], "firings": [state["firing"]],
                    "replay": state["replay"]}})
    monkeypatch.setattr(V, "policy_firings", lambda conn, key, limit: [state["firing"]])
    monkeypatch.setattr(V, "my_notifications", lambda conn, p, limit, is_admin=False: [state["note"]])
    monkeypatch.setattr(V, "list_deciders", lambda conn: [state["decider"]])
    return TestClient(app)


def _withheld(body) -> bool:
    return "[withheld]" in json.dumps(body)


_GETS = ["/agent-policies/approvals", "/agent-policies/approvals/41", "/agent-policies/notifications?mine=1",
         "/agent-policies/deciders", "/agent-policies/FIN-0012/firings"]


@pytest.mark.parametrize("path", _GETS)
def test_benign_answers_pass_the_scrubber_untouched(client, path):
    r = client.get(path, headers=HDR)
    assert r.status_code == 200, r.text
    assert not _withheld(r.json()), r.text


_EXEMPT_PATHS = ["/agent-policies/approvals", "/agent-policies/approvals/41", "/agent-policies/notifications",
                 "/agent-policies/deciders", "/agent-policies/FIN-0012/firings"]


def test_exactly_the_new_reads_are_exempt():
    from api import main as M
    for path in _EXEMPT_PATHS:
        assert M._agent_policy_screen_exempt("GET", path, 200), path
        assert not M._agent_policy_screen_exempt("GET", path, 404), path        # errors still scrubbed
        assert not M._agent_policy_screen_exempt("GET", path + "/", 200), path  # trailing slash
        assert not M._agent_policy_screen_exempt("POST", path, 200), path       # another method
    for path in ["/agent-policies/approvals/41/decide", "/agent-policies/notifications/5/read"]:
        assert not M._agent_policy_screen_exempt("POST", path, 200), path       # writes stay scrubbed
    for path in ["/agent-policies/approvals/abc", "/agent-policies/approvals/1234567890123456789",
                 "/agent-policies/fin-0012/firings", "/agent-policies/FIN-0012/firings/x",
                 "/agent-policies/deciders/Finance Manager", "/agent-policies/approvals/41/history"]:
        assert not M._agent_policy_screen_exempt("GET", path, 200), path


def test_decide_answers_are_still_scrubbed(client, monkeypatch):
    """POST decide is not exempt: a 2xx with an internal name in it is withheld, and so is a refusal."""
    monkeypatch.setattr(R.approval_views, "readable", lambda conn, did, p, is_admin: True)
    monkeypatch.setattr(R.approvals, "act", lambda conn, did, **kw: {
        "decisionId": did, "result": "rejected", "verb": "reject", "level": 0, "levelName": "Finance Manager",
        "actionId": 1, "reason": "see proc.bp_decision"})
    r = client.post("/agent-policies/approvals/41/decide", json={"verb": "reject", "reason": "x"}, headers=HDR)
    assert r.status_code == 200 and r.json()["reason"] != "see proc.bp_decision"

    def _refuse(conn, did, **kw):
        raise A.ApprovalRefused("not_eligible", "Only someone linked to proc.bp_policy_decider_map can decide.", 403)
    monkeypatch.setattr(R.approvals, "act", _refuse)
    r = client.post("/agent-policies/approvals/41/decide", json={"verb": "approve"}, headers=HDR)
    assert r.status_code == 403 and "bp_policy_decider_map" not in r.text


def test_an_error_answer_on_an_exempt_path_is_still_scrubbed(client, monkeypatch):
    def _boom(conn, did, p, is_admin):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="no row in proc.bp_decision")
    monkeypatch.setattr(V, "get_case", _boom)
    r = client.get("/agent-policies/approvals/41", headers=HDR)
    assert r.status_code == 404 and "bp_decision" not in r.text


# Values the scrubber withheld before the ruling: each must now arrive exactly as stored.
_REALISTIC = {
    "excerpt with a date": ("case", {"excerpt": "From 01/04/2026, refunds above $500 need approval."}),
    "document file name": ("case", {"document": "IT_SEC_POLICY_V3.pdf"}),
    "input value with slashes": ("case", {"inputs": [{"field": "args.deal_id", "name": "Deal",
                                                      "value": "DL/2024/001", "missing": False, "masked": False}]}),
    "agent reason naming a deal": ("case", {"agentReason": "Ranking suppliers for deal DL/2024/001"}),
    "firing matched value with slashes": ("firing", {"matchedValues": {"args.deal_id": "DL/2024/001"}}),
    "a real Cognito group (2+ underscores)": ("decider", {"groups": ["PROCWISE_PROCUMENT_BUYER_ANALYST"]}),
}
_PATH = {"case": "/agent-policies/approvals", "firing": "/agent-policies/FIN-0012/firings",
         "decider": "/agent-policies/deciders"}
_MAKE = {"case": _case, "firing": _firing, "decider": _decider}
_KEY = {"case": "approvals", "firing": "firings", "decider": "deciders"}


@pytest.mark.parametrize("label", list(_REALISTIC))
def test_realistic_values_survive_the_scrubber(client, state, label):
    kind, over = _REALISTIC[label]
    state[kind] = _MAKE[kind](**over)
    body = client.get(_PATH[kind], headers=HDR).json()
    assert body[_KEY[kind]] == [state[kind]], body


def test_realistic_values_survive_on_one_case_and_notifications(client, state):
    state["case"] = _case(excerpt="From 01/04/2026, refunds above $500 need approval.", document="IT_SEC_POLICY_V3.pdf")
    state["firing"] = _firing(matchedValues={"args.deal_id": "DL/2024/001"})
    state["note"] = _note(recipient="PROCWISE_FINANCE_REVIEWER_APPROVER")
    one = client.get("/agent-policies/approvals/41", headers=HDR).json()
    assert one["excerpt"] == state["case"]["excerpt"] and one["document"] == "IT_SEC_POLICY_V3.pdf"
    assert one["history"]["firings"] == [state["firing"]]
    notes = client.get("/agent-policies/notifications?mine=1", headers=HDR).json()["notifications"]
    assert notes == [state["note"]]


# ------------------------------------------------------------------ status mapping (no database)
def test_reject_without_reason_is_422_problems(client, monkeypatch):
    monkeypatch.setattr(R.approval_views, "readable", lambda conn, did, p, is_admin: True)
    def _act(conn, did, **kw):
        raise A.ApprovalRefused("reason_required", "A rejection needs a reason.", 422)
    monkeypatch.setattr(R.approvals, "act", _act)
    r = client.post("/agent-policies/approvals/41/decide", json={"verb": "reject"}, headers=HDR)
    assert r.status_code == 422
    assert r.json() == {"problems": [{"field": "reason", "code": "reason_required",
                                      "message": "A rejection needs a reason."}]}


@pytest.mark.parametrize("code,status", [("not_eligible", 403), ("self_approval", 403), ("not_open", 409),
                                         ("not_found", 404)])
def test_refusals_keep_their_status(client, monkeypatch, code, status):
    monkeypatch.setattr(R.approval_views, "readable", lambda conn, did, p, is_admin: True)
    def _act(conn, did, **kw):
        raise A.ApprovalRefused(code, "You cannot decide on an action your own request triggered.", status)
    monkeypatch.setattr(R.approvals, "act", _act)
    r = client.post("/agent-policies/approvals/41/decide", json={"verb": "approve"}, headers=HDR)
    assert r.status_code == status


def test_approvals_routes_are_not_read_as_policy_keys(client, monkeypatch):
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key: pytest.fail(f"read {key} as a policy"))
    for path in ["/agent-policies/approvals", "/agent-policies/notifications", "/agent-policies/deciders"]:
        assert client.get(path, headers=HDR).status_code == 200


def test_limits(client):
    assert client.get("/agent-policies/FIN-0012/firings?limit=201", headers=HDR).status_code == 422
    assert client.get("/agent-policies/FIN-0012/firings?limit=0", headers=HDR).status_code == 422
    assert client.get("/agent-policies/notifications?mine=0", headers=HDR).status_code == 422
    assert client.get("/agent-policies/approvals?status=everything", headers=HDR).status_code == 422
    assert client.get("/agent-policies/fin-12/firings", headers=HDR).status_code == 404


def test_replay_outcome_passes_the_scrubber(client, state):
    r = client.get("/agent-policies/approvals/41", headers=HDR).json()["history"]["replay"]
    assert r == state["replay"]
    state["replay"] = {"outcome": "error", "resultSummary": None, "error": "The supplier service did not answer.",
                       "at": "2026-10-08T09:05:00+00:00"}
    assert client.get("/agent-policies/approvals/41", headers=HDR).json()["history"]["replay"] == state["replay"]


def test_replay_subject_type_matches_the_replay_module():
    from services.agent_policy import replay
    assert V.REPLAY_SUBJECT_TYPE == replay.SUBJECT_TYPE


# ------------------------------------------------------------------ pure helpers
def test_decider_validation():
    p, g, e = V.decider_problems("Finance Manager", [" G1 ", "G1"], ["A@B.Co"], None)
    assert p == [] and g == ["G1"] and e == ["a@b.co"]
    assert V.decider_problems("Finance Manager", [], [], None)[0][0]["code"] == "someone_required"
    assert V.decider_problems("1bad", ["G"], [], None)[0][0]["field"] == "name"
    assert V.decider_problems("X", ["G"], ["nope"], None)[0][0]["field"] == "emails"
    assert V.decider_problems("X", [3], [], None)[0][0]["field"] == "groups"


def test_case_view_masks_unless_the_caller_may_decide():
    case = {"decision_id": 1, "subject_id": "FIN-0012:9", "policy_name": "FIN-0012", "status": "open",
            "levels": [{"name": "Finance Manager"}], "current_level": 0, "respond_by": None,
            "on_timeout": "reject", "options": ["approve", "reject"], "created_at": None,
            "facts": {"approvalInputs": ["args.iban", "args.amount"], "policy": {"id": "FIN-0012", "version": 3},
                      "action": {"tool": "refund.issue", "args": {"iban": "GB29", "amount": 900}, "reason": "r"},
                      "requestedBy": "req@x"}}
    masked = V.case_view(case, unmasked=False, sensitive={"args.iban"}, doc=None, decidable=False)
    assert [i["value"] for i in masked["inputs"]] == [MASK, 900] and masked["canDecide"] is False
    shown = V.case_view(case, unmasked=True, sensitive={"args.iban"}, doc=None, decidable=True)
    assert [i["value"] for i in shown["inputs"]] == ["GB29", 900] and shown["canDecide"] is True
    assert "args" not in json.dumps(shown["inputs"][0]["value"])   # full tool args are never returned


def test_link_refs_are_ids_not_routes():
    assert V._link_ref("decision:41") == {"decisionId": 41}
    assert V._link_ref("agent-policy:FIN-0012") == {"policyKey": "FIN-0012"}
    assert V._link_ref("https://x/y") == {}


def test_replay_later_resolves_agent_nick_on_the_request_thread(monkeypatch):
    import threading
    from services.agent_policy import replay
    seen = {}
    monkeypatch.setattr(replay, "_resolve_agent_nick",
                        lambda nick: seen.setdefault("resolved_on", threading.current_thread().name) and "NICK")
    monkeypatch.setattr(replay, "run", lambda did, agent_nick=None: seen.update(did=did, nick=agent_nick,
                                                                                 ran_on=threading.current_thread().name))
    R._replay_later(41)
    R._replay_pool.submit(lambda: None).result(timeout=5)   # the pool runs in order: wait for ours
    import time
    for _ in range(50):
        if "did" in seen:
            break
        time.sleep(0.05)
    assert seen["did"] == 41 and seen["nick"] == "NICK"
    assert seen["resolved_on"] == threading.current_thread().name and seen["ran_on"] != seen["resolved_on"]
