"""The stage-3 approval/notification/decider/firing endpoints through the WHOLE app (no database).

These GET paths are NOT on the user-approved screen exemption list (Task 7 ruling: never widen
it from here). Benign answers must pass the scrubber untouched; the realistic values the
scrubber DOES withhold are recorded below as strict xfails, pending a user ruling -- when the
ruling lands (an exemption, or a scrubber change), the xfail turns into a failure and this file
must be updated with it.
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
    return {"case": _case(), "firing": _firing(), "decider": _decider(), "note": _note()}


@pytest.fixture
def client(monkeypatch, state):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: "Viewer")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(V, "list_cases", lambda conn, p, is_admin, status="open": [state["case"]])
    monkeypatch.setattr(V, "get_case", lambda conn, did, p, is_admin: {
        **state["case"], "history": {"decisions": [], "notes": [state["note"]], "firings": [state["firing"]]}})
    monkeypatch.setattr(V, "policy_firings", lambda conn, key, limit: [state["firing"]])
    monkeypatch.setattr(V, "my_notifications", lambda conn, p, limit: [state["note"]])
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


def test_the_new_paths_are_not_exempt():
    """No exemption was added: the scrubber IS in the path for every one of them."""
    from api import main as M
    for path in ["/agent-policies/approvals", "/agent-policies/approvals/41", "/agent-policies/notifications",
                 "/agent-policies/deciders", "/agent-policies/FIN-0012/firings"]:
        assert not M._agent_policy_screen_exempt("GET", path, 200)


def test_the_scrubber_is_really_in_the_path(client, state):
    """Prove the benign test can fail: a value the scrubber always withholds comes back withheld."""
    state["case"] = _case(reference="see proc.bp_decision")
    assert _withheld(client.get("/agent-policies/approvals", headers=HDR).json())


# ------------------------------------------------------------------ withheld today (user ruling pending)
_WITHHELD = {
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


@pytest.mark.xfail(strict=True, reason="withheld by output safety; user ruling pending (Task 7 report)")
@pytest.mark.parametrize("label", list(_WITHHELD))
def test_realistic_values_survive_the_scrubber(client, state, label):
    kind, over = _WITHHELD[label]
    state[kind] = _MAKE[kind](**over)
    body = client.get(_PATH[kind], headers=HDR).json()
    assert not _withheld(body), body


# ------------------------------------------------------------------ status mapping (no database)
def test_reject_without_reason_is_422_problems(client, monkeypatch):
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
