"""The stage-4 conflict endpoints through the WHOLE app (no database).

User ruling Q5 (2026-10-09): the 2xx bodies of exactly GET /agent-policies/conflicts and
GET /agent-policies/conflicts/{decision_id} pass the output scrubber untouched, like the stage-3
approval reads. A conflict case quotes both policies' excerpts, their documents' file names and
the example action's own values (a deal id like "DL/2024/001"), and an unroutable owner can be a
Cognito group name. Any error answer on those paths, any other method or shape, and POST decide
are still scrubbed.
"""
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_views as CV
from services.agent_policy.enforcement import MASK

HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@example.test",
       "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


def _policy(key, outcome, owner, doc, **over):
    base = {"id": key, "version": 2, "outcome": outcome,
            "situation": "The agent is about to issue a refund or credit above $500.",
            "owner": owner, "businessArea": "Finance / Payments",
            "source": {"document": doc, "reference": "1.1",
                       "excerpt": "Refunds or credits above $500 need approval from the Finance Manager."},
            "latestVersion": 2, "status": "live"}
    if outcome == "approve":
        base["deciders"] = ["Finance Manager", "CFO"]
    base.update(over)
    return base


def _view(**over):
    base = {"caseId": "pc_41", "decisionId": 41, "status": "open", "raisedAt": "2026-10-09T09:00:00+00:00",
            "raisedBy": "save",
            "policies": [_policy("FIN-0012", "approve", "Finance Owner", "Finance Payments Policy"),
                         _policy("CUS-0003", "block", "Customer Owner", "Customer Care Policy")],
            "example": {"tool.name": "refund.issue", "args.amount": 12000, "args.iban": MASK},
            "actionPlain": "Example action that triggers both policies: tool.name refund.issue, args.amount 12000",
            "why": "One policy needs approval from Finance Manager then CFO; the other does not allow this at all.",
            "prior": {"sameConflict": 0, "lastOutcome": None},
            "options": ["keep_both:CUS-0003", "change:FIN-0012", "change:CUS-0003"],
            "optionLabels": {"keep_both:CUS-0003": "Keep both: CUS-0003 takes priority",
                             "change:FIN-0012": "Change FIN-0012", "change:CUS-0003": "Change CUS-0003"},
            "respondWithin": None, "unroutable": [], "proposal": None, "canDecide": True, "decision": None}
    base.update(over)
    return base


@pytest.fixture
def state():
    return {"view": _view(), "history": []}


@pytest.fixture
def client(monkeypatch, state):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: "Viewer")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(CV, "list_conflicts", lambda conn, p, status="open": [state["view"]])
    monkeypatch.setattr(CV, "get_conflict", lambda conn, did, p, **_kw: {**state["view"], "history": state["history"]})
    return TestClient(app)


def _withheld(body) -> bool:
    return "[withheld]" in json.dumps(body)


@pytest.mark.parametrize("path", ["/agent-policies/conflicts", "/agent-policies/conflicts?status=all",
                                  "/agent-policies/conflicts/41"])
def test_benign_answers_pass_the_scrubber_untouched(client, path):
    r = client.get(path, headers=HDR)
    assert r.status_code == 200, r.text
    assert not _withheld(r.json()), r.text


def test_exactly_the_conflict_reads_are_exempt():
    from api import main as M
    for path in ["/agent-policies/conflicts", "/agent-policies/conflicts/41",
                 "/agent-policies/conflicts/123456789012345678"]:
        assert M._agent_policy_screen_exempt("GET", path, 200), path
        assert not M._agent_policy_screen_exempt("GET", path, 404), path        # errors still scrubbed
        assert not M._agent_policy_screen_exempt("GET", path, 422), path
        assert not M._agent_policy_screen_exempt("GET", path + "/", 200), path  # trailing slash
        assert not M._agent_policy_screen_exempt("POST", path, 200), path       # another method
    assert not M._agent_policy_screen_exempt("POST", "/agent-policies/conflicts/41/decide", 200)
    for path in ["/agent-policies/conflicts/abc", "/agent-policies/conflicts/1234567890123456789",
                 "/agent-policies/conflicts/pc_41", "/agent-policies/conflicts/41/history",
                 "/agent-policies/conflicts/41/decide", "/agent-policies/conflictsx"]:
        assert not M._agent_policy_screen_exempt("GET", path, 200), path


@pytest.mark.parametrize("path", ["/agent-policies/conflicts", "/agent-policies/conflicts/41"])
def test_a_4xx_from_an_exempt_conflict_path_is_still_scrubbed(client, monkeypatch, path):
    def _boom(*a, **k):
        raise HTTPException(status_code=404, detail="no row in proc.bp_decision for subject_type policy_conflict")
    monkeypatch.setattr(CV, "list_conflicts", _boom)
    monkeypatch.setattr(CV, "get_conflict", _boom)
    r = client.get(path, headers=HDR)
    assert r.status_code == 404
    assert "bp_decision" not in r.text and "policy_conflict" not in r.text


def test_a_422_on_the_exempt_list_path_is_still_scrubbed_by_the_middleware(client):
    """A validation 422 echoes the query value and never passes the HTTPException handler: only
    the middleware's 2xx-only rule stands between it and the client."""
    r = client.get("/agent-policies/conflicts?status=proc.bp_decision", headers=HDR)
    assert r.status_code == 422
    assert "bp_decision" not in r.text and "[withheld]" in r.text


def test_a_404_for_a_missing_case_is_still_scrubbed(client, monkeypatch):
    monkeypatch.setattr(CV, "get_conflict", lambda conn, did, p, **_kw: None)
    r = client.get("/agent-policies/conflicts/41", headers=HDR)
    assert r.status_code == 404 and r.json() == {"detail": "No such conflict case."}


# Values the scrubber withholds without the ruling: each must arrive exactly as stored.
_REALISTIC = {
    "excerpt with a date": {"policies": [_policy("FIN-0012", "approve", "Finance Owner", "Finance Payments Policy",
                                                 source={"document": "Finance Payments Policy", "reference": "1.1",
                                                         "excerpt": "From 01/04/2026, refunds above $500 need "
                                                                    "approval."})]},
    "document file name": {"policies": [_policy("FIN-0012", "approve", "Finance Owner", "IT_SEC_POLICY_V3.pdf")]},
    "witness value with slashes": {"example": {"tool.name": "deal.award", "args.deal_id": "DL/2024/001"},
                                   "actionPlain": "Example action that triggers both policies: "
                                                  "tool.name deal.award, args.deal_id DL/2024/001"},
    "a real Cognito group (2+ underscores) as an unroutable owner":
        {"unroutable": ["PROCWISE_PROCUMENT_BUYER_ANALYST"], "canDecide": False},
    "a decided case": {"status": "actioned", "canDecide": False,
                       "decision": {"caseId": "pc_41", "decision": "keep_both:CUS-0003", "scope": "standing_rule",
                                    "decidedBy": "sub-owner", "decidedAt": "2026-10-09T10:00:00+00:00",
                                    "reason": "Customer care wins on 01/04/2026 refunds"}},
}


@pytest.mark.parametrize("label", list(_REALISTIC))
def test_realistic_values_survive_the_scrubber(client, state, label):
    state["view"] = _view(**_REALISTIC[label])
    listed = client.get("/agent-policies/conflicts", headers=HDR).json()
    assert listed == {"conflicts": [state["view"]]}, listed
    state["history"] = [state["view"]["decision"]] if state["view"]["decision"] else []
    one = client.get("/agent-policies/conflicts/41", headers=HDR).json()
    assert one == {**state["view"], "history": state["history"]}, one


# ------------------------------------------------------------------ decide (not exempt)
def test_decide_answers_are_still_scrubbed(client, monkeypatch):
    monkeypatch.setattr(R.conflict_cases, "decide_policy", lambda conn, did, **kw: {
        "caseId": f"pc_{did}", "decision": kw["option"], "scope": "this_action", "decidedBy": "u1",
        "decidedAt": "2026-10-09T10:00:00+00:00", "reason": "see proc.bp_agent_policy_conflict_rule",
        "actionId": 7, "applied": "draft_pending"})
    r = client.post("/agent-policies/conflicts/41/decide", json={"option": "change:FIN-0012", "reason": "x"},
                    headers=HDR)
    assert r.status_code == 200 and "bp_agent_policy_conflict_rule" not in r.text

    def _refuse(conn, did, **kw):
        raise CC.ConflictRefused("not_eligible", "Only someone linked in proc.bp_policy_decider_map can decide.", 403)
    monkeypatch.setattr(R.conflict_cases, "decide_policy", _refuse)
    r = client.post("/agent-policies/conflicts/41/decide", json={"option": "change:FIN-0012", "reason": "x"},
                    headers=HDR)
    assert r.status_code == 403 and "bp_policy_decider_map" not in r.text


@pytest.mark.parametrize("code,field", [("reason_required", "reason"), ("limit_required", "limitText"),
                                        ("unknown_option", "option")])
def test_422_refusals_are_problems(client, monkeypatch, code, field):
    def _refuse(conn, did, **kw):
        raise CC.ConflictRefused(code, "Not like that.", 422)
    monkeypatch.setattr(R.conflict_cases, "decide_policy", _refuse)
    r = client.post("/agent-policies/conflicts/41/decide", json={"option": "limit:FIN-0012"}, headers=HDR)
    assert r.status_code == 422
    assert r.json() == {"problems": [{"field": field, "code": code, "message": "Not like that."}]}


@pytest.mark.parametrize("code,status", [("not_eligible", 403), ("not_found", 404), ("not_open", 409),
                                         ("not_signed_in", 401)])
def test_refusals_keep_their_status(client, monkeypatch, code, status):
    def _refuse(conn, did, **kw):
        raise CC.ConflictRefused(code, "No.", status)
    monkeypatch.setattr(R.conflict_cases, "decide_policy", _refuse)
    r = client.post("/agent-policies/conflicts/41/decide", json={"option": "change:FIN-0012", "reason": "x"},
                    headers=HDR)
    assert r.status_code == status


def test_decide_is_audited_before_and_after(client, monkeypatch):
    audits = []
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: audits.append(kw))
    monkeypatch.setattr(R.conflict_cases, "decide_policy", lambda conn, did, **kw: {
        "caseId": f"pc_{did}", "decision": kw["option"], "scope": "this_action", "decidedBy": "u1",
        "decidedAt": "2026-10-09T10:00:00+00:00", "reason": kw["reason"], "actionId": 7, "applied": "draft_pending"})
    r = client.post("/agent-policies/conflicts/41/decide",
                    json={"option": "limit:FIN-0012", "reason": "Too broad", "limitText": "Only refunds"},
                    headers=HDR)
    assert r.status_code == 200, r.text
    assert [(a["action_type"], a["phase"], a["status"]) for a in audits] == [
        ("agent_policy.conflict_decide", "authorize", "allowed"),
        ("agent_policy.conflict_decide", "decide", "done")]
    assert audits[-1]["details"]["decision"] == 41 and audits[-1]["details"]["option"] == "limit:FIN-0012"
    assert audits[-1]["details"]["actionId"] == 7


def test_decide_refusal_and_error_are_audited(client, monkeypatch):
    audits = []
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: audits.append(kw))

    def _refuse(conn, did, **kw):
        raise CC.ConflictRefused("not_open", "Already decided.", 409)
    monkeypatch.setattr(R.conflict_cases, "decide_policy", _refuse)
    client.post("/agent-policies/conflicts/41/decide", json={"option": "change:FIN-0012", "reason": "x"}, headers=HDR)
    assert audits[-1]["status"] == "refused" and audits[-1]["details"]["code"] == "not_open"

    def _boom(conn, did, **kw):
        raise RuntimeError("db down")
    monkeypatch.setattr(R.conflict_cases, "decide_policy", _boom)
    r = client.post("/agent-policies/conflicts/41/decide", json={"option": "change:FIN-0012", "reason": "x"},
                    headers=HDR)
    assert r.status_code == 500 and "db down" not in r.text
    assert audits[-1]["status"] == "error"


def test_decide_passes_the_body_through(client, monkeypatch):
    seen = {}
    monkeypatch.setattr(R.conflict_cases, "decide_policy", lambda conn, did, **kw: seen.update(did=did, **kw) or {
        "caseId": "pc_41", "decision": "x", "scope": "this_action", "decidedBy": "u1", "decidedAt": None,
        "reason": "r", "actionId": 1, "applied": "draft_pending"})
    client.post("/agent-policies/conflicts/41/decide",
                json={"option": "limit:FIN-0012", "reason": "Too broad", "limitText": "Only refunds"}, headers=HDR)
    assert seen["did"] == 41 and seen["option"] == "limit:FIN-0012" and seen["reason"] == "Too broad"
    assert seen["limit_text"] == "Only refunds" and seen["principal"].subject == "u1"


def test_decide_body_limits(client):
    assert client.post("/agent-policies/conflicts/41/decide", json={"reason": "x"}, headers=HDR).status_code == 422
    assert client.post("/agent-policies/conflicts/41/decide", json={"option": "x" * 200, "reason": "x"},
                       headers=HDR).status_code == 422
    assert client.post("/agent-policies/conflicts/41/decide", json={"option": "change:FIN-0012", "reason": "x" * 2001},
                       headers=HDR).status_code == 422
    assert client.post("/agent-policies/conflicts/41/decide",
                       json={"option": "limit:FIN-0012", "reason": "x", "limitText": "y" * 501},
                       headers=HDR).status_code == 422


# ------------------------------------------------------------------ roles + paths
def test_role_floor_and_gateway_key(client, monkeypatch):
    assert client.get("/agent-policies/conflicts", headers={**HDR, "X-Gateway-Key": "nope"}).status_code == 401
    monkeypatch.setattr(R, "_role_of", lambda principal: "Nobody")
    for path in ["/agent-policies/conflicts", "/agent-policies/conflicts/41"]:
        assert client.get(path, headers=HDR).status_code == 403
    r = client.post("/agent-policies/conflicts/41/decide", json={"option": "change:FIN-0012", "reason": "x"},
                    headers=HDR)
    assert r.status_code == 403


def test_status_allow_list(client):
    for s in ("open", "closed", "all"):
        assert client.get(f"/agent-policies/conflicts?status={s}", headers=HDR).status_code == 200
    assert client.get("/agent-policies/conflicts?status=everything", headers=HDR).status_code == 422
    assert client.get("/agent-policies/conflicts/abc", headers=HDR).status_code == 422


def test_conflicts_routes_are_not_read_as_policy_keys(client, monkeypatch):
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key, **_kw: pytest.fail(f"read {key} as a policy"))
    assert client.get("/agent-policies/conflicts", headers=HDR).status_code == 200
    assert client.get("/agent-policies/conflicts/41", headers=HDR).status_code == 200
    paths = [getattr(r, "path", "") for r in R.router.routes]
    for p in ("/agent-policies/conflicts", "/agent-policies/conflicts/{decision_id}",
              "/agent-policies/conflicts/{decision_id}/decide"):
        assert paths.index(p) < paths.index("/agent-policies/{key}"), p


# ------------------------------------------------------------------ pure helpers
def test_link_ref_names_a_conflict_by_id():
    from services.agent_policy import approval_views as V
    assert V._link_ref("conflict:41") == {"conflictId": 41}
    assert V._link_ref("conflict:pc_41") == {}
    assert V._link_ref("decision:41") == {"decisionId": 41}


def test_conflict_block_masks_args_and_witness_unless_unmasked():
    from services.agent_policy import approval_views as V
    live = {"decision_id": 77, "rationale": "The policies name different approvers: A and B.",
            "options": ["approve", "reject"],
            "facts": {"policies": [{"id": "FIN-0012"}, {"id": "CUS-0003"}],
                      "priorDecisions": {"sameConflict": 2, "lastOutcome": "approve"},
                      "action": {"tool": "refund.issue", "args": {"amount": 900, "order": "A-1"},
                                 "plain": "Issuing a refund"},
                      "summary": {"respondWithin": "PT4H"}},
            "evidence": [{"kind": "overlap", "example": {"tool.name": "refund.issue", "args.amount": 900}}]}
    hidden = V.conflict_block(77, live, unmasked=False, sensitive={"args.amount"})
    assert hidden["args"] == {"amount": MASK, "order": "A-1"}
    assert hidden["example"] == {"tool.name": "refund.issue", "args.amount": MASK}
    assert "900" not in json.dumps(hidden)
    assert (hidden["caseId"], hidden["why"], hidden["options"], hidden["respondWithin"], hidden["prior"]) == \
           ("pc_77", live["rationale"], ["approve", "reject"], "PT4H", {"sameConflict": 2, "lastOutcome": "approve"})
    assert hidden["policies"] == live["facts"]["policies"] and hidden["actionPlain"] == "Issuing a refund"
    shown = V.conflict_block(77, live, unmasked=True, sensitive={"args.amount"})
    assert shown["args"] == {"amount": 900, "order": "A-1"} and shown["example"]["args.amount"] == 900
    assert V.conflict_block(78, None, unmasked=False, sensitive=set())["caseId"] == "pc_78"


def test_case_view_adds_conflict_only_for_a_member_case():
    from services.agent_policy import approval_views as V
    case = {"decision_id": 1, "subject_id": "FIN-0012:9", "policy_name": "FIN-0012", "status": "open",
            "levels": [{"name": "Finance Manager"}], "current_level": 0, "respond_by": None,
            "on_timeout": "reject", "options": ["approve", "reject"], "created_at": None,
            "facts": {"approvalInputs": ["args.amount"], "policy": {"id": "FIN-0012", "version": 3},
                      "action": {"tool": "refund.issue", "args": {"amount": 900}, "reason": "r"}}}
    assert "conflict" not in V.case_view(case, unmasked=False, sensitive=set(), doc=None, decidable=False)
    case["facts"]["liveConflict"] = 77
    live = {"decision_id": 77, "rationale": "why", "options": ["approve", "reject"],
            "facts": {"action": {"args": {"amount": 900}}}, "evidence": []}
    v = V.case_view(case, unmasked=False, sensitive={"args.amount"}, doc=None, decidable=False, live=live)
    assert v["conflict"]["caseId"] == "pc_77" and v["conflict"]["args"] == {"amount": MASK}
