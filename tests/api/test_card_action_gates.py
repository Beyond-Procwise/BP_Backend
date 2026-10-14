"""The three writes a card button reaches are refused to a caller who may not write.

The opportunity card offers Pursue and Reject, and the finding card Act and Reject.
Every one of them was authentication-only: `require_user` asked WHO the caller was and
nothing asked WHETHER they may. A Viewer -- the role whose whole definition is "read" --
could reject any opportunity, draft a mail to a supplier, and close any finding.

These tests go through the REAL gate (endpoint_gate.require -> guardrail.authorize ->
rbac) with only the policy engine and the audit writer faked, and each one asserts the
WORK did not happen, not merely that the status code was 403. A gate that refuses after
the row is written has refused nothing.

`write` is not an irreversible class under RoleDefinitionPolicy, so these actions need no
policy row to be governed: the role cap decides, Viewer is refused, Buyer and above
proceed, and every attempt is audited either way. A policy row can narrow it later
(required_role: Approver on a rejection, say) with no change here.
"""
from __future__ import annotations

import contextlib

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import decisions as dr
from api.routers import workflows as wr
from src.services import agent_actions, guardrail, rbac
from tests.guardrails.test_guardrail_gate import engine_with
from tests.guardrails.test_rbac import FakePrincipal


def viewer():
    return FakePrincipal("sub-viewer", {"cognito:groups": ["bp-viewers"]})


def buyer():
    return FakePrincipal("sub-buyer", {"cognito:groups": ["bp-buyers"]})


@pytest.fixture(autouse=True)
def _quiet(monkeypatch):
    """Policy observation and escalation are bookkeeping; they are not what is tested."""
    monkeypatch.setattr(guardrail.policy_observation, "record", lambda **k: True)
    monkeypatch.setattr(guardrail, "_raise_for_a_human", lambda **k: 1)


@pytest.fixture
def serve(monkeypatch):
    """A client over one router, with the gate real and the work spied on.

    `did` collects every call the handler makes to the thing that WRITES. An empty
    `did` after a refusal is the assertion that matters.
    """
    did, audit = [], []
    monkeypatch.setattr(agent_actions, "record_action_or_fail", lambda **k: audit.append(k))

    # --- the opportunity rejection's write -------------------------------------
    monkeypatch.setattr(
        wr, "record_opportunity_feedback",
        lambda nick, oid, **k: did.append(("reject", oid)) or {
            "opportunity_id": oid, "opportunity_ref_id": None, "status": "rejected",
            "reason": k.get("reason"), "user_id": k.get("user_id"),
            "metadata": k.get("metadata"), "updated_on": None,
        },
    )

    # --- the draft's write -----------------------------------------------------
    class _Drafter:
        def __init__(self, *a, **k):
            pass

        def _store_draft(self, draft, *a, **k):
            # The real one STAMPS the committed row's ids onto the dict it is given;
            # the handler treats their absence as "not persisted". Fake it the same way
            # or the buyer's 200 would never be reachable.
            did.append(("draft", None))
            draft.update(unique_id="UID-1", workflow_id="WF-1", draft_record_id=1)

    monkeypatch.setattr(wr, "EmailDraftingAgent", _Drafter)

    # --- the finding decision's write ------------------------------------------
    import engines.decision_engine as de

    class _Engine:
        def __init__(self, *a, **k):
            pass

        def execute(self, finding_id, action, **k):
            did.append(("decide", finding_id, action))
            return {"applied": True, "finding_id": finding_id, "action": action}

    monkeypatch.setattr(de, "DecisionEngine", _Engine)

    def _serve(router, principal, *policies):
        monkeypatch.setattr(rbac, "policy_engine", lambda: engine_with(*policies))
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[router_auth(router)] = lambda: principal
        app.dependency_overrides[router_nick(router)] = lambda: object()
        client = TestClient(app, raise_server_exceptions=False)
        client.did, client.audit = did, audit
        return client

    return _serve


def router_auth(router):
    return wr.require_user if router is wr.router else dr.require_user


def router_nick(router):
    return wr.get_agent_nick if router is wr.router else dr.get_agent_nick


# What each card button posts, and the action name that must govern it.
REJECT = ("/workflows/opportunities/OPP-1/reject", {"reason": "not worth pursuing"},
          "opportunity.reject")
DRAFT = ("/workflows/email/prepare",
         {"to": ["supplier@example.com"], "subject": "s", "body": "b"}, "email.draft")
DECIDE = ("/decisions/finding/77/action", {"action": "dismiss"}, "finding.resolve")

CASES = [REJECT, DRAFT, DECIDE]


def _router_for(path):
    return wr.router if path.startswith("/workflows") else dr.router


@pytest.mark.parametrize("path,body,action", CASES, ids=["reject", "draft", "decide"])
def test_a_viewer_is_refused_and_nothing_is_written(serve, path, body, action):
    client = serve(_router_for(path), viewer())

    r = client.post(path, json=body)

    assert r.status_code == 403, r.text
    assert client.did == [], f"the write happened anyway: {client.did}"


@pytest.mark.parametrize("path,body,action", CASES, ids=["reject", "draft", "decide"])
def test_a_buyer_proceeds(serve, path, body, action):
    client = serve(_router_for(path), buyer())

    r = client.post(path, json=body)

    assert r.status_code == 200, r.text
    assert client.did, "the buyer was permitted but nothing was written"


@pytest.mark.parametrize("path,body,action", CASES, ids=["reject", "draft", "decide"])
def test_every_attempt_is_audited_under_its_own_action_name(serve, path, body, action):
    """Refused and allowed alike. "We saw no denials" and "we were not looking" differ."""
    refused = serve(_router_for(path), viewer())
    refused.post(path, json=body)
    assert (action, "denied") in [(a["action_type"], a["status"]) for a in refused.audit]

    allowed = serve(_router_for(path), buyer())
    allowed.audit.clear()
    allowed.post(path, json=body)
    assert (action, "allowed") in [(a["action_type"], a["status"]) for a in allowed.audit]


@pytest.mark.parametrize("path,body,action", CASES, ids=["reject", "draft", "decide"])
def test_no_principal_is_refused(serve, path, body, action):
    """ASK_AUTH_MODE=off makes require_user yield None. That is the live exposure.

    The CODE differs by route and both answers are right: `act_on_finding` calls
    `_actor` first and refuses an unattributed caller 401 before any gate is asked,
    which is stricter than the gate would be. So this asserts what actually matters --
    refused, and nothing written -- rather than pinning a number that would force the
    weaker of the two behaviours onto whichever route has the stronger one.
    """
    client = serve(_router_for(path), None)

    r = client.post(path, json=body)

    assert r.status_code in (401, 403), r.text
    assert client.did == []
