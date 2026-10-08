import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

GOOD = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Email": "u1@x", "X-User-Groups": json.dumps(["PROCWISE_ADMIN"])}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setenv("AGENT_POLICY_ORCHESTRATOR_KEY", "o1")
    roles = {"PROCWISE_ADMIN": "Admin", "PROCWISE_VIEWER": "Viewer", "PROCWISE_PROCUMENT_BUYER_ANALYST": "Buyer"}
    monkeypatch.setattr(R, "_role_of", lambda principal: roles.get((principal.claims or {}).get("cognito:groups", [""])[0], "Viewer"))
    audits = []
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: audits.append(kw))
    monkeypatch.setattr(R, "load_registry", lambda conn=None: REGISTRY)
    monkeypatch.setattr(R, "load_settings", lambda conn=None: SETTINGS)
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R.repo, "never_suggest_for", lambda conn, area: False)
    app = FastAPI(); app.include_router(R.router); app.include_router(R.orchestrator_router)
    c = TestClient(app); c.audits = audits
    return c


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


def test_missing_or_wrong_gateway_key_is_refused(client):
    assert client.get("/agent-policies").status_code == 401
    assert client.get("/agent-policies", headers={**GOOD, "X-Gateway-Key": "nope"}).status_code == 401


def test_unset_key_env_refuses_everything(client, monkeypatch):
    monkeypatch.delenv("AGENT_POLICY_GATEWAY_KEY")
    assert client.get("/agent-policies", headers=GOOD).status_code == 503


def test_viewer_cannot_create(client, monkeypatch):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}
    assert client.post("/agent-policies", json={"form": {"name": "x"}}, headers=hdr).status_code == 403
    assert client.audits and client.audits[-1]["status"] == "denied"


def test_buyer_creates_and_write_is_audited(client, monkeypatch):
    monkeypatch.setattr(R.repo, "create_draft", lambda conn, form, actor: {"policyKey": "GEN-0001", "version": 1})
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_PROCUMENT_BUYER_ANALYST"])}
    r = client.post("/agent-policies", json={"form": {"name": "x"}}, headers=hdr)
    assert r.status_code == 200 and r.json() == {"policyKey": "GEN-0001", "version": 1}
    assert client.audits[-1]["action_type"] == "agent_policy.write" and client.audits[-1]["status"] == "allowed"


def test_buyer_cannot_activate(client):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_PROCUMENT_BUYER_ANALYST"])}
    body = {"form": FORM_EXAMPLE, "baseVersion": 1, "intent": "activate", "changeNote": ""}
    assert client.post("/agent-policies/GEN-0001/versions", json=body, headers=hdr).status_code == 403


def test_not_ready_returns_every_problem(client, monkeypatch):
    def refuse(*a, **k): raise R.repo.NotReady([{"field": "owner", "message": "Owner is required."}])
    monkeypatch.setattr(R.repo, "save_version", refuse)
    body = {"form": {"name": "x"}, "baseVersion": 1, "intent": "activate", "changeNote": ""}
    r = client.post("/agent-policies/GEN-0001/versions", json=body, headers=GOOD)
    assert r.status_code == 422 and r.json()["problems"][0]["field"] == "owner"


def test_stale_save_is_409(client, monkeypatch):
    def stale(*a, **k): raise R.repo.StaleVersion("latest is 3")
    monkeypatch.setattr(R.repo, "save_version", stale)
    body = {"form": {"name": "x"}, "baseVersion": 1, "intent": "draft", "changeNote": ""}
    assert client.post("/agent-policies/GEN-0001/versions", json=body, headers=GOOD).status_code == 409


def test_retiring_an_already_retired_policy_is_409(client, monkeypatch):
    def again(*a, **k): raise R.repo.InvalidTransition("already retired")
    monkeypatch.setattr(R.repo, "retire", again)
    r = client.post("/agent-policies/GEN-0001/retire", json={"baseVersion": 2, "changeNote": ""}, headers=GOOD)
    assert r.status_code == 409 and r.json()["detail"] == "This policy is already retired."


def test_preview_hides_json_from_non_admins(client):
    hdr = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=hdr).json()
    assert "compiled" not in r and r["examples"][0]["label"] == "A person decides"
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=GOOD).json()
    assert r["compiled"]["schema"] == "hard-policy/2"


def test_feed_needs_its_own_key(client):
    assert client.get("/orchestrator/agent-policies/v2/live").status_code == 401
    assert client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "k1"}).status_code == 401


def _docs():
    from services.agent_policy.compiler import compile_policy
    good = dict(FORM_EXAMPLE, checked={"by": "u", "at": "2026-10-08T00:00:00Z"})
    bad = json.loads(json.dumps(good)); bad["hidden"]["condition"]["all"][0]["value"] = ["gone.tool"]
    return [compile_policy(good, policy_key="FIN-0001", version=1, status="live", settings=SETTINGS, never_suggest=False),
            compile_policy(bad, policy_key="FIN-0002", version=1, status="live", settings=SETTINGS, never_suggest=False)]


def test_feed_refuses_policy_whose_tool_left_registry(client, monkeypatch):
    docs = _docs()
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: docs)
    r = client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"}).json()
    assert [p["id"] for p in r["policies"]] == ["FIN-0001"]
    assert r["refused"][0]["id"] == "FIN-0002" and r["feed"] == "hard-policy-feed/2"


def test_one_document_that_cannot_be_validated_does_not_sink_the_feed(client, monkeypatch):
    docs = _docs()
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: docs)
    real = R.contract.validate

    def flaky(doc, registry):
        if doc.get("id") == "FIN-0002":
            raise KeyError("boom")
        return real(doc, registry)
    monkeypatch.setattr(R.contract, "validate", flaky)
    r = client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"})
    assert r.status_code == 200
    body = r.json()
    assert [p["id"] for p in body["policies"]] == ["FIN-0001"]
    assert body["refused"] == [{"id": "FIN-0002", "problems": ["could not be validated: KeyError"]}]


# ---- fix round 1 -------------------------------------------------------------------------
BUYER = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_PROCUMENT_BUYER_ANALYST"])}
VIEWER = {**GOOD, "X-User-Groups": json.dumps(["PROCWISE_VIEWER"])}
OKBODY = {"form": {"name": "x"}}


@pytest.mark.parametrize("sub", [None, "", "   "])
def test_blank_or_missing_user_sub_is_401(client, sub):
    hdr = {k: v for k, v in GOOD.items() if k != "X-User-Sub"}
    if sub is not None:
        hdr["X-User-Sub"] = sub
    assert client.get("/agent-policies", headers=hdr).status_code == 401


@pytest.mark.parametrize("groups", ["not json", json.dumps({"a": 1}), json.dumps("PROCWISE_ADMIN"), json.dumps([1, None])])
def test_malformed_or_non_list_groups_mean_no_groups(client, monkeypatch, groups):
    seen = []
    monkeypatch.setattr(R, "_role_of", lambda p: seen.append(p.claims["cognito:groups"]) or "Viewer")
    hdr = {**GOOD, "X-User-Groups": groups}
    assert client.post("/agent-policies", json=OKBODY, headers=hdr).status_code == 403
    assert seen == [[]]


def _policy():
    return {"policyKey": "FIN-0001", "versions": [{"version": 1, "compiled": {"a": 1}}, {"version": 2, "compiled": {"a": 2}}]}


def test_get_policy_strips_compiled_for_non_admin_only(client, monkeypatch):
    monkeypatch.setattr(R.repo, "get_policy", lambda conn, key: _policy())
    for hdr in (VIEWER, BUYER):
        assert all("compiled" not in v for v in client.get("/agent-policies/FIN-0001", headers=hdr).json()["versions"])
    assert all("compiled" in v for v in client.get("/agent-policies/FIN-0001", headers=GOOD).json()["versions"])


def test_feed_is_503_when_its_key_is_unset(client, monkeypatch):
    monkeypatch.delenv("AGENT_POLICY_ORCHESTRATOR_KEY")
    assert client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"}).status_code == 503


class _Cur:
    def execute(self, *a, **k): pass
    def fetchone(self): return ("Finance",)


class _CurConn:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cursor(self): return _Cur()


def test_taxonomy_update_needs_admin(client, monkeypatch):
    monkeypatch.setattr(R, "_conn", _CurConn)
    body = {"subAreas": ["Refunds"], "neverSuggest": False, "secondReviewer": False}
    assert client.put("/agent-policies/taxonomy/Finance", json=body, headers=BUYER).status_code == 403
    r = client.put("/agent-policies/taxonomy/Finance", json=body, headers=GOOD)
    assert r.status_code == 200 and r.json()["subAreas"] == ["General", "Refunds"]


def test_retire_needs_approver(client, monkeypatch):
    monkeypatch.setattr(R.repo, "retire", lambda *a, **k: {"policyKey": "FIN-0001", "version": 3})
    body = {"baseVersion": 2, "changeNote": ""}
    assert client.post("/agent-policies/FIN-0001/retire", json=body, headers=BUYER).status_code == 403
    assert client.post("/agent-policies/FIN-0001/retire", json=body, headers=GOOD).status_code == 200


def test_roles_resolve_through_the_real_rbac_table(client, monkeypatch):
    """_role_of is left REAL: groups -> role goes through rbac with a stand-in policy engine."""
    from services import rbac
    from tests.guardrails.test_rbac import FakePolicyEngine, ROLE_ASSIGNMENT, ROLE_DEFINITION
    import copy
    assign = copy.deepcopy(ROLE_ASSIGNMENT)
    assign["details"]["rules"]["group_to_role"] = {"PROCWISE_ADMIN": "Admin", "PROCWISE_VIEWER": "Viewer"}
    engine = FakePolicyEngine({"role_definition": ROLE_DEFINITION, "role_assignment": assign})
    monkeypatch.setattr(R, "_role_of", lambda principal: rbac.effective_role(principal))
    monkeypatch.setattr(rbac, "_build_engine", lambda: engine)
    monkeypatch.setattr(rbac, "_load_role_assignments", lambda: {})
    rbac.reset_policy_cache()
    monkeypatch.setattr(R.repo, "create_draft", lambda conn, form, actor: {"policyKey": "GEN-0001", "version": 1})
    try:
        assert client.post("/agent-policies", json=OKBODY, headers=VIEWER).status_code == 403
        assert client.post("/agent-policies", json=OKBODY, headers=GOOD).status_code == 200
    finally:
        rbac.reset_policy_cache()


def test_a_non_dict_stored_document_is_refused_not_a_500(client, monkeypatch):
    good = _docs()[0]
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: [None, good])
    r = client.get("/orchestrator/agent-policies/v2/live", headers={"X-Orchestrator-Key": "o1"})
    assert r.status_code == 200
    body = r.json()
    assert [p["id"] for p in body["policies"]] == ["FIN-0001"] and len(body["refused"]) == 1
    assert body["refused"][0]["id"] is None


# ---- final review fixes ------------------------------------------------------------------
def test_preview_returns_the_company_response_time(client):
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=VIEWER).json()
    assert r["companyResponseTime"] == SETTINGS["response_time"] == "PT4H"
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=GOOD).json()
    assert r["companyResponseTime"] == "PT4H"


def test_preview_uses_the_business_areas_real_never_suggest(client, monkeypatch):
    asked = []
    monkeypatch.setattr(R.repo, "never_suggest_for", lambda conn, area: asked.append(area) or area == "Finance")
    r = client.post("/agent-policies/preview", json={"form": FORM_EXAMPLE}, headers=GOOD).json()
    assert asked == ["Finance"] and r["compiled"]["learning"]["eligible"] is False
    other = dict(FORM_EXAMPLE, businessArea="Operations")
    r = client.post("/agent-policies/preview", json={"form": other}, headers=GOOD).json()
    assert r["compiled"]["learning"]["eligible"] is True


def test_create_with_withheld_text_is_a_422(client, monkeypatch):
    def refuse(conn, form, actor): raise R.repo.NotReady([{"field": "form", "code": "withheld_text", "message": "m"}])
    monkeypatch.setattr(R.repo, "create_draft", refuse)
    r = client.post("/agent-policies", json={"form": {"name": "[withheld]"}}, headers=BUYER)
    assert r.status_code == 422 and r.json()["problems"][0]["code"] == "withheld_text"
