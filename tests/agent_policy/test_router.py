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
