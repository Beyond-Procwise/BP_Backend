"""The orchestrator feed through the WHOLE app -- output-safety middleware included.

The router's own tests use a bare FastAPI app, which is how the final review's Critical
shipped: the middleware rewrote `outputs.toAgent.reason` (it says "tool call"), withheld an
excerpt holding a date like 01/04/2026 (two slashes look like a route), and would have
withheld a condition value like "DL/2024/001" -- changing what the orchestrator enforces.
The feed is exempt (GET, that exact path); the screens are not.
"""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy.compiler import compile_policy
from tests.agent_policy.fixtures import FORM_EXAMPLE, REGISTRY, SETTINGS

FEED = "/orchestrator/agent-policies/v2/live"
EXCERPT = "From 01/04/2026, refunds or credits above $500 need approval from the Finance Manager."
TO_AGENT = "Do not book a container or retry the tool call; it needs Finance approval."


def _doc():
    form = copy.deepcopy(FORM_EXAMPLE)
    form["messageForAgent"] = TO_AGENT
    form["source"]["excerpt"] = EXCERPT
    form["checked"] = {"by": "u1", "at": "2026-10-08T00:00:00Z"}
    return compile_policy(form, policy_key="FIN-0012", version=2, status="live",
                          settings=SETTINGS, never_suggest=False)


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


@pytest.fixture
def app_client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setenv("AGENT_POLICY_ORCHESTRATOR_KEY", "o1")
    monkeypatch.setattr(R, "load_registry", lambda conn=None: REGISTRY)
    monkeypatch.setattr(R, "load_settings", lambda conn=None: SETTINGS)
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: "Admin")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    monkeypatch.setattr(R.repo, "never_suggest_for", lambda conn, area: False)
    return TestClient(app)


def test_the_feed_arrives_byte_for_byte(app_client, monkeypatch):
    doc = _doc()
    assert R.contract.validate(doc, REGISTRY) == []
    assert doc["outputs"]["toAgent"]["reason"] == TO_AGENT and doc["source"]["excerpt"] == EXCERPT
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: [copy.deepcopy(doc)])
    r = app_client.get(FEED, headers={"X-Orchestrator-Key": "o1"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] == []
    assert json.dumps(body["policies"][0], sort_keys=True) == json.dumps(doc, sort_keys=True)


def test_a_condition_value_with_slashes_is_not_withheld(app_client, monkeypatch):
    doc = _doc()
    doc["trigger"]["condition"]["all"].append({"field": "agent.reason", "op": "ne", "value": "DL/2024/001"})
    assert R.contract.validate(doc, REGISTRY) == []
    monkeypatch.setattr(R.repo, "live_documents", lambda conn: [copy.deepcopy(doc)])
    body = app_client.get(FEED, headers={"X-Orchestrator-Key": "o1"}).json()
    assert "[withheld]" not in json.dumps(body)
    assert body["policies"][0]["trigger"]["condition"] == doc["trigger"]["condition"]


def test_feed_errors_are_still_handled(app_client):
    r = app_client.get(FEED)
    assert r.status_code == 401 and r.json() == {"detail": "not accepted"}


def test_the_screens_are_still_scrubbed(app_client):
    """Only the feed is exempt: the same excerpt in the Admin preview is still withheld."""
    form = copy.deepcopy(FORM_EXAMPLE)
    form["source"]["excerpt"] = EXCERPT
    form["hidden"]["inputs"].append({"name": "refunds in 30 days", "field": "agg.refunds_30d", "type": "number",
                                     "from": "total:refunds_30d", "showApprover": False, "sensitive": False})
    hdr = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Groups": json.dumps(["PROCWISE_ADMIN"])}
    r = app_client.post("/agent-policies/preview", json={"form": form}, headers=hdr)
    assert r.status_code == 200
    assert r.json()["compiled"]["source"]["excerpt"] == "[withheld]"
    assert r.json()["howEnforced"]["cantEnforceNames"] == ["refunds in 30 days"]
