"""/compliance/po-variance is for the gateway only: its key and verified identity, nothing else."""
import json

import pytest
from fastapi.testclient import TestClient

KEY = "test-gateway-key"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", KEY)
    monkeypatch.setattr("api.routers.compliance.po_variance", lambda deal_id=None: {"cases": 3, "deal": deal_id})
    from api.main import app
    return TestClient(app)


def _headers(key=KEY, sub="u1"):
    return {"x-gateway-key": key, "x-user-sub": sub, "x-user-groups": json.dumps(["PROCWISE_VIEWER"])}


def test_no_key_is_refused(client):
    assert client.get("/compliance/po-variance").status_code == 401


def test_wrong_key_is_refused(client):
    assert client.get("/compliance/po-variance", headers=_headers(key="nope")).status_code == 401


def test_key_without_an_identity_is_refused(client):
    assert client.get("/compliance/po-variance", headers=_headers(sub="")).status_code == 401


def test_gateway_call_is_answered_and_passes_the_deal(client):
    r = client.get("/compliance/po-variance?deal_id=D9", headers=_headers())
    assert r.status_code == 200
    assert r.json() == {"cases": 3, "deal": "D9"}
