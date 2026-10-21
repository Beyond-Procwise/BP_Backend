"""Who may activate or retire an agent policy is the ``agent_policy_use`` onboarding setting, not a
role written into the router. Drafting stays with Buyers whatever the setting says."""
import json

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services import permission_settings as ps

HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u1", "X-User-Groups": json.dumps(["x"])}
SAVE = "/agent-policies/FIN-0012/versions"
RETIRE = "/agent-policies/FIN-0012/retire"
OK = {"policyKey": "FIN-0012", "version": 2}


class _Conn:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


@pytest.fixture
def st():
    return {"role": "Approver", "setting": "Admin", "audits": [], "saved": []}


@pytest.fixture
def client(monkeypatch, st):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_conn", _Conn)
    monkeypatch.setattr(R, "_role_of", lambda p: st["role"])
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: st["audits"].append(kw))
    monkeypatch.setattr(R, "_remember", lambda p: None)
    monkeypatch.setattr(R, "_detect_conflicts", lambda k: None)
    monkeypatch.setattr(R, "_close_moot", lambda k: None, raising=False)
    monkeypatch.setattr(R.live_policies, "invalidate", lambda: None)
    monkeypatch.setattr(ps, "minimum_role", lambda key, conn=None, fallback="Admin": st["setting"])

    def save_version(conn, key, form, **kw):
        st["saved"].append(kw["intent"])
        return dict(OK)
    monkeypatch.setattr(R.repo, "save_version", save_version)
    monkeypatch.setattr(R.repo, "retire", lambda *a, **k: dict(OK))
    return TestClient(app)


def _save(c, intent):
    return c.post(SAVE, headers=HDR, json={"form": {}, "baseVersion": 1, "intent": intent})


def test_an_approver_is_refused_activation_when_the_setting_says_admin(client, st):
    r = _save(client, "activate")
    assert r.status_code == 403 and st["saved"] == []
    assert st["audits"][-1]["status"] == "denied" and st["audits"][-1]["details"]["minimum"] == "Admin"


def test_an_admin_may_activate_when_the_setting_says_admin(client, st):
    st["role"] = "Admin"
    assert _save(client, "activate").status_code == 200 and st["saved"] == ["activate"]


def test_lowering_the_setting_lets_an_approver_activate(client, st):
    st["setting"] = "Approver"
    assert _save(client, "activate").status_code == 200


def test_drafting_needs_only_a_buyer_whatever_the_setting(client, st):
    st["role"] = "Buyer"
    assert _save(client, "draft").status_code == 200 and st["saved"] == ["draft"]


def test_retiring_follows_the_same_setting(client, st):
    assert client.post(RETIRE, headers=HDR, json={"baseVersion": 1}).status_code == 403
    st["role"] = "Admin"
    assert client.post(RETIRE, headers=HDR, json={"baseVersion": 1}).status_code == 200


def test_a_setting_naming_no_known_role_means_admin(client, st):
    st["setting"] = "Wizard"
    assert _save(client, "activate").status_code == 403
    st["role"] = "Admin"
    assert _save(client, "activate").status_code == 200


def test_an_unreadable_setting_falls_back_to_admin(monkeypatch):
    monkeypatch.setattr(ps, "current", lambda conn=None: (_ for _ in ()).throw(RuntimeError("db down")))
    assert ps.minimum_role("agent_policy_use") == "Admin"
