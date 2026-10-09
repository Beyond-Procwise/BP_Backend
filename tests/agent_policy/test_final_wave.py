"""Stage 3 final fix wave: the checks that need no database (T4, gate logging, D1, D2 through the app).

Live counterparts (I2, I3, I4, T6, D2/D3 values, unroutable notice) are in test_final_wave_live.py.
"""
import builtins
import json
import logging

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import approval_views as V
from services.agent_policy import approvals as A
from services.agent_policy import gate as G

SECRET = "GB29NWBK60161331926819"


# ------------------------------------------------------------------ T4: broken replay is not "missing"
def _import_raising(monkeypatch, exc):
    real = builtins.__import__

    def fake(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "services.agent_policy" and fromlist and "replay" in fromlist:
            raise exc
        return real(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", fake)


def test_a_missing_replay_module_is_reported_missing(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=A.__name__)
    _import_raising(monkeypatch, ModuleNotFoundError("No module named x", name="services.agent_policy.replay"))
    A._replay_after_commit(7, None)
    assert "is missing" in caplog.text


@pytest.mark.parametrize("exc", [ImportError("cannot import name 'build_tools'", name="orchestration"),
                                 ModuleNotFoundError("No module named 'orchestration'", name="orchestration")],
                         ids=["ImportError inside replay.py", "a dependency of replay.py is missing"])
def test_an_import_error_inside_replay_is_logged_with_its_type_not_as_missing(monkeypatch, caplog, exc):
    caplog.set_level(logging.DEBUG, logger=A.__name__)
    _import_raising(monkeypatch, exc)
    A._replay_after_commit(7, None)
    assert "is missing" not in caplog.text
    rec = [r for r in caplog.records if r.name == A.__name__][-1]
    assert type(exc).__name__ in rec.getMessage() and rec.exc_info and rec.exc_info[0] is type(exc)


# ------------------------------------------------------------------ gate: the type only, never the message
def test_store_unavailable_logs_the_exception_type_only(monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=G.__name__)

    def down():
        raise RuntimeError(f"duplicate key (iban)=({SECRET})")
    monkeypatch.setattr(G, "_load_policies", down)
    monkeypatch.setattr(G, "_connect", lambda: (_ for _ in ()).throw(RuntimeError("no db")))
    out = G.before_tool(tool_name="refund.issue", args={"iban": SECRET})
    assert out.allow is False and out.to_agent["reasonCode"] == "policy_check_unavailable"
    text = caplog.text + "".join(str(r.exc_info) + str(r.exc_text) for r in caplog.records)
    assert "RuntimeError" in caplog.text and SECRET not in text
    assert all(r.exc_info is None for r in caplog.records if r.name == G.__name__)


# ------------------------------------------------------------------ D1 / D2 through the whole app
HDR = {"X-Gateway-Key": "k1", "X-User-Sub": "u-admin", "X-User-Email": "admin@example.test",
       "X-User-Groups": json.dumps(["PROCWISE_ADMIN"])}


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


@pytest.fixture
def client(monkeypatch):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: "Admin")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    return TestClient(app)


def test_decider_put_answers_name_and_time_only_and_nothing_is_withheld(client, monkeypatch):
    monkeypatch.setattr(V, "upsert_decider", lambda conn, name, groups, emails, notes, actor: {
        "name": name, "groups": list(groups), "emails": list(emails), "notes": notes,
        "lastModifiedBy": actor, "lastModifiedAt": "2026-10-09T09:00:00+00:00"})
    r = client.put("/agent-policies/deciders/Finance Manager", headers=HDR,
                   json={"groups": ["PROCWISE_PROCUMENT_BUYER_ANALYST"], "emails": ["fm@acme.example"]})
    assert r.status_code == 200, r.text
    assert r.json() == {"name": "Finance Manager", "savedAt": "2026-10-09T09:00:00+00:00"}
    assert "[withheld]" not in r.text


def test_decide_answer_carries_the_group_through_the_scrubber(client, monkeypatch):
    monkeypatch.setattr(R.approval_views, "readable", lambda conn, did, p, is_admin: True)
    group = {"open": 1, "approved": 1, "rejected": 0, "total": 2}
    monkeypatch.setattr(R.approvals, "act", lambda conn, did, **kw: {
        "decisionId": did, "result": "approved", "verb": "approve", "level": 0, "levelName": "Finance Manager",
        "actionId": 1, "reason": None, "decidedBy": "u-admin", "decidedAt": "2026-10-09T09:00:00+00:00",
        "group": group})
    monkeypatch.setattr(R, "_replay_later", lambda did: None)
    r = client.post("/agent-policies/approvals/41/decide", json={"verb": "approve"}, headers=HDR)
    assert r.status_code == 200 and r.json()["group"] == group


def test_admins_read_the_administrators_notices(monkeypatch):
    class P:
        subject, email = "u-admin", "admin@example.test"
    monkeypatch.setattr(V.deciders, "load_map", lambda conn: {})
    assert A.ADMIN_RECIPIENT in V._recipients(object(), P(), is_admin=True)
    assert A.ADMIN_RECIPIENT not in V._recipients(object(), P(), is_admin=False)


# ------------------------------------------------------------------ stage 4 task 0, m1: the reserved decider name
@pytest.mark.parametrize("name", ["Administrators", "administrators", "ADMINISTRATORS"])
def test_the_administrators_recipient_cannot_be_mapped_as_a_decider(client, monkeypatch, name):
    """'Administrators' is the fixed recipient every Admin reads; mapping members to it would let
    non-Admins read unroutable notices. Refused 422 `reserved_name`, case-insensitively, trimmed."""
    saved = []
    monkeypatch.setattr(V, "upsert_decider", lambda *a, **kw: saved.append(a) or {})
    r = client.put(f"/agent-policies/deciders/{name}", headers=HDR, json={"groups": ["G1"]})
    assert r.status_code == 422, r.text
    assert [p["code"] for p in r.json()["problems"]] == ["reserved_name"]
    assert saved == []


def test_the_reserved_name_is_matched_after_trimming_and_before_other_checks():
    probs, _, _ = V.decider_problems("  administrators ", ["G1"], [], None)
    assert [p["code"] for p in probs] == ["reserved_name"] and probs[0]["field"] == "name"
    assert V.decider_problems("Administrators Assistant", ["G1"], [], None)[0] == []
