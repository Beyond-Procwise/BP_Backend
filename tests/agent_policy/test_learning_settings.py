"""The learning thresholds as settings an administrator reads and changes (stage 5b, B2), through
the WHOLE app with no database: the role floor, the closed body and its ranges, the answer's shape
after the output filter, and the scan switch. The real row (create, lock, merge, audit in one
transaction) is test_learning_settings_live's.
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.routers import agent_policies as R
from services.agent_policy import settings as S
from services.backend_scheduler import BackendScheduler

URL = "/agent-policies/settings/learning"
DEFAULTS = {"min_decisions": 30, "min_days": 30, "min_approvers": 3, "wilson_lower": 0.85,
            "median_seconds_floor": 30, "not_yet_more": 30, "dismiss_more": 30}
GOOD = {"min_decisions": 40, "min_days": 45, "min_approvers": 4, "wilson_lower": 0.9,
        "median_seconds_floor": 20, "not_yet_more": 50, "dismiss_more": 60}
WHOLE = {"min_decisions": (10, 1000), "min_days": (1, 365), "min_approvers": (2, 50),
         "median_seconds_floor": (0, 3600), "not_yet_more": (1, 1000), "dismiss_more": (1, 1000)}


def _hdr(sub="u1"):
    return {"X-Gateway-Key": "k1", "X-User-Sub": sub, "X-User-Email": f"{sub}@example.test",
            "X-User-Groups": json.dumps([])}


class _FakeConnCtx:
    def __enter__(self): return object()
    def __exit__(self, *a): return False


@pytest.fixture
def state():
    return {"role": "Admin", "audits": [], "stored": {"learning": dict(DEFAULTS)}, "saves": []}


@pytest.fixture
def client(monkeypatch, state):
    from api.main import app
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.delenv("AGENT_POLICY_LEARNING_SCAN", raising=False)
    monkeypatch.setattr(R, "_conn", _FakeConnCtx)
    monkeypatch.setattr(R, "_role_of", lambda principal: state["role"])
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: state["audits"].append(kw))
    monkeypatch.setattr(S, "load_settings", lambda conn=None: S.merge(state["stored"]))

    def save_learning(conn, values, *, actor, record):
        before = {k: S.merge(state["stored"])["learning"].get(k) for k in S.LEARNING_KEYS}
        state["stored"] = {**state["stored"], "learning": dict(values)}
        state["saves"].append((dict(values), actor))
        record(before, dict(values))
        return before, dict(values)

    monkeypatch.setattr(S, "save_learning", save_learning)
    return TestClient(app)


def test_the_seven_ruled_defaults_are_the_ones_in_the_settings_module():
    assert S.LEARNING_KEYS == tuple(DEFAULTS) and S.DEFAULTS["learning"] == DEFAULTS


# ------------------------------------------------------------------ reading
def test_an_admin_reads_the_thresholds_the_defaults_and_the_scan_switch(client, state):
    state["stored"] = {"learning": {**GOOD, "something_else": 9}, "response_time": "PT8H"}
    r = client.get(URL, headers=_hdr())
    assert r.status_code == 200, r.text
    # exactly this, after the output filter (this read is NOT exempt): numbers and booleans survive
    assert r.json() == {"learning": GOOD, "defaults": DEFAULTS, "scan": {"enabled": False}}
    assert state["audits"] == [], "a read is not audited"


@pytest.mark.parametrize("role", ["Viewer", "Buyer", "Approver", "Nobody"])
def test_reading_and_changing_need_the_admin_role(client, state, role):
    state["role"] = role
    assert client.get(URL, headers=_hdr()).status_code == 403
    r = client.put(URL, json=GOOD, headers=_hdr())
    assert r.status_code == 403 and state["saves"] == []
    assert [(a["action_type"], a["phase"], a["status"]) for a in state["audits"]] == \
        [("agent_policy.admin", "authorize", "denied")]


def test_no_gateway_key_no_answer(client):
    assert client.get(URL, headers={**_hdr(), "X-Gateway-Key": "wrong"}).status_code == 401
    assert client.put(URL, json=GOOD, headers={**_hdr(), "X-Gateway-Key": "wrong"}).status_code == 401


@pytest.mark.parametrize("value,enabled", [(None, False), ("", False), ("off", False), ("0", False), ("yes", False),
                                           ("on", True), (" On ", True), ("1", True), ("TRUE", True)])
def test_the_scan_switch_is_the_schedulers_own_rule(client, monkeypatch, value, enabled):
    if value is not None:
        monkeypatch.setenv("AGENT_POLICY_LEARNING_SCAN", value)
    assert S.learning_scan_enabled() is enabled
    assert client.get(URL, headers=_hdr()).json()["scan"] == {"enabled": enabled}

    class _Stub:                                             # the scheduler registers on the same answer
        AGENT_POLICY_LEARNING_SCAN_JOB_NAME = BackendScheduler.AGENT_POLICY_LEARNING_SCAN_JOB_NAME
        AGENT_POLICY_LEARNING_SCAN_LANE = BackendScheduler.AGENT_POLICY_LEARNING_SCAN_LANE
        _register_agent_policy_learning_scan_job = BackendScheduler._register_agent_policy_learning_scan_job
        _run_agent_policy_learning_scan = BackendScheduler._run_agent_policy_learning_scan
        _jobs: dict = {}

        def register_job(self, name, *a, **kw):
            self._jobs = {name: True}

    stub = _Stub()
    stub._register_agent_policy_learning_scan_job()
    assert bool(stub._jobs) is enabled


def test_the_scheduler_asks_the_shared_rule(monkeypatch):
    """Not a second copy of the rule: switch the shared function and the scheduler follows it."""
    monkeypatch.setenv("AGENT_POLICY_LEARNING_SCAN", "off")
    monkeypatch.setattr(S, "learning_scan_enabled", lambda: True)
    jobs = {}

    class _Stub:
        AGENT_POLICY_LEARNING_SCAN_JOB_NAME = "j"
        AGENT_POLICY_LEARNING_SCAN_LANE = "l"
        _run_agent_policy_learning_scan = None
        _jobs = jobs
        _register = BackendScheduler._register_agent_policy_learning_scan_job

        def register_job(self, name, *a, **kw):
            jobs[name] = True

    _Stub()._register()
    assert jobs == {"j": True}


# ------------------------------------------------------------------ changing
def test_an_admin_changes_the_thresholds(client, state):
    r = client.put(URL, json=GOOD, headers=_hdr("admin-1"))
    assert r.status_code == 200, r.text
    assert r.json() == {"learning": GOOD, "defaults": DEFAULTS, "scan": {"enabled": False}}, "after the output filter"
    assert state["saves"] == [(GOOD, "admin-1")]
    assert [(a["action_type"], a["phase"], a["status"]) for a in state["audits"]] == [
        ("agent_policy.admin", "authorize", "allowed"), ("agent_policy.admin", "act", "done")]
    assert state["audits"][0]["details"]["requested"] == GOOD and state["audits"][0]["details"]["minimum"] == "Admin"
    done = state["audits"][1]["details"]
    assert (done["before"], done["after"], done["principal"]) == (DEFAULTS, GOOD, "admin-1")
    assert client.get(URL, headers=_hdr()).json()["learning"] == GOOD


@pytest.mark.parametrize("key", sorted(WHOLE))
def test_each_whole_number_has_its_range(client, state, key):
    low, high = WHOLE[key]
    message = f"Enter a whole number from {low} to {high}."
    for good in (low, high):
        assert client.put(URL, json={**GOOD, key: good}, headers=_hdr()).status_code == 200, (key, good)
    saved = len(state["saves"])
    for bad in (low - 1, high + 1, True, False, None, str(low), float(low), low + 0.5, [low], {"v": low}):
        r = client.put(URL, json={**GOOD, key: bad}, headers=_hdr())
        assert r.status_code == 422, (key, bad, r.text)
        assert r.json() == {"problems": [{"field": key, "message": message}]}, (key, bad)
    missing = {k: v for k, v in GOOD.items() if k != key}
    r = client.put(URL, json=missing, headers=_hdr())
    assert r.status_code == 422 and r.json() == {"problems": [{"field": key, "message": message}]}
    assert len(state["saves"]) == saved, "a refused body saves nothing"


def test_the_confidence_is_a_number_from_a_half_to_099(client, state):
    message = "Enter a number from 0.5 to 0.99."
    for good in (0.5, 0.99, 0.75):
        assert client.put(URL, json={**GOOD, "wilson_lower": good}, headers=_hdr()).status_code == 200, good
    saved = len(state["saves"])
    for bad in (0.49, 0.995, 1, 0, 1.0, True, False, None, "0.85", [0.85]):
        r = client.put(URL, json={**GOOD, "wilson_lower": bad}, headers=_hdr())
        assert r.status_code == 422, (bad, r.text)
        assert r.json() == {"problems": [{"field": "wilson_lower", "message": message}]}, bad
    r = client.put(URL, content=json.dumps({**GOOD, "wilson_lower": float("nan")}), headers=_hdr())
    assert r.status_code == 422, "not a number at all"
    assert len(state["saves"]) == saved


def test_the_body_is_exactly_the_seven_keys_and_every_problem_is_told_at_once(client, state):
    r = client.put(URL, json={**GOOD, "min_days": 0, "wilson_lower": 2, "scan": True, "response_time": "PT1S"},
                   headers=_hdr())
    assert r.status_code == 422
    assert r.json() == {"problems": [
        {"field": "min_days", "message": "Enter a whole number from 1 to 365."},
        {"field": "wilson_lower", "message": "Enter a number from 0.5 to 0.99."},
        {"field": "response_time", "message": "This is not a learning threshold."},
        {"field": "scan", "message": "This is not a learning threshold."}]}
    assert state["saves"] == []
    assert [(a["phase"], a["status"]) for a in state["audits"]] == [("authorize", "allowed"), ("act", "refused")]
    assert state["audits"][1]["details"]["fields"] == ["min_days", "wilson_lower", "response_time", "scan"]
    for not_an_object in ([1, 2], "x", 3, None):
        assert client.put(URL, json=not_an_object, headers=_hdr()).status_code == 422
    assert client.put(URL, json={}, headers=_hdr()).json()["problems"][0]["field"] == "min_decisions"
    assert len(client.put(URL, json={}, headers=_hdr()).json()["problems"]) == 7
    assert state["saves"] == []


def test_a_save_that_fails_is_a_plain_error_and_is_audited(client, state, monkeypatch):
    def boom(conn, values, *, actor, record):
        raise RuntimeError("TST the row could not be locked")
    monkeypatch.setattr(S, "save_learning", boom)
    r = client.put(URL, json=GOOD, headers=_hdr())
    assert r.status_code == 500 and "TST" not in r.text and "RuntimeError" not in r.text
    assert [(a["phase"], a["status"]) for a in state["audits"]] == [("authorize", "allowed"), ("act", "error")]


def test_the_routes_are_declared_before_the_policy_key_routes():
    paths = [r.path for r in R.router.routes]
    first_key = min(i for i, p in enumerate(paths) if p.startswith("/agent-policies/{key}"))
    assert all(i < first_key for i, p in enumerate(paths) if p == "/agent-policies/settings/learning")
    assert paths.count("/agent-policies/settings/learning") == 2
