"""conflict_cases without a database: lock first, best-effort hooks, and who calls detection."""
import json
import logging
from datetime import datetime, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.agent_policy import conflict_cases as CC
from services.agent_policy import extraction_run as ER
from tests.agent_policy.test_conflict_detect import make

NOW = datetime(2026, 10, 9, 9, 0, tzinfo=timezone.utc)


class _Cur:
    """Records every statement; answers 'covered' to the first dedup check."""

    def __init__(self, covered=True):
        self.sql = []
        self.covered = covered

    def execute(self, sql, params=None):
        self.sql.append(" ".join(sql.split()))

    def fetchone(self):
        return (1,) if self.covered else None


def _pair():
    a = make("FIN-0012", cond={"all": [{"field": "args.amount", "op": "gt", "value": 500}]})
    b = make("CUS-0004", outcome="block", source="Customer Refund Standard",
             cond={"all": [{"field": "args.amount", "op": "gt", "value": 10000}]})
    return a, b


def test_advisory_lock_is_taken_before_any_check():
    a, b = _pair()
    cur = _Cur(covered=True)
    assert CC.raise_policy_case(cur, a, b, {"args.amount": 10001}, raised_by="save", now=NOW, mapping={}) is None
    assert cur.sql[0].startswith("SELECT pg_advisory_xact_lock(hashtext(")
    assert "bp_agent_policy_conflict" in cur.sql[1] and not any("INSERT" in s for s in cur.sql)


def test_unreadable_condition_skips_the_pair_and_counts_an_error():
    a, b = _pair()
    bad = make("LEG-0001", outcome="block", source="Legal", cond={"all": [{"field": "args.amount", "op": "??", "value": 1}]})
    stats = {"pairs": 0, "raised": 0, "errors": 0}
    found = CC._pairs(a, [], [bad, b], stats)
    assert [k for k, _o, _w in found] == ["CUS-0004|FIN-0012"] and stats["errors"] == 1


def test_pairs_are_in_one_global_lock_order_latest_version_first():
    a, b = _pair()
    b2 = dict(b, version=2)
    c = make("AAA-0001", outcome="block", source="Other", cond={"all": [{"field": "args.amount", "op": "gt", "value": 1}]})
    stats = {"pairs": 0, "raised": 0, "errors": 0}
    found = CC._pairs(a, [], [b, c, b2], stats)
    assert [(k, o["version"]) for k, o, _w in found] == [("AAA-0001|FIN-0012", 1), ("CUS-0004|FIN-0012", 2),
                                                         ("CUS-0004|FIN-0012", 1)]


@pytest.mark.conflict_detection
def test_after_save_never_raises(monkeypatch, caplog):
    def boom(conn, key, **kw):
        raise RuntimeError("db down")
    monkeypatch.setattr(CC, "detect_for", boom)
    with caplog.at_level(logging.ERROR, logger=CC.__name__):
        assert CC.after_save(object(), "FIN-0012") is None
    assert "conflict detection failed for FIN-0012: RuntimeError" in caplog.text


def test_detect_all_never_raises(monkeypatch):
    class _Broken:
        def cursor(self):
            raise RuntimeError("down")
    assert CC.detect_all(_Broken(), now=NOW) == {"pairs": 0, "raised": 0, "errors": 1, "capped": False}


def test_cap_is_a_company_setting_default_25():
    from services.agent_policy import settings as S
    assert S.DEFAULTS["conflict_cases_per_run"] == 25
    assert CC.cap_of(S.merge(None)) == 25 and CC.cap_of(S.merge({"conflict_cases_per_run": 2})) == 2
    assert CC.cap_of({"conflict_cases_per_run": "junk"}) == 25 and CC.cap_of({"conflict_cases_per_run": 0}) == 25


# ---------------------------------------------------------------- extraction
def _run_decide(monkeypatch, decision):
    calls = []
    monkeypatch.setattr(CC, "after_save", lambda conn, key: calls.append(key))
    monkeypatch.setattr(ER.repo, "create_draft", lambda conn, form, **kw: {"policyKey": "GEN-0101", "version": 1})
    monkeypatch.setattr(ER.repo, "save_version", lambda conn, key, form, **kw: {"policyKey": key, "version": 4})
    monkeypatch.setattr(ER.repo, "update_source", lambda *a, **k: None)
    emitted = []
    form = {"name": "x", "source": {"reference": "1.1"}, "hidden": {}}
    d = {"decision": decision, "policyKey": "GEN-0100"}
    by_key = {"GEN-0100": {"latestVersion": 3, "reference": "1.1", "split": None, "form": {"name": "x"}}}
    monkeypatch.setattr(ER, "_replaces_edits", lambda old: False)
    monkeypatch.setattr(ER, "_carry_person_fields", lambda form, old: None)
    monkeypatch.setattr(ER.matching, "split_key", lambda form: None)
    counts = {"new": 0, "changed": 0, "unchanged": 0, "errors": 0}
    ER._decide(object(), lambda *a, **k: emitted.append(a), form, d, by_key, counts, actor="t", title="T",
               version=1, document_id=None, text="", at={})
    return calls, counts


@pytest.mark.parametrize("decision,expected", [("new", ["GEN-0101"]), ("changed", ["GEN-0100"]), ("unchanged", [])])
def test_extraction_save_triggers_detection(monkeypatch, decision, expected):
    calls, counts = _run_decide(monkeypatch, decision)
    assert calls == expected and counts[decision] == 1


def test_extraction_failed_save_triggers_no_detection(monkeypatch):
    calls = []
    monkeypatch.setattr(CC, "after_save", lambda conn, key: calls.append(key))

    def stale(*a, **k):
        raise ER.repo.StaleVersion("x")
    monkeypatch.setattr(ER.repo, "create_draft", stale)
    counts = {"new": 0, "errors": 0}
    ER._decide(object(), lambda *a, **k: None, {"name": "x", "source": {}, "hidden": {}},
               {"decision": "new", "policyKey": None}, {}, counts, actor="t", title="T", version=1,
               document_id=None, text="", at={})
    assert calls == [] and counts["errors"] == 1


# ---------------------------------------------------------------- router
@pytest.fixture
def client(monkeypatch):
    from api.routers import agent_policies as R
    from tests.agent_policy.test_router import GOOD, _FakeConnCtx
    monkeypatch.setenv("AGENT_POLICY_GATEWAY_KEY", "k1")
    monkeypatch.setattr(R, "_role_of", lambda principal: "Admin")
    monkeypatch.setattr(R.agent_actions, "record_action_or_fail", lambda **kw: None)
    state = {"open": 0, "log": []}

    class _Ctx(_FakeConnCtx):
        def __enter__(self):
            state["open"] += 1
            return object()

        def __exit__(self, *a):
            state["open"] -= 1
            return False

    monkeypatch.setattr(R, "_conn", _Ctx)
    monkeypatch.setattr(R.live_policies, "invalidate", lambda: state["log"].append(("invalidate", state["open"])))
    monkeypatch.setattr(CC, "after_save", lambda conn, key: state["log"].append(("detect", key, state["open"])))
    app = FastAPI(); app.include_router(R.router)
    c = TestClient(app); c.state, c.R, c.hdr = state, R, GOOD
    return c


def test_create_runs_detection_after_the_save_on_a_fresh_connection(client, monkeypatch):
    monkeypatch.setattr(client.R.repo, "create_draft", lambda conn, form, actor: {"policyKey": "GEN-0001", "version": 1})
    r = client.post("/agent-policies", json={"form": {"name": "x"}}, headers=client.hdr)
    assert r.status_code == 200 and r.json() == {"policyKey": "GEN-0001", "version": 1}
    assert client.state["log"] == [("detect", "GEN-0001", 1)]     # one fresh connection, the save's is closed


def test_save_runs_detection_after_invalidate(client, monkeypatch):
    monkeypatch.setattr(client.R.repo, "save_version", lambda *a, **k: {"policyKey": "GEN-0001", "version": 2})
    body = {"form": {"name": "x"}, "baseVersion": 1, "intent": "draft", "changeNote": ""}
    assert client.post("/agent-policies/GEN-0001/versions", json=body, headers=client.hdr).status_code == 200
    assert client.state["log"] == [("invalidate", 0), ("detect", "GEN-0001", 1)]


@pytest.mark.parametrize("exc,status", [("StaleVersion", 409), ("NotFound", 404), ("NotReady", 422)])
def test_failed_save_runs_no_detection(client, monkeypatch, exc, status):
    def refuse(*a, **k):
        E = getattr(client.R.repo, exc)
        raise E([{"field": "owner"}]) if exc == "NotReady" else E("x")
    monkeypatch.setattr(client.R.repo, "save_version", refuse)
    body = {"form": {"name": "x"}, "baseVersion": 1, "intent": "draft", "changeNote": ""}
    assert client.post("/agent-policies/GEN-0001/versions", json=body, headers=client.hdr).status_code == status
    assert client.state["log"] == [("invalidate", 0)]


@pytest.mark.conflict_detection
def test_detection_that_cannot_connect_never_fails_the_save(client, monkeypatch):
    monkeypatch.setattr(client.R.repo, "create_draft", lambda conn, form, actor: {"policyKey": "GEN-0001", "version": 1})
    calls = {"n": 0}

    class _Once:
        def __enter__(self):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("no connection")
            return object()

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(client.R, "_conn", _Once)
    r = client.post("/agent-policies", json={"form": {"name": "x"}}, headers=client.hdr)
    assert r.status_code == 200 and r.json()["policyKey"] == "GEN-0001"
