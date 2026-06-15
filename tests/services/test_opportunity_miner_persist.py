"""The opportunity miner persists findings into proc.bp_opportunity directly
(via _output_db), and the scheduler exposes an opt-in mining job."""
from __future__ import annotations

import os
from datetime import timedelta


class _FakeCur:
    def execute(self, *a, **k):
        return None


class _FakeConn:
    def __init__(self):
        self.autocommit = True

    def cursor(self):
        return _FakeCur()

    def commit(self):
        pass

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Finding:
    def __init__(self, oid):
        self._oid = oid

    def as_dict(self):
        return {"opportunity_id": self._oid, "detector_type": "Maverick Spend Detection",
                "financial_impact_gbp": 100.0, "calculation_details": {}, "source_records": []}


def test_output_db_upserts_each_finding(monkeypatch):
    from src.agents.opportunity_miner_agent import OpportunityMinerAgent
    import src.services.db as db
    import src.services.opportunity_store as store

    seen = []
    monkeypatch.setattr(db, "get_conn", lambda: _FakeConn())
    monkeypatch.setattr(store, "upsert_opportunity", lambda cur, rec: seen.append(rec["opportunity_id"]))

    agent = object.__new__(OpportunityMinerAgent)  # no __init__: _output_db uses no instance state
    OpportunityMinerAgent._output_db(agent, [_Finding("a"), _Finding("b")])
    assert seen == ["a", "b"]


def test_output_db_is_non_fatal(monkeypatch):
    from src.agents.opportunity_miner_agent import OpportunityMinerAgent
    import src.services.db as db
    monkeypatch.setattr(db, "get_conn", lambda: (_ for _ in ()).throw(RuntimeError("db down")))
    agent = object.__new__(OpportunityMinerAgent)
    # must swallow the error — mining must never break on a persistence hiccup
    OpportunityMinerAgent._output_db(agent, [_Finding("a")])


def _bare_scheduler(orchestrator=None):
    from src.services.backend_scheduler import BackendScheduler
    s = BackendScheduler.__new__(BackendScheduler)
    s._orchestrator = orchestrator if orchestrator is not None else object()
    return s


_CHANGED = {"forward_linked": 2, "backward_linked": 0, "reconciled": 0,
            "propagated": 0, "metadata_filled": 0, "unassigned_review": 0,
            "conflicts_flagged": 0}
_NO_CHANGE = {**_CHANGED, "forward_linked": 0}


def _patch_assign(monkeypatch, result):
    import src.services.deal_assignment_service as das
    monkeypatch.setattr(das, "assign_deals", lambda: result)


def test_deal_assignment_chains_mining_when_deals_changed(monkeypatch):
    _patch_assign(monkeypatch, _CHANGED)
    monkeypatch.setenv("OPPORTUNITY_MINING_ENABLED", "1")
    s = _bare_scheduler()
    called = []
    monkeypatch.setattr(s, "_run_opportunity_mining", lambda: called.append(True))
    s._run_deal_assignment()
    assert called == [True]   # mining fired right after deals changed


def test_no_chain_when_no_deal_changes(monkeypatch):
    _patch_assign(monkeypatch, _NO_CHANGE)
    monkeypatch.setenv("OPPORTUNITY_MINING_ENABLED", "1")
    s = _bare_scheduler()
    called = []
    monkeypatch.setattr(s, "_run_opportunity_mining", lambda: called.append(True))
    s._run_deal_assignment()
    assert called == []   # nothing changed -> no heavy mining run


def test_no_chain_when_disabled(monkeypatch):
    _patch_assign(monkeypatch, _CHANGED)
    monkeypatch.setenv("OPPORTUNITY_MINING_ENABLED", "0")
    s = _bare_scheduler()
    called = []
    monkeypatch.setattr(s, "_run_opportunity_mining", lambda: called.append(True))
    s._run_deal_assignment()
    assert called == []


def test_no_chain_without_orchestrator(monkeypatch):
    _patch_assign(monkeypatch, _CHANGED)
    monkeypatch.setenv("OPPORTUNITY_MINING_ENABLED", "1")
    s = _bare_scheduler(orchestrator=None)
    s._orchestrator = None
    called = []
    monkeypatch.setattr(s, "_run_opportunity_mining", lambda: called.append(True))
    s._run_deal_assignment()
    assert called == []
