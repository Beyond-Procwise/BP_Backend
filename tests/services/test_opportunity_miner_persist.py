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


def test_opportunity_mining_job_opt_in(monkeypatch):
    import threading
    from src.services.backend_scheduler import BackendScheduler

    def _fresh():
        s = BackendScheduler.__new__(BackendScheduler)
        s._jobs = {}
        s._lock = threading.Lock()
        s._orchestrator = object()  # present
        return s

    monkeypatch.setenv("OPPORTUNITY_MINING_ENABLED", "0")
    s = _fresh()
    s._register_opportunity_mining_job()
    assert BackendScheduler.OPPORTUNITY_MINING_JOB_NAME not in s._jobs   # off by default

    monkeypatch.setenv("OPPORTUNITY_MINING_ENABLED", "1")
    s = _fresh()
    s._register_opportunity_mining_job()
    assert BackendScheduler.OPPORTUNITY_MINING_JOB_NAME in s._jobs       # opt-in registers


def test_opportunity_mining_job_skips_without_orchestrator(monkeypatch):
    import threading
    from src.services.backend_scheduler import BackendScheduler
    monkeypatch.setenv("OPPORTUNITY_MINING_ENABLED", "1")
    s = BackendScheduler.__new__(BackendScheduler)
    s._jobs = {}
    s._lock = threading.Lock()
    s._orchestrator = None
    s._register_opportunity_mining_job()
    assert BackendScheduler.OPPORTUNITY_MINING_JOB_NAME not in s._jobs
