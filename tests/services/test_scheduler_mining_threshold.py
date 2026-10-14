"""The opportunity-mining threshold comes from policy, or mining does not run.

The scheduled job read opportunity_mining_min_impact through governed_limits
and then caught ValueError and used 100.0 -- so a policy value that could not
be read as a number quietly became a number carried in code. That threshold
decides which opportunities a buyer is told about; P9 moved it into policy
precisely so it would not be decided by a constant nobody can see.

A missing policy already refused (LimitUnavailable is not a ValueError). An
unusable one now refuses the same way: no mining, and a log line that says why.
"""

from __future__ import annotations

import logging

import pytest

from src.services import governed_limits as GL
from src.services.backend_scheduler import BackendScheduler


class _Engine:
    def __init__(self, rules):
        self._rules = rules

    def get_policy(self, slug):
        if slug != "autonomous_operation" or self._rules is None:
            return None
        return {"policyName": slug, "details": {"policy_identifier": slug,
                                                "rules": dict(self._rules)}}


class _Orchestrator:
    def __init__(self):
        self.calls = []

    def execute_workflow(self, name, payload, user_id=None):
        self.calls.append(payload)
        return {"status": "completed"}


def _scheduler(monkeypatch, rules):
    GL.reset_cache()
    monkeypatch.setattr(GL, "_engine", lambda: _Engine(rules))
    monkeypatch.delenv("OPPORTUNITY_MINING_MIN_IMPACT", raising=False)
    sched = BackendScheduler.__new__(BackendScheduler)
    sched._orchestrator = _Orchestrator()
    return sched


def test_mining_runs_at_the_governed_threshold(monkeypatch):
    sched = _scheduler(monkeypatch, {"opportunity_mining_min_impact": 250})

    sched._run_opportunity_mining()

    assert [c["min_financial_impact"] for c in sched._orchestrator.calls] == [250.0]


@pytest.mark.parametrize("rules", [
    {"opportunity_mining_min_impact": "a lot"},   # stated, but not a number
    {},                                            # not stated
    None,                                          # no policy at all
], ids=["unusable", "unstated", "no-policy"])
def test_an_ungoverned_threshold_means_no_mining(monkeypatch, caplog, rules):
    sched = _scheduler(monkeypatch, rules)

    with caplog.at_level(logging.ERROR):
        sched._run_opportunity_mining()

    assert sched._orchestrator.calls == [], (
        f"mining ran at {sched._orchestrator.calls[0]['min_financial_impact']!r} "
        f"with no governed threshold")
    assert caplog.records, "mining was skipped without a word"
