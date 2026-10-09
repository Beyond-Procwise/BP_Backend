"""The decision engine decides a live clash between agent policies on precedent, or escalates.

Rule 1: it decides only from what it looks up (the governed N and the clash's own settled cases).
Rule 2: anything short of N of N agreeing decisions by people, at identical versions, escalates.
"""
import json
from datetime import datetime, timezone

import pytest

from engines import decision_engine as DE
from services.agent_policy import conflict_engine as CE
from src.services import governed_limits as GL
from tests.agent_policy.fixtures import precedent_n

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
AT = datetime(2026, 10, 9, 11, 0, tzinfo=timezone.utc)


def _hit(key, version, outcome="approve"):
    return {"id": key, "version": version, "outcome": outcome, "policy": {"id": key, "version": version}}


A, B, C = _hit("TST-0001", 3), _hit("TST-0002", 1), _hit("TST-0003", 1)


def _lc(kind="human", involved=(A, B), required=None):
    involved = list(involved)
    return CE.LiveConflict(kind, involved, [("TST-0001", "TST-0002")],
                           list(required if required is not None else involved), {h["id"] for h in involved})


class _Cur:
    def __init__(self, rows=(), fail=False):
        self.rows, self.fail, self.sql = list(rows), fail, []

    def execute(self, sql, params=None):
        self.sql.append((" ".join(sql.split()), params))
        if self.fail and "bp_agent_policy_conflict" in sql:
            raise RuntimeError("lookup down")

    def fetchall(self):
        return list(self.rows)


def _n(monkeypatch, n):
    monkeypatch.setattr(DE, "_precedent_count", lambda: n)


def test_resolves_when_the_last_n_person_decisions_agree(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.decision, d.subject_type, d.subject_id) == (DE.RESOLVED, "approve", "live_conflict",
                                                                       "TST-0001|TST-0002")
    assert [e.reference for e in d.evidence] == ["pc_11", "pc_10"]
    assert d.evidence[0].to_dict() == {"fact": "precedent", "source": "proc.bp_agent_policy_conflict",
                                       "reference": "pc_11",
                                       "value": {"decision_id": 11, "outcome": "approve", "actioned_by": "sub-b",
                                                 "actioned_at": AT.isoformat()}}
    assert d.facts["citedCases"] == ["pc_11", "pc_10"] and d.facts["precedentCount"] == 2
    statements = [s for s, _ in cur.sql]
    assert statements[0] == "SAVEPOINT live_conflict_precedent"
    assert statements[-1] == "RELEASE SAVEPOINT live_conflict_precedent"
    [(_sql, params)] = [x for x in cur.sql if "bp_agent_policy_conflict" in x[0]]
    assert params == ("TST-0001|TST-0002", json.dumps({"TST-0001": 3, "TST-0002": 1}, sort_keys=True), 2)


def test_the_lookup_counts_only_settled_person_decisions_at_identical_versions():
    sql = " ".join(DE.PRECEDENT_SQL.split())
    for clause in ("c.kind = 'live'", "c.pair_key = %s", "NOT c.is_open", "c.by_person",
                   "c.policy_versions = %s::jsonb", "ORDER BY c.decided_at DESC, c.decision_id DESC", "LIMIT %s"):
        assert clause in sql, clause


def test_rejects_on_precedent_too(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur([(11, "reject", "sub-b", AT), (10, "reject", "sub-a", AT)]), _lc(),
                                ctx={}, now=NOW)
    assert (d.resolution, d.decision) == (DE.RESOLVED, "reject")


def test_fewer_than_n_escalates(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur([(10, "approve", "sub-a", AT)]), _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "only 1 of 2 decisions by people on this exact clash")


def test_disagreement_escalates(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur([(11, "approve", "sub-b", AT), (10, "reject", "sub-a", AT)]), _lc(),
                                ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "decisions disagree")


@pytest.mark.parametrize("n", [0, None, -1])
def test_zero_or_null_switches_it_off(monkeypatch, n):
    _n(monkeypatch, n)
    cur = _Cur()
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent is switched off") and cur.sql == []


def test_an_unreadable_limit_escalates_with_a_warning(monkeypatch, caplog):
    def missing():
        raise GL.LimitUnavailable("no active agent_policy_conflicts policy")
    monkeypatch.setattr(DE, "_precedent_count", missing)
    cur = _Cur()
    with caplog.at_level("WARNING", logger=DE.__name__):
        d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent limit unavailable") and cur.sql == []
    assert "precedent limit unavailable, the clash TST-0001|TST-0002 goes to people" in caplog.text


def test_a_failed_lookup_rolls_back_to_its_savepoint_and_escalates(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur(fail=True)
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent lookup failed")
    assert [s for s, _ in cur.sql if "SAVEPOINT" in s] == [
        "SAVEPOINT live_conflict_precedent", "ROLLBACK TO SAVEPOINT live_conflict_precedent",
        "RELEASE SAVEPOINT live_conflict_precedent"]


def test_an_approval_outside_the_clash_escalates(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    d = DE.decide_live_conflict(cur, _lc(required=[A, B, C]), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "TST-0003 also needs approval and is not part of this clash")
    assert cur.sql == []


def test_only_a_human_clash_is_considered(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur(), _lc(kind="auto"), ctx={}, now=NOW)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "only a clash that needs people can be decided on precedent")


def test_the_real_count_is_the_governed_fresh_value(monkeypatch):
    precedent_n(monkeypatch, 4)
    assert DE._precedent_count() == 4
    precedent_n(monkeypatch, None, missing=True)
    with pytest.raises(GL.LimitUnavailable):
        DE._precedent_count()
