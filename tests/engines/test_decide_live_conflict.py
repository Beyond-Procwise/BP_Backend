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
OTHER = "sub-requester"                 # decided none of the cited cases


def _hit(key, version, outcome="approve"):
    return {"id": key, "version": version, "outcome": outcome, "policy": {"id": key, "version": version}}


A, B, C = _hit("TST-0001", 3), _hit("TST-0002", 1), _hit("TST-0003", 1)


def _lc(kind="human", involved=(A, B), required=None):
    involved = list(involved)
    return CE.LiveConflict(kind, involved, [("TST-0001", "TST-0002")],
                           list(required if required is not None else involved), {h["id"] for h in involved})


class _Cur:
    """The precedent lookup answers `rows`; the cited cases' member approvers answer `approvers`."""
    def __init__(self, rows=(), fail=False, approvers=(), fail_approvers=False):
        self.rows, self.fail, self.sql = list(rows), fail, []
        self.approvers, self.fail_approvers = [(a,) for a in approvers], fail_approvers
        self._last = ""

    def execute(self, sql, params=None):
        self.sql.append((" ".join(sql.split()), params))
        self._last = sql
        if self.fail and "bp_agent_policy_conflict" in sql:
            raise RuntimeError("lookup down")
        if self.fail_approvers and "memberCases" in sql:
            raise RuntimeError("member lookup down")

    def fetchall(self):
        return list(self.approvers if "memberCases" in self._last else self.rows)


def _n(monkeypatch, n):
    monkeypatch.setattr(DE, "_precedent_count", lambda: n)


def test_resolves_when_the_last_n_person_decisions_agree(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester=OTHER)
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
    assert statements[2] == "RELEASE SAVEPOINT live_conflict_precedent"
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
                                ctx={}, now=NOW, requester=OTHER)
    assert (d.resolution, d.decision) == (DE.RESOLVED, "reject")


def test_fewer_than_n_escalates(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur([(10, "approve", "sub-a", AT)]), _lc(), ctx={}, now=NOW, requester=OTHER)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "only 1 of 2 decisions by people on this exact clash")


def test_disagreement_escalates(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur([(11, "approve", "sub-b", AT), (10, "reject", "sub-a", AT)]), _lc(),
                                ctx={}, now=NOW, requester=OTHER)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "decisions disagree")


@pytest.mark.parametrize("n", [0, None, -1])
def test_zero_or_null_switches_it_off(monkeypatch, n):
    _n(monkeypatch, n)
    cur = _Cur()
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester=OTHER)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent is switched off") and cur.sql == []


def test_an_unreadable_limit_escalates_with_a_warning(monkeypatch, caplog):
    def missing():
        raise GL.LimitUnavailable("no active agent_policy_conflicts policy")
    monkeypatch.setattr(DE, "_precedent_count", missing)
    cur = _Cur()
    with caplog.at_level("WARNING", logger=DE.__name__):
        d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester=OTHER)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent limit unavailable") and cur.sql == []
    assert "precedent limit unavailable, the clash TST-0001|TST-0002 goes to people" in caplog.text


def test_a_failed_lookup_rolls_back_to_its_savepoint_and_escalates(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur(fail=True)
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester=OTHER)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent lookup failed")
    assert [s for s, _ in cur.sql if "SAVEPOINT" in s] == [
        "SAVEPOINT live_conflict_precedent", "ROLLBACK TO SAVEPOINT live_conflict_precedent",
        "RELEASE SAVEPOINT live_conflict_precedent"]


def test_an_approval_outside_the_clash_escalates(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    d = DE.decide_live_conflict(cur, _lc(required=[A, B, C]), ctx={}, now=NOW, requester=OTHER)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "TST-0003 also needs approval and is not part of this clash")
    assert cur.sql == []


def test_only_a_human_clash_is_considered(monkeypatch):
    _n(monkeypatch, 2)
    d = DE.decide_live_conflict(_Cur(), _lc(kind="auto"), ctx={}, now=NOW, requester=OTHER)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "only a clash that needs people can be decided on precedent")


def test_the_real_count_is_the_governed_fresh_value(monkeypatch):
    precedent_n(monkeypatch, 4)
    assert DE._precedent_count() == 4
    precedent_n(monkeypatch, None, missing=True)
    with pytest.raises(GL.LimitUnavailable):
        DE._precedent_count()


# ---------------------------------------------------------------- the self-approval bar (final review C1)
SAME = "the requester decided an earlier case of this clash"


def test_a_requester_who_is_the_credited_decider_of_a_cited_case_escalates(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester="sub-a")
    assert (d.resolution, d.decision, d.rationale) == (DE.ESCALATED, "escalate", SAME)


@pytest.mark.parametrize("requester", ["SUB-A", "  sub-a  "])
def test_the_same_person_test_is_the_approvals_one(monkeypatch, requester):
    """Same semantics as approvals._same_person: trimmed, case-insensitive."""
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    assert DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester=requester).rationale == SAME


def test_a_requester_who_approved_a_member_case_of_a_cited_case_escalates(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)], approvers=["sub-c", "sub-x"])
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester="sub-x")
    assert (d.resolution, d.rationale) == (DE.ESCALATED, SAME)
    [(_sql, params)] = [x for x in cur.sql if "memberCases" in x[0]]
    assert params == (DE.LIVE_CONFLICT_SUBJECT, "TST-0001|TST-0002", ["pc_11", "pc_10"], "agent_policy_approval")


def test_a_requester_who_decided_nothing_is_decided_on_precedent(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "reject", "sub-b", AT), (10, "reject", "sub-a", AT)], approvers=["sub-c"])
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester="sub-z")
    assert (d.resolution, d.decision) == (DE.RESOLVED, "reject")
    assert [s for s, _ in cur.sql if "SAVEPOINT" in s] == [
        "SAVEPOINT live_conflict_precedent", "RELEASE SAVEPOINT live_conflict_precedent",
        "SAVEPOINT live_conflict_precedent_members", "RELEASE SAVEPOINT live_conflict_precedent_members"]


def test_a_failed_member_lookup_escalates_and_rolls_back_to_its_savepoint(monkeypatch):
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)], fail_approvers=True)
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester="sub-z")
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent lookup failed")
    assert [s for s, _ in cur.sql if "members" in s and "SAVEPOINT" in s] == [
        "SAVEPOINT live_conflict_precedent_members", "ROLLBACK TO SAVEPOINT live_conflict_precedent_members",
        "RELEASE SAVEPOINT live_conflict_precedent_members"]


def test_no_requester_bars_nobody_and_looks_up_no_member(monkeypatch):
    """A call no person asked for (a watcher): nobody's own request, so no member lookup."""
    _n(monkeypatch, 2)
    cur = _Cur([(11, "approve", "sub-b", AT), (10, "approve", "sub-a", AT)])
    d = DE.decide_live_conflict(cur, _lc(), ctx={}, now=NOW, requester=None)
    assert d.resolution == DE.RESOLVED and not [x for x in cur.sql if "memberCases" in x[0]]


def test_the_requester_is_required():
    with pytest.raises(TypeError):
        DE.decide_live_conflict(_Cur(), _lc(), ctx={}, now=NOW)   # noqa - the bar needs to know who asked


# ---------------------------------------------------------------- the governed value range (Task 12)
from tests.agent_policy.fixtures import precedent_range  # noqa: E402

AMOUNT = "args.amount"


def _ranged(key, version, *, fields=(AMOUNT, "args.urgent"), sensitive=()):
    cond = {"all": [{"field": "tool.name", "op": "in", "value": ["tst_refund"]}]
            + [{"field": f, "op": "exists"} for f in fields]}
    policy = {"id": key, "version": version, "trigger": {"condition": cond},
              "inputs": [{"field": f, "sensitive": True} for f in sensitive]}
    return {"id": key, "version": version, "outcome": "approve", "policy": policy}


def _rlc(**kw):
    return _lc(involved=(_ranged("TST-0001", 3, **kw), _ranged("TST-0002", 1)))


def _cases(*examples):
    """Cited cases, newest first, each with the overlap example its live record stored."""
    return [(20 - i, "approve", f"sub-{i}", AT, ex) for i, ex in enumerate(examples)]


def _decide(monkeypatch, cur, amount, *, pct=20, missing=False, lc=None, **ctx):
    _n(monkeypatch, len(cur.rows))
    precedent_range(monkeypatch, pct, missing=missing)
    ctx = {"tool.name": "tst_refund", "args": {"amount": amount, **ctx}}
    return DE.decide_live_conflict(cur, lc or _rlc(), ctx=ctx, now=NOW, requester=OTHER)


def test_within_the_range_is_decided_on_precedent(monkeypatch):
    cur = _Cur(_cases({AMOUNT: 900}, {AMOUNT: 1000}))
    d = _decide(monkeypatch, cur, 1200)
    assert (d.resolution, d.decision) == (DE.RESOLVED, "approve")
    assert d.facts["valueRange"] == {"pct": 20.0, "fields": {AMOUNT: {"value": 1200, "max": 1000, "limit": 1200.0}}}


def test_a_hundredth_of_a_percent_above_the_range_goes_to_people(monkeypatch):
    cur = _Cur(_cases({AMOUNT: 900}, {AMOUNT: 1000}))
    d = _decide(monkeypatch, cur, 1200.1)
    assert (d.resolution, d.decision) == (DE.ESCALATED, "escalate")
    assert d.rationale == "args.amount 1200.1 is more than 20% above the largest approved (1000)"


def test_the_lookup_returns_each_cited_cases_stored_example():
    sql = " ".join(DE.PRECEDENT_SQL.split())
    assert "e->>'kind' = 'overlap'" in sql and "e->'example'" in sql
    assert "JOIN proc.bp_decision d ON d.decision_id = c.decision_id" in sql


@pytest.mark.parametrize("amount,resolution", [(1000, DE.RESOLVED), (1000.01, DE.ESCALATED)])
def test_zero_percent_is_never_above_the_largest_approved(monkeypatch, amount, resolution):
    d = _decide(monkeypatch, _Cur(_cases({AMOUNT: 1000}, {AMOUNT: 400})), amount, pct=0)
    assert d.resolution == resolution
    if resolution == DE.ESCALATED:
        assert d.rationale == "args.amount 1000.01 is more than 0% above the largest approved (1000)"


def test_null_means_no_range_check(monkeypatch):
    d = _decide(monkeypatch, _Cur(_cases({AMOUNT: 10}, {AMOUNT: 10})), 1_000_000, pct=None)
    assert (d.resolution, d.facts["valueRange"]) == (DE.RESOLVED, {"pct": None, "fields": {}})


@pytest.mark.parametrize("pct,exc", [(None, "LimitUnavailable"), ("twenty", "ValueError"), (-5, "ValueError")])
def test_a_missing_or_unreadable_range_goes_to_people(monkeypatch, caplog, pct, exc):
    with caplog.at_level("WARNING", logger=DE.__name__):
        d = _decide(monkeypatch, _Cur(_cases({AMOUNT: 900}, {AMOUNT: 900})), 900, pct=pct, missing=pct is None)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "precedent value range unavailable")
    assert f"precedent value range unavailable, the clash TST-0001|TST-0002 goes to people: {exc}" in caplog.text


def test_a_cited_case_without_the_field_goes_to_people(monkeypatch):
    d = _decide(monkeypatch, _Cur(_cases({AMOUNT: 900}, {"args.other": 1})), 900)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "args.amount has no number in an earlier case")


@pytest.mark.parametrize("bad", ["900", True, None])
def test_a_cited_value_that_is_not_a_number_goes_to_people(monkeypatch, bad):
    d = _decide(monkeypatch, _Cur(_cases({AMOUNT: 900}, {AMOUNT: bad})), 900)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "args.amount has no number in an earlier case")


def test_a_cited_case_with_no_stored_example_goes_to_people(monkeypatch):
    d = _decide(monkeypatch, _Cur(_cases({AMOUNT: 900}, None)), 900)
    assert (d.resolution, d.rationale) == (DE.ESCALATED, "args.amount has no number in an earlier case")


def test_a_clash_with_no_numeric_condition_value_passes(monkeypatch):
    cur = _Cur(_cases({"tool.name": "tst_refund"}, {"tool.name": "tst_refund"}))
    d = _decide(monkeypatch, cur, "nine hundred")
    assert (d.resolution, d.facts["valueRange"]) == (DE.RESOLVED, {"pct": 20.0, "fields": {}})


def test_a_boolean_is_not_a_number(monkeypatch):
    """args.urgent True is 1 to Python, but never compared: only args.amount is."""
    cur = _Cur(_cases({AMOUNT: 900, "args.urgent": False}, {AMOUNT: 900, "args.urgent": False}))
    d = _decide(monkeypatch, cur, 900, urgent=True)
    assert d.resolution == DE.RESOLVED and list(d.facts["valueRange"]["fields"]) == [AMOUNT]


def test_the_first_failing_field_in_sorted_order_is_named(monkeypatch):
    lc = _lc(involved=(_ranged("TST-0001", 3, fields=("args.zeta", AMOUNT)), _ranged("TST-0002", 1)))
    cur = _Cur(_cases({AMOUNT: 1, "args.zeta": 1}))
    d = _decide(monkeypatch, cur, 5, lc=lc, zeta=5)
    assert d.rationale == "args.amount 5 is more than 20% above the largest approved (1)"


def test_a_sensitive_value_is_masked_in_the_rationale_and_the_facts(monkeypatch):
    lc = _rlc(sensitive=(AMOUNT,))
    d = _decide(monkeypatch, _Cur(_cases({AMOUNT: 900})), 5000, lc=lc)
    assert d.rationale == "args.amount ••• is more than 20% above the largest approved (•••)"
    assert "5000" not in d.rationale and "900" not in d.rationale
    ok = _decide(monkeypatch, _Cur(_cases({AMOUNT: 900})), 1000, lc=lc)
    assert ok.facts["valueRange"]["fields"] == {AMOUNT: {"value": "•••", "max": "•••", "limit": "•••"}}


def test_the_range_is_checked_only_after_n_of_n_agree(monkeypatch):
    cur = _Cur([(11, "approve", "sub-b", AT, {AMOUNT: 1}), (10, "reject", "sub-a", AT, {AMOUNT: 1})])
    assert _decide(monkeypatch, cur, 5000).rationale == "decisions disagree"


def test_the_real_range_is_the_governed_fresh_value(monkeypatch):
    precedent_range(monkeypatch, 12.5)
    assert DE._precedent_value_range_pct() == 12.5
    precedent_range(monkeypatch, None)
    assert DE._precedent_value_range_pct() is None
    precedent_range(monkeypatch, None, missing=True)
    with pytest.raises(GL.LimitUnavailable):
        DE._precedent_value_range_pct()
