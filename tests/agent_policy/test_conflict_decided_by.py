"""facts.decidedBy is written where a conflict case closes, never parsed from an actor (design §3.1)."""
import json
from datetime import datetime, timezone

import pytest

from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_engine as CE
from services.agent_policy import conflict_live as CL
from tests.agent_policy.test_conflict_live_gate import _Cur, make_doc

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=timezone.utc)
ACTION = {"tool": "refund.issue", "args": {"amount": 900, "note": "x"}, "agent": "a", "workflowId": "wf",
          "userId": "u"}


def _lc():
    a = make_doc("TSB-0101", tool="refund.issue", outcome="approve", deciders=["FM"], source="Finance")
    b = make_doc("TSB-0102", tool="refund.issue", outcome="approve", deciders=["CFO"], source="Customer")
    hits = [{"id": d["id"], "version": d["version"], "outcome": "approve", "policy": d} for d in (a, b)]
    return CE.LiveConflict("human", hits, [("TSB-0101", "TSB-0102")], hits, {"TSB-0101", "TSB-0102"})


def _insert(**kw):
    log = []
    CL.insert_live(_Cur(log, iter(range(7, 9))), _lc(), ctx={"tool.name": "refund.issue", "args": ACTION["args"]},
                   action=ACTION, now=NOW, default_response_time="PT4H", **kw)
    sql, params = next((s, json.loads(p)) for s, p in log if s.startswith("INSERT INTO proc.bp_decision"))
    return params, json.loads(params[7]), json.loads(params[8])


def test_a_precedent_record_says_precedent_and_cites_its_cases():
    cited = [{"kind": "precedent", "caseId": "pc_3", "decision_id": 3, "outcome": "approve",
              "actioned_by": "sub-x", "actioned_at": "2026-10-09T10:00:00+00:00"}]
    params, facts, evidence = _insert(status="actioned", decision="approve", actor=CL.PRECEDENT,
                                      reason="Decided the same way 1 times before", extra_evidence=cited)
    assert facts["decidedBy"] == {"kind": "precedent", "name": "system:precedent"}
    assert facts["versionsAtDecision"] == {"TSB-0101": 1, "TSB-0102": 1}
    assert evidence[0]["kind"] == "overlap" and evidence[1:] == cited
    assert (params[2], params[5], params[15], params[16]) == ("approve", "actioned", "this_action", "system:precedent")


@pytest.mark.parametrize("decision,kind,name", [("block", "block", "system:not_allowed"),
                                                ("standing_rule", "standing_rule", "system:standing_rule")])
def test_block_and_standing_rule_records_say_so(decision, kind, name):
    _params, facts, _ev = _insert(status="actioned", decision=decision)
    assert facts["decidedBy"] == {"kind": kind, "name": name}
    assert facts["versionsAtDecision"] == {"TSB-0101": 1, "TSB-0102": 1}


def test_only_precedent_may_record_approve_or_reject():
    with pytest.raises(ValueError):
        _insert(status="actioned", decision="approve", actor="sub-someone")
    with pytest.raises(ValueError):
        _insert(status="actioned", decision="maybe")


def test_an_open_record_has_no_decided_by_and_keeps_its_extras():
    _params, facts, _ev = _insert(status="open", extra_facts={"history": [{"caseId": "pc_1"}],
                                                               "precedent": {"why": "decisions disagree"}})
    assert "decidedBy" not in facts and "versionsAtDecision" not in facts
    assert facts["history"] == [{"caseId": "pc_1"}] and facts["precedent"] == {"why": "decisions disagree"}


def test_decided_by_refuses_an_unknown_kind():
    assert CC.decided_by("person", "sub-x") == {"kind": "person", "name": "sub-x"}
    with pytest.raises(ValueError):
        CC.decided_by("guess", "system:timeout")


# ------------------------------------------- settle_for_group: who settled a live clash (fix round 1)
class _SettleCur:
    """Answers the reads settle_for_group makes for one member group and one open live case (id 40)."""

    def __init__(self, acts):
        self.acts, self.log, self.description, self._last = acts, [], None, ""

    def execute(self, sql, params=None):
        self._last = " ".join(sql.split())
        self.log.append((self._last, params))

    def fetchall(self):
        if "a.override_reason FROM proc.bp_decision a" in self._last:
            return list(self.acts)
        if "'liveConflict'" in self._last:
            return [(40,)]
        return []

    def fetchone(self):
        if "RETURNING" in self._last:
            return (41,)
        if "SELECT policy_versions" in self._last:
            return ({"TSB-0101": 1, "TSB-0102": 1},)
        if "FROM proc.bp_decision WHERE decision_id = %s AND subject_type = %s FOR UPDATE" in self._last:
            self.description = [(c,) for c in ("decision_id", "subject_id", "resolution", "rationale",
                                               "policy_name", "facts", "status", "workflow_id")]
            return (40, "TSB-0101|TSB-0102", "escalated", "why", "TSB-0101|TSB-0102", "{}", "open", "wf")
        return None


def _settle(monkeypatch, acts, *, refused, actor):
    from services.agent_policy import approvals as A
    monkeypatch.setattr(A, "_member_ids", lambda cur, did, facts: [11, 12])
    proposed = []
    monkeypatch.setattr(CL, "_propose_safely", lambda cur, live_id, now: proposed.append(live_id))
    cur = _SettleCur(acts)
    CL.settle_for_group(cur, {"decision_id": 11, "facts": {}}, {"total": 2, "rejected": 1, "approved": 0},
                        refused=refused, actor=actor, reason=None, now=NOW)
    facts = next(json.loads(p[6]) for s, p in cur.log if s.startswith("INSERT INTO proc.bp_decision"))
    by_person = next(p[3] for s, p in cur.log if s.startswith("UPDATE proc.bp_agent_policy_conflict"))
    return facts, by_person, proposed


def test_a_credited_timeout_row_settles_as_a_timeout(monkeypatch):
    facts, by_person, proposed = _settle(monkeypatch, [("reject", "system:timeout", "timed out")],
                                         refused="timed_out", actor="system:timeout")
    assert facts["decidedBy"] == {"kind": "timeout", "name": "system:timeout"}
    assert facts["versionsAtDecision"] == {"TSB-0101": 1, "TSB-0102": 1}
    assert by_person is False and proposed == []


def test_a_credited_person_settles_as_a_person(monkeypatch):
    facts, by_person, proposed = _settle(monkeypatch, [("reject", "sub-fm", "no")], refused=None, actor="sub-fm")
    assert facts["decidedBy"] == {"kind": "person", "name": "sub-fm"}
    assert by_person is True and proposed == [40]


def test_a_credited_group_row_is_system_never_timeout_or_person(monkeypatch, caplog):
    facts, by_person, proposed = _settle(monkeypatch, [("reject", "system:group", None)],
                                         refused="rejected", actor="system:group")
    assert facts["decidedBy"] == {"kind": "system", "name": "system:group"}
    assert by_person is False and proposed == []
    assert "settled by unexpected system actor system:group" in caplog.text
