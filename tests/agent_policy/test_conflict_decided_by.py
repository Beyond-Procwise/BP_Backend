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
