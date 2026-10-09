"""Each closing path writes facts.decidedBy (and live settlements facts.versionsAtDecision).

Needs PROCWISE_TEST_LIVE_DB=1 and DB_NAME=bp_testdb. Uses the conflict-endpoint world: DRAFTS with
tool tst_<tag>, retired at teardown; the gate gets the same documents injected.
"""
import os
from datetime import timedelta

import pytest

from services.agent_policy import approvals as A
from services.agent_policy import conflict_cases as CC
from services.agent_policy import conflict_live as CL
from tests.agent_policy import test_conflict_live_gate as LG
from tests.agent_policy.test_conflict_endpoints_live import (  # noqa: F401 - fixtures
    NOW, _a, _b, _c, _design_case, _gate, conn, world)

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


def _closing_facts(conn, subject_type, case_id):
    [row] = LG.rows(conn, "SELECT facts FROM proc.bp_decision WHERE subject_type = %s AND status = 'actioned' "
                          "AND actioned_by IS NOT NULL AND facts->>'caseId' = %s "
                          "ORDER BY decision_id DESC LIMIT 1", (subject_type, f"pc_{case_id}"))
    return row["facts"]


def test_a_person_deciding_a_policy_case(conn, world):
    a, c, did = _design_case(conn, world)
    CC.decide_policy(conn, did, principal=LG.who(world, world.oa), option=f"retire:{c}", reason="Old rule",
                     limit_text=None, now=NOW)
    assert _closing_facts(conn, CC.SUBJECT_POLICY, did)["decidedBy"] == \
        {"kind": "person", "name": f"sub-{world.email_a}"}


def test_a_retired_policy_closing_its_case(conn, world):
    a, c, did = _design_case(conn, world)
    assert CC.close_moot(conn, c, now=NOW) == 1
    assert _closing_facts(conn, CC.SUBJECT_POLICY, did)["decidedBy"] == {"kind": "retired", "name": "system:retired"}


def test_people_settling_a_live_clash(conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    ma, mb = LG.members(conn, world)
    LG.act(conn, world, ma["decision_id"], world.la2)
    LG.act(conn, world, mb["decision_id"], world.lb)
    facts = _closing_facts(conn, CL.SUBJECT_LIVE, lv["decision_id"])
    assert facts["decidedBy"] == {"kind": "person", "name": f"sub-{world.people[world.lb]}"}
    assert facts["versionsAtDecision"] == {a: 1, b: 1}


def test_a_timeout_settling_a_live_clash(conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    A.sweep(conn, lv["respond_by"] + timedelta(seconds=1), decision_ids=[m["decision_id"] for m in LG.members(conn, world)])
    facts = _closing_facts(conn, CL.SUBJECT_LIVE, lv["decision_id"])
    assert facts["decidedBy"] == {"kind": "timeout", "name": "system:timeout"}
    assert facts["versionsAtDecision"] == {a: 1, b: 1}


def test_a_block_record(conn, world, monkeypatch):
    (a, da), (c, dc) = _a(conn, world), _c(conn, world)
    _gate(monkeypatch, world, [da, dc], amount=20000)
    [lv] = LG.lives(conn, world)
    assert lv["facts"]["decidedBy"] == {"kind": "block", "name": "system:not_allowed"}
    assert lv["facts"]["versionsAtDecision"] == {a: 1, c: 1}


def test_a_standing_rule_record(conn, world, monkeypatch):
    (a, da), (b, db) = _a(conn, world), _b(conn, world)
    da = {**da, "conflicts": [{"with": b, "prevails": a, "rule": f"{a} takes priority over {b}"}]}
    _gate(monkeypatch, world, [da, db])
    [lv] = LG.lives(conn, world)
    assert lv["facts"]["decidedBy"] == {"kind": "standing_rule", "name": "system:standing_rule"}
