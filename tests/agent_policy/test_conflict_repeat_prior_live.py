"""A repeat proposal's "Previous decisions" on bp_testdb (stage 5b, Task B6). Needs PROCWISE_TEST_LIVE_DB=1.

test_conflict_live_live's world: two REAL live policies on one tool, retired at teardown; every
case, action, conflict and notification row made here is removed by that fixture.
"""
import os

import pytest

from services.agent_policy import conflict_live as CL
from tests.agent_policy import fixtures as F
from tests.agent_policy import test_conflict_live_gate as T
from tests.agent_policy.test_conflict_live_live import conn, once, repeat_cases, world  # noqa: F401 - fixtures

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live DB required")


def _live_priors(conn, w):
    return [r["facts"]["priorDecisions"] for r in T.rows(
        conn, "SELECT facts FROM proc.bp_decision WHERE subject_type = %s AND workflow_id = ANY(%s) "
              "AND decision = 'approve_or_reject' ORDER BY decision_id", (CL.SUBJECT_LIVE, w.wfs))]


def test_the_proposal_raised_from_three_live_decisions_says_three_not_first_time(conn, world, monkeypatch):
    F.precedent_n(monkeypatch, 3)
    for _ in range(3):
        once(conn, world, monkeypatch)
    [pc] = repeat_cases(conn, world)
    assert pc["facts"]["proposal"] == {"from": "repeat", "count": 3, "outcome": "approve"}
    assert pc["facts"]["priorDecisions"] == {"sameConflict": 3, "lastOutcome": "approve"}
    assert pc["facts"]["summary"]["prior"] == {"sameConflict": 3, "lastOutcome": "approve"}
    # no other case's count moved: each live clash still counts the live clashes settled before it
    assert _live_priors(conn, world) == [{"sameConflict": 0, "lastOutcome": None},
                                         {"sameConflict": 1, "lastOutcome": "approve"},
                                         {"sameConflict": 2, "lastOutcome": "approve"}]


def test_a_proposal_raised_from_rejections_says_rejected(conn, world, monkeypatch):
    F.precedent_n(monkeypatch, 2)
    once(conn, world, monkeypatch)                    # an approval first: it is not one of the two
    once(conn, world, monkeypatch, "reject")
    assert repeat_cases(conn, world) == []
    once(conn, world, monkeypatch, "reject")
    [pc] = repeat_cases(conn, world)
    assert pc["facts"]["priorDecisions"] == {"sameConflict": 2, "lastOutcome": "reject"}, \
        "the two it was raised from, not every clash these policies ever had"
