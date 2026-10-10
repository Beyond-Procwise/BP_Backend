"""A repeat proposal's "Previous decisions" (stage 5b, Task B6; stage 4 defect D3). No database.

A policy case the repeat rule raises (facts.proposal.from == "repeat") was raised FROM decisions:
the last N live clashes of the same policies that people settled the same way. It reports those,
not "first time". No other case's count changes.
"""
import json
from datetime import datetime, timezone

import pytest

from services.agent_policy import conflict_cases as CC
from tests.agent_policy.test_conflict_cases import _pair

NOW = datetime(2026, 10, 10, 9, 0, tzinfo=timezone.utc)
NONE_BEFORE = {"sameConflict": 0, "lastOutcome": None}


@pytest.mark.parametrize("policy_prior,proposal,expected", [
    (NONE_BEFORE, {"from": "repeat", "count": 5, "outcome": "approve"}, {"sameConflict": 5, "lastOutcome": "approve"}),
    (NONE_BEFORE, {"from": "repeat", "count": 2, "outcome": "reject"}, {"sameConflict": 2, "lastOutcome": "reject"}),
    # the pair's own earlier policy cases are still counted; the live decisions are the newest
    ({"sameConflict": 1, "lastOutcome": "change:FIN-0012"}, {"from": "repeat", "count": 5, "outcome": "approve"},
     {"sameConflict": 6, "lastOutcome": "approve"}),
    # every other case: exactly what it was
    (NONE_BEFORE, None, NONE_BEFORE),
    ({"sameConflict": 3, "lastOutcome": "keep_both:FIN-0012"}, None, {"sameConflict": 3, "lastOutcome": "keep_both:FIN-0012"}),
    ({"sameConflict": 3, "lastOutcome": "retire:CUS-0004"}, {"from": "save", "count": 5, "outcome": "approve"},
     {"sameConflict": 3, "lastOutcome": "retire:CUS-0004"}),
    (NONE_BEFORE, {"count": 5, "outcome": "approve"}, NONE_BEFORE),
    # a proposal whose count cannot be read adds nothing (never a guess, never an error)
    (NONE_BEFORE, {"from": "repeat", "count": None, "outcome": "approve"}, NONE_BEFORE),
    (NONE_BEFORE, {"from": "repeat", "count": "five", "outcome": "approve"}, NONE_BEFORE),
    (NONE_BEFORE, {"from": "repeat", "count": True, "outcome": "approve"}, NONE_BEFORE),
    (NONE_BEFORE, {"from": "repeat", "count": -1, "outcome": "approve"}, NONE_BEFORE),
    ({"sameConflict": 1, "lastOutcome": "limit:FIN-0012"}, {"from": "repeat", "count": 2},
     {"sameConflict": 3, "lastOutcome": "limit:FIN-0012"}),
])
def test_only_a_repeat_proposal_counts_the_live_decisions_it_was_raised_from(policy_prior, proposal, expected):
    before = dict(policy_prior)
    got = CC._prior_for(policy_prior, proposal)
    assert got == expected and set(got) == {"sameConflict", "lastOutcome"}, "the shape is what it was"
    assert policy_prior == before, "the pair's own count is not altered in place"


class _Cur:
    """A scripted cursor for raise_policy_case: nothing covers the pair, `prior` policy cases were
    decided before, no standing rule; it keeps what is inserted."""

    def __init__(self, prior):
        self.prior, self.inserted, self._one, self._all = prior, [], None, []

    def execute(self, sql, params=None):
        sql = " ".join(sql.split())
        self._one, self._all = None, []
        if sql.startswith("SELECT count(*)"):
            self._one = self.prior
        elif sql.startswith("INSERT INTO proc.bp_decision"):
            self.inserted.append(params)
            self._one = (41,)

    def fetchone(self): return self._one
    def fetchall(self): return self._all


def _raised_facts(prior, **kw):
    a, b = _pair()
    cur = _Cur(prior)
    assert CC.raise_policy_case(cur, a, b, {"args.amount": 10001}, now=NOW, mapping={}, **kw) == 41
    [params] = cur.inserted
    return json.loads(params[4])


def test_the_case_a_repeat_proposal_raises_does_not_read_as_first_time():
    facts = _raised_facts((0, None), raised_by="repeat", proposal={"from": "repeat", "count": 5, "outcome": "approve"})
    assert facts["priorDecisions"] == {"sameConflict": 5, "lastOutcome": "approve"}
    assert facts["summary"]["prior"] == facts["priorDecisions"], "the card's summary says the same"
    assert facts["proposal"] == {"from": "repeat", "count": 5, "outcome": "approve"}


def test_a_case_raised_any_other_way_reports_what_it_always_did():
    for kw in ({"raised_by": "save"}, {"raised_by": "scan"}, {"raised_by": "extraction", "proposal": None}):
        facts = _raised_facts((0, None), **kw)
        assert facts["priorDecisions"] == NONE_BEFORE == facts["summary"]["prior"], kw
    facts = _raised_facts((2, "keep_both:FIN-0012"), raised_by="save")
    assert facts["priorDecisions"] == {"sameConflict": 2, "lastOutcome": "keep_both:FIN-0012"}
