"""One finding, one proposal -- however many times the sweep runs.

The insert is exercised against a fake cursor that records statements, so the
SQL's shape is pinned without a database. The live behaviour of the unique
index is proved in Task 10's guard proofs, by dropping it.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.services.playbooks import proposer  # noqa: E402
from src.services.playbooks.finding_source import Finding  # noqa: E402
from src.services.playbooks.selector import Selection  # noqa: E402
from src.services.playbooks.store import Playbook  # noqa: E402


class FakeCursor:
    """Answers the INSERT ... RETURNING, and remembers what it was asked."""

    def __init__(self, returns):
        self._returns = list(returns)
        self.statements = []
        self.params = []

    def execute(self, sql, params=None):
        self.statements.append(" ".join(sql.split()))
        self.params.append(params)

    def fetchone(self):
        return self._returns.pop(0) if self._returns else None

    def close(self):
        pass


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture(autouse=True)
def no_audit(monkeypatch):
    """The audit writer is best-effort and tested in its own suite."""
    recorded = []
    monkeypatch.setattr(
        proposer.agent_actions, "record_action",
        lambda **kw: recorded.append(kw),
    )
    return recorded


def playbook(pid=7, source="detection_finding", version=1):
    return Playbook(
        playbook_id=pid, playbook_name="Recover the duplicate",
        trigger_source=source, trigger_match={"rule_id": "duplicate"},
        agent_workflow_id=958, params={}, version=version,
    )


def selection(pb=None):
    pb = pb or playbook()
    return Selection(
        playbook=pb,
        evidence={"matched": {"rule_id": "duplicate"}, "key_count": 1,
                  "playbook_version": pb.version},
    )


def finding(source="detection_finding", fid="4211", deal="D-900"):
    return Finding(source=source, finding_id=fid, deal_id=deal,
                   attrs={"rule_id": "duplicate"})


def test_a_new_finding_gets_one_proposal():
    cur = FakeCursor([(55,)])
    pid = proposer.propose(finding(), selection(), conn=FakeConn(cur))
    assert pid == 55
    assert "INSERT INTO proc.bp_playbook_proposal" in cur.statements[0]
    assert "ON CONFLICT DO NOTHING" in cur.statements[0]
    assert "RETURNING proposal_id" in cur.statements[0]


def test_the_same_finding_twice_proposes_once():
    """ON CONFLICT DO NOTHING returns no row on the second attempt."""
    cur = FakeCursor([(55,), None])
    conn = FakeConn(cur)
    assert proposer.propose(finding(), selection(), conn=conn) == 55
    assert proposer.propose(finding(), selection(), conn=conn) is None


def test_a_version_bump_does_not_re_propose():
    """The unique index is keyed on playbook_id, not version, on purpose: an
    edited playbook must not raise a second proposal for a finding already
    queued."""
    cur = FakeCursor([(55,), None])
    conn = FakeConn(cur)
    proposer.propose(finding(), selection(playbook(version=1)), conn=conn)
    assert proposer.propose(finding(), selection(playbook(version=2)), conn=conn) is None


def test_the_same_id_in_both_stores_is_two_different_findings():
    """bp_detection_finding.finding_id 123 and bp_opportunity.opportunity_id
    '123' both land in one TEXT column; finding_source keeps them apart."""
    cur = FakeCursor([(1,), (2,)])
    conn = FakeConn(cur)
    a = proposer.propose(finding(source="detection_finding", fid="123"), selection(), conn=conn)
    b = proposer.propose(
        finding(source="opportunity", fid="123"),
        selection(playbook(pid=8, source="opportunity")),
        conn=conn,
    )
    assert (a, b) == (1, 2)
    assert cur.params[0][1] == "detection_finding"
    assert cur.params[1][1] == "opportunity"


def test_the_row_carries_playbook_finding_deal_and_evidence():
    cur = FakeCursor([(55,)])
    proposer.propose(finding(), selection(), conn=FakeConn(cur))
    playbook_id, source, finding_id, deal_id, evidence = cur.params[0]
    assert playbook_id == 7
    assert source == "detection_finding"
    assert finding_id == "4211"
    assert deal_id == "D-900"
    assert '"rule_id": "duplicate"' in evidence


def test_a_proposal_is_audited(no_audit):
    proposer.propose(finding(), selection(), conn=FakeConn(FakeCursor([(55,)])))
    assert len(no_audit) == 1
    event = no_audit[0]
    assert event["phase"] == "playbook"
    assert event["action_type"] == "playbook.propose"
    assert event["deal_id"] == "D-900"
    assert "Recover the duplicate" in event["summary"]


def test_a_duplicate_is_not_audited_as_a_new_proposal(no_audit):
    """A sweep that re-reads 4,838 open findings every fifteen minutes would
    otherwise write 4,838 audit rows an hour saying nothing happened."""
    cur = FakeCursor([None])
    assert proposer.propose(finding(), selection(), conn=FakeConn(cur)) is None
    assert no_audit == []


def test_ambiguity_is_recorded_as_its_own_event(no_audit):
    proposer.record_ambiguous(finding(), [playbook(1), playbook(2)])
    assert len(no_audit) == 1
    event = no_audit[0]
    assert event["action_type"] == "playbook.ambiguous"
    assert event["status"] == "skipped"
    assert "1" in event["summary"] and "2" in event["summary"]
