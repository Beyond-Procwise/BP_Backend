"""The rules that make an approval mean something."""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.repositories.playbook_repo import (  # noqa: E402
    LifecycleError,
    check_approval,
    next_status_for_edit,
)


def test_editing_an_active_playbook_returns_it_for_approval_and_bumps_version():
    """An approved strategy cannot be changed underneath its approval."""
    assert next_status_for_edit("active") == ("pending_approval", True)


def test_editing_a_draft_leaves_it_a_draft():
    assert next_status_for_edit("draft") == ("draft", False)


def test_editing_something_awaiting_approval_leaves_it_awaiting_approval():
    assert next_status_for_edit("pending_approval") == ("pending_approval", False)


def test_a_retired_playbook_cannot_be_edited():
    with pytest.raises(LifecycleError) as exc:
        next_status_for_edit("retired")
    assert "retired" in str(exc.value)


def test_approval_moves_a_pending_playbook():
    check_approval(authored_by="ana", approver="bo",
                   current_status="pending_approval", workflow_is_active=True)


def test_nobody_approves_their_own_playbook():
    with pytest.raises(LifecycleError) as exc:
        check_approval(authored_by="ana", approver="ana",
                       current_status="pending_approval", workflow_is_active=True)
    assert "own" in str(exc.value).lower()


def test_self_approval_is_barred_whatever_the_case_or_spacing():
    with pytest.raises(LifecycleError):
        check_approval(authored_by="Ana@x.com", approver=" ana@X.com ",
                       current_status="pending_approval", workflow_is_active=True)


def test_an_anonymous_approver_is_refused():
    """Without a subject the self-approval bar cannot be applied at all, so an
    unattributable approval is worse than no approval."""
    for approver in (None, "", "   "):
        with pytest.raises(LifecycleError):
            check_approval(authored_by="ana", approver=approver,
                           current_status="pending_approval", workflow_is_active=True)


def test_only_a_pending_playbook_can_be_approved():
    for status in ("draft", "active", "retired"):
        with pytest.raises(LifecycleError) as exc:
            check_approval(authored_by="ana", approver="bo",
                           current_status=status, workflow_is_active=True)
        assert status in str(exc.value)


def test_a_playbook_pointing_at_an_inactive_workflow_cannot_be_approved():
    """Approving it would create a strategy that can only ever fail at the
    moment somebody accepts its proposal."""
    with pytest.raises(LifecycleError) as exc:
        check_approval(authored_by="ana", approver="bo",
                       current_status="pending_approval", workflow_is_active=False)
    assert "workflow" in str(exc.value).lower()


# -- the transitions themselves, against a cursor that records the SQL --------
#
# get_conn() is AUTOCOMMIT and hands out a fresh connection per call, so a
# read-then-write across two of them shares no snapshot and holds no lock. The
# status the decision was made on has to travel into the WHERE clause, or a
# concurrent edit lands between the two and the write overwrites it.

import src.repositories.playbook_repo as repo  # noqa: E402


class RecordingCursor:
    def __init__(self, rows, rowcount=1):
        self._rows = list(rows)
        self.rowcount = rowcount
        self.statements = []
        self.params = []

    def execute(self, sql, params=None):
        self.statements.append(" ".join(sql.split()))
        self.params.append(params)

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchall(self):
        return list(self._rows)

    def close(self):
        pass


class RecordingConn:
    def __init__(self, cursor):
        self.cursor_obj = cursor

    def cursor(self):
        return self.cursor_obj

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _existing(status="pending_approval", version=1):
    """The tuple shape _row() expects, in _COLUMNS order."""
    return (12, "Recover it", None, "detection_finding", {"rule_id": "duplicate"},
            958, {}, status, version, "ana", None, None, None, None, "ana")


def test_approve_pins_the_status_it_decided_on(monkeypatch):
    cur = RecordingCursor([_existing(), (True,), _existing("active")])
    monkeypatch.setattr(repo, "get_conn", lambda: RecordingConn(cur))
    repo.approve(12, approver="bo")
    update = [s for s in cur.statements if s.startswith("UPDATE")][0]
    assert "playbook_status = %s" in update and "WHERE playbook_id = %s AND playbook_status = %s" in update


def test_approve_refuses_when_the_row_moved_under_it(monkeypatch):
    """Ana edits between Bo's read and Bo's write: Bo's UPDATE matches nothing
    and must say so rather than reporting an approval that did not happen."""
    cur = RecordingCursor([_existing(), (True,)], rowcount=0)
    monkeypatch.setattr(repo, "get_conn", lambda: RecordingConn(cur))
    with pytest.raises(LifecycleError) as exc:
        repo.approve(12, approver="bo")
    assert "changed" in str(exc.value).lower()


def test_update_pins_the_version_it_read(monkeypatch):
    cur = RecordingCursor([_existing("active", version=3), _existing("pending_approval", 4)])
    monkeypatch.setattr(repo, "get_conn", lambda: RecordingConn(cur))
    repo.update(12, name="x", trigger_source="detection_finding", trigger_match={},
                agent_workflow_id=958, params={}, description=None, modified_by="ana")
    update = [s for s in cur.statements if s.startswith("UPDATE")][0]
    assert "AND playbook_status = %s AND version = %s" in update


def test_submit_refuses_when_the_row_moved_under_it(monkeypatch):
    cur = RecordingCursor([_existing("draft")], rowcount=0)
    monkeypatch.setattr(repo, "get_conn", lambda: RecordingConn(cur))
    with pytest.raises(LifecycleError):
        repo.submit(12, modified_by="ana")


def test_claiming_a_proposal_is_what_stops_a_double_run(monkeypatch):
    """The claim must be a conditional UPDATE whose rowcount decides, so two
    simultaneous approvals cannot both proceed to start a workflow."""
    cur = RecordingCursor([], rowcount=1)
    monkeypatch.setattr(repo, "get_conn", lambda: RecordingConn(cur))
    assert repo.claim_proposal(5, "bo") is True
    stmt = cur.statements[0]
    assert stmt.startswith("UPDATE proc.bp_playbook_proposal")
    # The statuses are bound, not inlined: the claim moves proposed -> approved
    # and the old status is the guard in the WHERE.
    assert "WHERE proposal_id = %s AND proposal_status = %s" in stmt
    assert cur.params[0][0] == "approved"
    assert cur.params[0][-1] == "proposed"

    lost = RecordingCursor([], rowcount=0)
    monkeypatch.setattr(repo, "get_conn", lambda: RecordingConn(lost))
    assert repo.claim_proposal(5, "bo") is False
