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
