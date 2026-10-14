"""Each guard, broken on purpose, watched fail.

A test that only exercises the happy path proves the code runs, not that the
guard guards. These invert each one.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from src.repositories.playbook_repo import LifecycleError, check_approval  # noqa: E402
from src.services.playbooks.finding_source import Finding, validate_trigger_match  # noqa: E402
from src.services.playbooks.selector import select  # noqa: E402
from src.services.playbooks.store import Playbook, PlaybookStore  # noqa: E402


def pb(pid, match, source="detection_finding"):
    return Playbook(playbook_id=pid, playbook_name=f"pb{pid}", trigger_source=source,
                    trigger_match=match, agent_workflow_id=958, params={}, version=1)


def f(**attrs):
    return Finding(source="detection_finding", finding_id="1", deal_id="D-1", attrs=attrs)


def test_guard_the_tie_rule_actually_refuses():
    """Remove the tie and one is chosen; restore it and nothing is. If the
    first half of this passes and the second does too, the tie rule is doing
    the work rather than the test asserting a foregone conclusion."""
    one = [pb(1, {"rule_id": "duplicate"})]
    two = [pb(1, {"rule_id": "duplicate"}), pb(2, {"severity": "critical"})]
    finding = f(rule_id="duplicate", severity="critical")
    assert select(finding, one) is not None      # the same finding DOES match
    assert select(finding, two) is None          # and the tie is what stops it


def test_guard_the_self_approval_bar_is_the_thing_refusing():
    """Change only the approver and the same call succeeds."""
    with pytest.raises(LifecycleError):
        check_approval(authored_by="ana", approver="ana",
                       current_status="pending_approval", workflow_is_active=True)
    check_approval(authored_by="ana", approver="bo",
                   current_status="pending_approval", workflow_is_active=True)


def test_guard_the_match_key_allow_list_is_the_thing_refusing():
    validate_trigger_match("detection_finding", {"severity": "critical"})
    with pytest.raises(ValueError):
        validate_trigger_match("detection_finding", {"sevrity": "critical"})


def test_guard_active_only_is_the_thing_filtering():
    row = {
        "playbook_id": 1, "playbook_name": "x", "trigger_source": "detection_finding",
        "trigger_match": {}, "agent_workflow_id": 958, "params": {},
        "playbook_status": "active", "version": 1,
    }
    assert len(PlaybookStore(playbook_rows=[dict(row)]).active_playbooks()) == 1
    assert PlaybookStore(
        playbook_rows=[dict(row, playbook_status="retired")]
    ).active_playbooks() == []


def test_guard_a_null_attribute_is_not_a_wildcard():
    """If this ever starts passing with doc_type=None, the comparison has
    started treating absence as a match and every specific playbook has
    quietly become a catch-all."""
    specific = [pb(1, {"rule_id": "duplicate", "doc_type": "invoice"})]
    assert select(f(rule_id="duplicate", doc_type="invoice"), specific) is not None
    assert select(f(rule_id="duplicate", doc_type=None), specific) is None
