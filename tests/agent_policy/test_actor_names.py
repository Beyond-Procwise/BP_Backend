"""Who is shown by name (stage 5b, B1): the pure rule. The stored email when there is one, else
the sign-in id; a `system:*` decider in the words the conflict history uses, never as an address."""
import pytest

from services.agent_policy import approval_views as V
from services.agent_policy import conflict_history as H


@pytest.mark.parametrize("who,email,shown", [
    ("sub-123", "ana@example.test", "ana@example.test"),
    ("sub-123", "  ana@example.test  ", "ana@example.test"),
    ("sub-123", None, "sub-123"),            # an old row, or a sign-in that carries no email
    ("sub-123", "", "sub-123"),
    ("sub-123", "   ", "sub-123"),
    (None, None, None),                      # nobody decided: there is no name either
    ("", None, None),
    (None, "ana@example.test", None),        # no id shown -> no name shown
])
def test_a_person_is_shown_by_email_else_by_id(who, email, shown):
    assert V.shown_name(who, email) == shown


@pytest.mark.parametrize("who,shown", [
    ("system:timeout", "Timeout"),
    ("system:precedent", "Precedent"),
    ("system:retired", "Policy retired"),
    ("system:group", "By the system"),       # no word of its own
    ("System:Timeout", "Timeout"),
    (" system:timeout", "Timeout"),
    ("system:person", "By the system"),      # never passed off as a person
    ("system:", "By the system"),
])
def test_a_system_decider_is_worded_as_the_conflict_history_words_it(who, shown):
    assert H.system_words(who) == shown
    assert V.shown_name(who) == shown
    assert V.shown_name(who, "ana@example.test") == shown, "a system decider never takes an email"
    assert shown in H.DECIDED_BY_WORDS.values()


def test_a_person_is_not_a_system_decider():
    assert H.system_words("sub-123") is None and H.system_words(None) is None and H.system_words("") is None


def test_the_firing_view_names_the_decider_only_beside_the_id():
    row = {"firing_id": 1, "policy_key": "TST-1a", "policy_version": 1, "checkpoint": "c", "action_name": "a",
           "agent": None, "workflow_id": None, "requested_by": None, "outcome": "approve", "result": "approved",
           "matched_values": {}, "missing_inputs": [], "decision_id": None, "decided_level": 0,
           "decided_by": "sub-123", "decided_at": None, "reason": None, "duration_ms": None, "reversal_of": None,
           "created_at": None}
    assert V._firing_view(row, set())["decidedByName"] == "sub-123", "no email is recorded for a decision"
    assert V._firing_view({**row, "decided_by": "system:timeout"}, set())["decidedByName"] == "Timeout"
    assert V._firing_view({**row, "decided_by": None}, set())["decidedByName"] is None
    plain = V._firing_view(row, set())
    assert "reversed" not in plain, "a read that did not join the reversal does not claim to know"
    old = V._firing_view({**row, "rv_by": "sub-9", "rv_at": None, "rv_reason": "r"}, set())["reversed"]
    assert old == {"by": "sub-9", "byName": "sub-9", "at": None, "reason": "r"}, "a read without the email column"
    new = V._firing_view({**row, "rv_by": "sub-9", "rv_email": "bo@example.test", "rv_at": None, "rv_reason": "r"},
                         set())["reversed"]
    assert new == {"by": "sub-9", "byName": "bo@example.test", "at": None, "reason": "r"}
    assert V._firing_view({**row, "rv_by": None, "rv_email": None, "rv_at": None, "rv_reason": None},
                          set())["reversed"] is None
