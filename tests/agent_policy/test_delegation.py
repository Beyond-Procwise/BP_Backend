"""Handing an approval to a named person (D-73 A): the pure rules, no database."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from services.agent_policy import delegation as D
from services.agent_policy.approvals import ApprovalRefused

NOW = datetime(2026, 10, 22, 19, 0, tzinfo=timezone.utc)


def person(subject="p-1", active=True, limit=50000, email="p1@example.test", name="Priya Sharma"):
    return {"subject": subject, "is_active": active, "approval_limit": limit, "email": email, "display_name": name}


def test_receiver_must_be_an_active_known_person():
    assert D.receiver_problem(None, {}, by="me") == "That person is not an active, known person."
    assert "not an active" in D.receiver_problem(person(active=False), {}, by="me")


def test_receiver_cannot_be_the_one_handing_it_on_or_the_requester():
    assert D.receiver_problem(person(subject="me"), {}, by="ME") == "You cannot hand an approval to yourself."
    out = D.receiver_problem(person(), {"requestedBy": "P1@example.test"}, by="me")
    assert out == "Priya Sharma asked for this action and cannot decide on it."
    assert "cannot decide on it" in D.receiver_problem(person(), {"requestedBy": "p-1"}, by="me")


def test_receiver_needs_a_limit_that_covers_the_amount():
    assert D.receiver_problem(person(limit=None), {}, by="me") == \
        "Priya Sharma has no approval limit, so cannot decide approvals."
    facts = {"action": {"args": {"amount": "60000"}}}
    assert D.receiver_problem(person(limit=50000), facts, by="me") == \
        "Priya Sharma's approval limit does not cover this amount."
    assert D.receiver_problem(person(limit=60000), facts, by="me") is None
    assert D.receiver_problem(person(limit=-1), facts, by="me") is None          # unlimited
    assert D.receiver_problem(person(limit=0), {"amount": "not a number"}, by="me") is None   # no amount named


def test_amount_is_read_from_facts_then_the_action_args():
    assert D.amount_of({"amount": 12.5}) == 12.5
    assert D.amount_of({"action": {"args": {"amount": "3"}}}) == 3
    assert D.amount_of({"action": {"args": {"amount": True}}}) is None
    assert D.amount_of({}) is None


def test_the_person_it_was_handed_to_counts_as_a_delegate_by_subject_or_email():
    facts = {"delegatedTo": {"subject": "p-1", "email": "P1@example.test"}}
    assert D.is_delegate(SimpleNamespace(subject="p-1", email=None), facts)
    assert D.is_delegate(SimpleNamespace(subject="x", email="p1@example.test"), facts)
    assert not D.is_delegate(SimpleNamespace(subject="x", email="y@example.test"), facts)
    assert not D.is_delegate(SimpleNamespace(subject="p-1", email=None), {})
    assert D.current({"delegatedTo": {"subject": ""}}) is None


@pytest.mark.parametrize("kw, code", [
    (dict(to_subject="  ", note=None), "receiver_required"),
    (dict(to_subject="p-1", note="x" * (D.MAX_NOTE + 1)), "note_too_long"),
])
def test_refusals_that_need_no_database(kw, code):
    with pytest.raises(ApprovalRefused) as e:
        D.delegate(None, 1, principal=SimpleNamespace(subject="me"), now=NOW, is_admin=False, mapping={}, **kw)
    assert e.value.code == code and e.value.status == 422


def test_an_anonymous_caller_is_refused_before_any_database_work():
    with pytest.raises(ApprovalRefused) as e:
        D.delegate(None, 1, principal=SimpleNamespace(subject=""), to_subject="p-1", note=None, now=NOW,
                   is_admin=False, mapping={})
    assert e.value.code == "not_signed_in" and e.value.status == 401


def test_the_event_type_is_in_the_catalogue_and_mandatory():
    from services.people.events import CATALOGUE
    assert CATALOGUE[D.EVENT_TYPE] == ("high", True)
