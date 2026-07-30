"""The turn's event stream must carry the scope and the question it refers to.

A client that renders only `question` events would show "Does this scope look right?"
with no scope above it; one that stops at `complete` would show a finished
requirement and swallow the confirm question. Both happened.
"""
from src.api.routers.requirements import _events_for

SCOPE = {"family_label": "technology / SaaS",
         "areas": [{"area": "Service levels", "requirement": "99.95% availability."}] * 3}


def _kinds(result):
    return [e["event"] for e in _events_for(result)]


def test_a_proposed_scope_gets_its_own_event_before_the_question():
    events = _events_for({"mode": "proposed_scope", "complete": False, "scope": SCOPE,
                          "next_question": "Does this scope look right?"})
    assert [e["event"] for e in events] == ["thinking", "scope", "question"]
    assert "3 requirement areas" in events[1]["message"]
    assert "technology / SaaS" in events[1]["message"]
    assert events[0]["message"] == "Drafting a scope"


def test_a_completing_turn_still_carries_its_confirm_question():
    kinds = _kinds({"mode": "complete", "complete": True, "scope": SCOPE,
                    "summary": "Requirement REQ-1 captured.",
                    "next_question": "Does this scope look right?"})
    assert kinds == ["thinking", "scope", "complete", "question"]


def test_plain_elicitation_is_unchanged():
    events = _events_for({"mode": "elicitation", "complete": False,
                          "next_question": "How many users?"})
    assert [e["event"] for e in events] == ["thinking", "question"]
    assert events[0]["message"] == "Reviewing requirement"


def test_no_empty_question_event():
    # A turn where the LLM produced nothing must not emit a blank question bubble.
    assert _kinds({"mode": "elicitation", "complete": False, "next_question": ""}) == ["thinking"]
