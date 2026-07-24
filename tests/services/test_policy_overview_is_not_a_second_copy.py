"""The overview line orients the reader; it does not restate the sections.

`_extract_policy_payload` built the overview from the whole document summary::

    overview = doc_overview_candidates[0]

while that *same* summary was simultaneously fed to `_ingest_text`, which split
it into sentences and filed each one under a heading. So every rule in the
summary was printed twice — once inside the opening line, once as a bullet::

    Here's what Travel Policy says about travel: Travel is booked through the
    corporate portal. Flights must be booked 14 days in advance. Alcohol
    cannot be claimed.

    What's allowed
    - Flights must be booked 14 days in advance.

    What's not allowed
    - Alcohol cannot be claimed.

The `focus_answer` fallback had the same defect in a milder form: it took
`sentences[0]`, one sentence rather than the blob, but `focus_answer` is a
topic Q&A answer that `_ingest_text` also categorises, so its first sentence
could still surface twice.

The overview's job is the part of the source that has no heading — what the
policy is, who it covers, when it took effect. Anything that belongs under a
heading is shown there and only there. When the source is nothing but rules the
overview is empty, and `_render_policy_response` already opens on the sections.

Related: [[test_policy_categorisation]] decides which sentences have a heading.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from agents.rag_agent import QARecord, RAGAgent, TopicRecord


@pytest.fixture()
def agent() -> RAGAgent:
    return RAGAgent.__new__(RAGAgent)


class SimpleDoc:
    def __init__(self, payload):
        self.payload = payload


def _payload_for(agent, summary, *, focus_answer="", depth_mode="concise", qas=()):
    topic = TopicRecord(topic="travel", context_prompts=(), qas=tuple(qas))
    return agent._extract_policy_payload(
        policy_name="travel policy",
        topic_entry=topic,
        focus_answer=focus_answer,
        depth_mode=depth_mode,
        policy_docs=[SimpleDoc({"summary": summary})] if summary else [],
    )


SUMMARY = (
    "The Travel Policy covers business travel for all Group employees. "
    "Flights must be booked 14 days in advance. "
    "Alcohol cannot be claimed on any trip."
)


# --------------------------------------------------------------------------
# The overview is not a second copy of the sections
# --------------------------------------------------------------------------


def test_the_overview_does_not_repeat_a_clause_shown_under_a_heading(agent):
    payload = _payload_for(agent, SUMMARY)

    assert "Flights must be booked 14 days in advance" not in payload["overview"]
    assert "Alcohol cannot be claimed" not in payload["overview"]
    # ...and those clauses are still on screen, under their headings.
    assert payload["requirements"] == ["Flights must be booked 14 days in advance."]
    assert payload["restrictions"] == ["Alcohol cannot be claimed on any trip."]


def test_the_overview_keeps_the_sentence_that_has_no_heading(agent):
    payload = _payload_for(agent, SUMMARY)
    assert payload["overview"] == (
        "The Travel Policy covers business travel for all Group employees."
    )


def test_the_overview_is_one_sentence_not_the_whole_summary(agent):
    """It was the entire blob, so the opening line ran to a paragraph."""
    payload = _payload_for(agent, SUMMARY)
    assert payload["overview"].count(".") == 1


def test_a_summary_that_is_only_rules_yields_no_overview(agent):
    payload = _payload_for(
        agent,
        "Flights must be booked 14 days in advance. Alcohol cannot be claimed.",
    )
    assert payload["overview"] == ""


def test_a_later_scope_sentence_is_still_found(agent):
    """The orienting sentence does not have to come first."""
    payload = _payload_for(
        agent,
        "Alcohol cannot be claimed. This policy applies to all Group entities.",
    )
    assert payload["overview"] == "This policy applies to all Group entities."


# --------------------------------------------------------------------------
# The focus_answer fallback behaves the same way
# --------------------------------------------------------------------------


def test_the_focus_answer_fallback_does_not_repeat_a_categorised_sentence(agent):
    qa = QARecord(
        question="What is the travel policy?",
        answer="Flights must be booked 14 days in advance. It is reviewed each July.",
    )
    payload = _payload_for(agent, "", focus_answer=qa.answer, qas=[qa])

    assert "Flights must be booked 14 days in advance" not in payload["overview"]
    assert payload["requirements"] == ["Flights must be booked 14 days in advance."]


def test_the_focus_answer_fallback_still_provides_an_overview(agent):
    answer = "The Travel Policy governs how staff book business trips."
    payload = _payload_for(agent, "", focus_answer=answer)
    assert payload["overview"] == answer


# --------------------------------------------------------------------------
# What the reader actually sees
# --------------------------------------------------------------------------


def test_no_clause_is_printed_twice_in_the_rendered_answer(agent):
    payload = _payload_for(agent, SUMMARY)
    out = agent._render_policy_response(payload, "concise", "what is the travel policy?")

    assert out.count("Flights must be booked 14 days in advance") == 1
    assert out.count("Alcohol cannot be claimed") == 1


def test_the_rendered_answer_still_opens_on_the_overview(agent):
    payload = _payload_for(agent, SUMMARY)
    out = agent._render_policy_response(payload, "concise", "what is the travel policy?")
    assert out.startswith("Here’s what Travel Policy says about")
    assert "covers business travel for all Group employees" in out


def test_a_rules_only_policy_opens_on_the_sections(agent):
    payload = _payload_for(agent, "Alcohol cannot be claimed on any trip.")
    out = agent._render_policy_response(payload, "concise", "what is the travel policy?")
    assert out.startswith("From Travel Policy:")
    assert ":." not in out


# --------------------------------------------------------------------------
# What must keep working
# --------------------------------------------------------------------------


def test_a_wholly_descriptive_summary_is_still_the_overview(agent):
    summary = "The Travel Policy covers business travel for all Group employees."
    payload = _payload_for(agent, summary)
    assert payload["overview"] == summary


def test_no_source_at_all_leaves_the_overview_empty(agent):
    payload = _payload_for(agent, "")
    assert payload["overview"] == ""


def test_the_overview_is_still_punctuated(agent):
    payload = _payload_for(
        agent, "The Travel Policy covers business travel for all Group employees"
    )
    assert payload["overview"].endswith(".")
