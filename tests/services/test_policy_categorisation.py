"""Each policy sentence belongs under one heading — the right one.

`_extract_policy_payload` sorted retrieved policy sentences into seven buckets
by running seven independent keyword tests over each one. Two consequences:

* **A sentence landed in every bucket it matched.** "All expenses must be
  submitted with a valid receipt and approved by a line manager" matched the
  obligation test, the approval test *and* the documentation test, so the same
  sentence was printed three times under three different headings.

* **The catch-all patterns swallowed nearly everything.** `review` alone routed
  a sentence to the approval process; `document|record|submit|process` alone
  routed it to operational notes; a bare `include` — how policies enumerate
  their own rules — routed it to examples.

Worse, nothing ranked polarity above topic, so a sentence could be filed as a
requirement (rendered under "What's allowed") purely because it contained
"must", even when what it said was "must not".

There was also a backfill: in expanded mode any empty section was filled with
the policy overview, which then appeared verbatim under "What's allowed",
"What's not allowed" and "Other conditions" at once — asserting the overview as
a prohibition it never was.

Related: [[test_policy_bullets_are_grounded]] covers what happens to a clause
once it has been categorised.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from agents.rag_agent import QARecord, RAGAgent, TopicRecord


@pytest.fixture()
def agent() -> RAGAgent:
    return RAGAgent.__new__(RAGAgent)


# --------------------------------------------------------------------------
# Polarity outranks topic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence",
    [
        "Alcohol cannot be claimed on any expense report.",
        "First class travel is not allowed for domestic journeys.",
        "Personal entertainment is non-claimable.",
        "Employees must not book travel outside the corporate portal.",
        "Fines and penalties are prohibited expenses.",
        "Gifts to public officials may not be offered under any circumstances.",
        "Claims submitted after 90 days will be declined.",
    ],
)
def test_a_prohibition_is_a_restriction(agent, sentence):
    assert agent._categorise_policy_sentence(sentence) == "restrictions"


def test_must_not_is_never_read_as_must(agent):
    """"must" matched before anything checked whether it was "must not"."""
    assert agent._categorise_policy_sentence(
        "Staff must not approve their own expense claims."
    ) == "restrictions"


def test_a_prohibition_that_also_names_an_amount_stays_a_restriction(agent):
    assert agent._categorise_policy_sentence(
        "Gifts over £50 cannot be claimed."
    ) == "restrictions"


# --------------------------------------------------------------------------
# The remaining sections
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("All expenses must be submitted within 30 days.", "requirements"),
        ("Employees are required to retain the original receipt.", "requirements"),
        ("Hotel accommodation is capped at £150 per night.", "spending_limits"),
        ("Meals are limited to £30 per person.", "spending_limits"),
        ("Operational expenditure has a threshold of $10,000.", "spending_limits"),
        ("Purchases require sign-off from the Category Lead.", "approval_process"),
        ("Spend above the threshold is escalated for authorisation.", "approval_process"),
        ("Overnight stays are allowed unless the journey is under two hours.", "exceptions"),
        ("An exemption may be sought from the Finance Business Partner.", "exceptions"),
        ("Claimable travel includes flights, such as economy airfare.", "examples"),
        ("For example, taxis to client sites are claimable.", "examples"),
        ("Attach the purchase order reference to every invoice.", "operational_notes"),
        ("Each claim is reconciled against the supporting receipt.", "operational_notes"),
    ],
)
def test_sentences_route_to_the_right_section(agent, sentence, expected):
    assert agent._categorise_policy_sentence(sentence) == expected


# --------------------------------------------------------------------------
# The catch-alls no longer catch everything
# --------------------------------------------------------------------------


def test_the_word_review_alone_is_not_an_approval_process(agent):
    assert agent._categorise_policy_sentence(
        "This policy is subject to annual review by the Executive Committee."
    ) != "approval_process"


def test_the_word_include_alone_is_not_an_example(agent):
    """Policies enumerate their own rules with "include"; that is not an example."""
    assert agent._categorise_policy_sentence(
        "The policy applies to all Group entities, and its scope includes contractors."
    ) != "examples"


def test_a_sentence_matching_nothing_is_left_uncategorised(agent):
    assert agent._categorise_policy_sentence(
        "This policy was ratified by the board in July."
    ) is None


def test_an_empty_sentence_is_left_uncategorised(agent):
    assert agent._categorise_policy_sentence("   ") is None


# --------------------------------------------------------------------------
# One sentence, one heading
# --------------------------------------------------------------------------


def _payload_for(agent, text, depth_mode="concise"):
    topic = TopicRecord(topic="expenses", context_prompts=(), qas=())
    return agent._extract_policy_payload(
        policy_name="expense policy",
        topic_entry=topic,
        focus_answer="",
        depth_mode=depth_mode,
        policy_docs=[SimpleDoc({"summary": text})],
    )


class SimpleDoc:
    def __init__(self, payload):
        self.payload = payload


def test_a_sentence_appears_under_exactly_one_heading(agent):
    """It matched obligation, approval and documentation, and printed thrice."""
    sentence = (
        "All expenses must be submitted with a valid receipt and approved by a line manager."
    )
    payload = _payload_for(agent, sentence)

    sections_containing = [
        key
        for key, clauses in payload.items()
        if isinstance(clauses, list) and any(sentence in c for c in clauses)
    ]
    assert len(sections_containing) == 1, (
        f"sentence filed under {sections_containing}"
    )


def test_a_prohibition_never_lands_under_requirements(agent):
    payload = _payload_for(agent, "Alcohol cannot be claimed on any expense report.")
    assert payload["requirements"] == []
    assert payload["restrictions"] == ["Alcohol cannot be claimed on any expense report."]


# --------------------------------------------------------------------------
# The overview is not evidence for every section
# --------------------------------------------------------------------------


def test_expanded_mode_does_not_backfill_empty_sections_with_the_overview(agent):
    overview = "Employees may claim business expenses incurred in the course of their work."
    payload = _payload_for(agent, overview, depth_mode="expanded")

    assert payload["overview"]
    for key in ("requirements", "restrictions", "spending_limits", "approval_process"):
        assert overview not in payload[key], (
            f"the overview was asserted as a {key} entry"
        )


# --------------------------------------------------------------------------
# What must keep working
# --------------------------------------------------------------------------


def test_document_sourced_clauses_still_reach_the_payload(agent):
    payload = _payload_for(
        agent,
        "All expenses must be submitted within 30 days. Alcohol cannot be claimed.",
    )
    assert payload["requirements"] == ["All expenses must be submitted within 30 days."]
    assert payload["restrictions"] == ["Alcohol cannot be claimed."]


def test_topic_qa_answers_still_populate_sections(agent):
    topic = TopicRecord(
        topic="expenses",
        context_prompts=(),
        qas=(
            QARecord(
                question="What are the spending limits?",
                answer="Meals are limited to £30 per person.",
            ),
        ),
    )
    payload = agent._extract_policy_payload(
        policy_name="expense policy",
        topic_entry=topic,
        focus_answer="",
        depth_mode="concise",
    )
    assert "Meals are limited to £30 per person." in payload["spending_limits"]
