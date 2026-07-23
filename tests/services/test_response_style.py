"""The assistant must sound like it read the question, not like a mail merge.

Every one of these tests pins a specific canned-response generator that was
firing on the live /ask path. They are grouped by where the boilerplate was
being manufactured:

  * the prompt we send the model (it was ordered to open with a greeting),
  * the persona row (same order, phrased differently, so both had to go),
  * the post-processor (it injected a sentiment-picked opening sentence and
    flattened every answer into one run-on paragraph),
  * the HTML renderer (a fixed "Here's what I found" heading on every reply,
    plus a list-splitter that turned ordinary numbers into list items),
  * the static-QA formatter (fixed intro, hash-picked sign-off, and a
    thesaurus pass that rewrote the source wording),
  * the retrieval draft and the policy renderer (stock acknowledgements,
    emoji section headers, and a sign-off about expense claims that was
    appended to answers about anything at all).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from agents.rag_agent import RAGAgent
from services.model_selector import RAGPipeline
from services.nltk_pipeline import NLTKProcessor


@pytest.fixture()
def pipeline() -> RAGPipeline:
    """A bare pipeline: these are pure text helpers, no I/O involved."""
    obj = RAGPipeline.__new__(RAGPipeline)
    obj._citation_guidelines = RAGPipeline._load_citation_guidelines(obj)
    return obj


@pytest.fixture(scope="module")
def proc() -> NLTKProcessor:
    processor = NLTKProcessor()
    if not processor.available:
        pytest.skip("NLTK resources unavailable in test environment")
    return processor


# --------------------------------------------------------------------------
# What we ask the model for
# --------------------------------------------------------------------------


def test_the_model_is_not_told_to_open_with_an_acknowledgement(pipeline):
    """The instruction that produced "Thanks for the question—" on every reply."""
    prompt = pipeline._compose_llm_prompt(
        "How much did we spend with Veruca last quarter?",
        "Knowledge snippet",
        "Draft",
        None,
        "",
    )
    lowered = prompt.lower()
    assert "acknowledgement" not in lowered
    assert "collegial" not in lowered
    assert "lead-in" not in lowered


def test_the_persona_does_not_order_a_stock_greeting(pipeline):
    """The persona said "avoid boilerplate openers" and then supplied two."""
    persona = RAGPipeline._ASK_PERSONA_FALLBACK
    assert "Happy to help!" not in persona
    assert "Thanks for the question" not in persona
    assert "brief acknowledgement or collegial greeting" not in persona
    # The parts that keep the answer honest must survive untouched.
    assert "Never add amounts denominated in different currencies." in persona
    assert "avoid boilerplate openers or stock phrases" in persona


# --------------------------------------------------------------------------
# Post-processing of what the model returned
# --------------------------------------------------------------------------


def test_postprocess_does_not_prepend_a_sentiment_sentence(proc):
    """A negative-sounding question earned a canned sympathy line, every time."""
    out = proc.postprocess(
        "Late fees apply to overdue invoices and the supplier has charged them.",
        sentiment={"compound": -0.6},
    )
    assert "I understand this situation" not in out
    assert "may feel frustrating" not in out
    assert "Late fees apply" in out


def test_postprocess_does_not_invent_good_news(proc):
    out = proc.postprocess(
        "The quote is within budget and the supplier is approved.",
        sentiment={"compound": 0.8},
    )
    assert "Great news" not in out
    assert "within budget" in out


def test_postprocess_keeps_the_lists_the_model_wrote(proc):
    """It joined every sentence with a space, so bullets arrived as one blob."""
    text = "Here are the next steps:\n- Confirm the quote\n- Raise the PO\n- Notify finance"
    out = proc.postprocess(text)
    assert "- Confirm the quote" in out
    assert "- Raise the PO" in out
    assert out.count("\n") >= 3


def test_postprocess_does_not_full_stop_a_list_lead_in(proc):
    """"Key reminders:" is introducing the list below it, not a sentence."""
    out = proc.postprocess("Key reminders:\n- Upload files with unique names")
    assert "Key reminders:." not in out
    assert "Key reminders:" in out


def test_postprocess_keeps_paragraph_breaks(proc):
    text = "Spend rose 14% year on year.\n\nThe increase sits with two suppliers."
    out = proc.postprocess(text)
    assert "\n\n" in out


def test_postprocess_still_drops_contentless_filler(proc):
    """The filler filter is not what we are removing — it must keep working."""
    out = proc.postprocess("Sure, I can help. The contract expires in March.")
    assert "Sure, I can help" not in out
    assert "The contract expires in March." in out


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------


def test_rendered_answer_has_no_fixed_heading(pipeline):
    html = pipeline._plain_text_to_html("Spend with Veruca was £12,400 last quarter.")
    assert "what I found" not in html
    assert "easy-to-scan summary" not in html
    assert "agent-answer__heading" not in html
    assert "£12,400" in html


def test_a_single_paragraph_answer_is_not_declared_empty(pipeline):
    """The "is it empty?" check counted elements, and the header was one of them.

    With the header removed, a one-paragraph answer hit the same count an empty
    body used to, so a complete answer had "No answer available." stapled to the
    end of it.
    """
    html = pipeline._plain_text_to_html("We have 51 invoices in total, 45 of them against a PO.")
    assert "No answer available" not in html
    assert "51 invoices" in html


def test_an_empty_body_still_says_so(pipeline):
    html = pipeline._plain_text_to_html("Sources: contract.pdf")
    assert "No answer available" in html


def test_a_thousands_separated_number_is_not_split_into_a_list(pipeline):
    """`£1,200. Delivery` was being rendered as list item "200. Delivery"."""
    html = pipeline._plain_text_to_html(
        "The invoice totalled £1,200. Delivery was late by two weeks."
    )
    assert "£1,200" in html
    assert "<ol" not in html
    assert ">200." not in html


def test_a_sentence_ending_in_a_number_is_not_split_into_a_list(pipeline):
    html = pipeline._plain_text_to_html(
        "The contract runs to clause 14. Payment terms sit in the schedule."
    )
    assert "<ol" not in html
    assert "clause 14" in html


def test_a_genuine_inline_enumeration_still_becomes_a_list(pipeline):
    html = pipeline._plain_text_to_html(
        "Next steps: 1. Confirm the quote 2. Raise the PO 3. Notify finance"
    )
    assert "<ol" in html
    assert "Confirm the quote" in html
    assert "Notify finance" in html


# --------------------------------------------------------------------------
# Static-QA formatting
# --------------------------------------------------------------------------


def test_static_answer_gets_no_manufactured_intro_or_sign_off(pipeline):
    out = pipeline._format_static_answer(
        "Spend rose 14% against the same quarter last year.",
        question="How does this compare with the same period last year?",
        topic="spend",
    )
    assert "quick take" not in out.lower()
    assert "Let me know if you'd like supporting detail" not in out
    assert "Happy to pull the supporting policy" not in out
    assert "We can dig into related metrics" not in out
    assert out.strip() == "Spend rose 14% against the same quarter last year."


def test_static_answer_does_not_rewrite_the_source_wording(pipeline):
    """It swapped "approximately"→"about" and "due to"→"thanks to" in policy text."""
    source = "Approximately 12% of spend is non-claimable due to the VAT rules."
    out = pipeline._format_static_answer(source, question="What is non-claimable?")
    assert "Approximately" in out
    assert "due to" in out
    assert "thanks to" not in out


def test_static_answer_keeps_multi_sentence_prose_as_prose(pipeline):
    """Every sentence after the first was turned into a bullet."""
    source = "Travel must be booked through the portal. Receipts are required within 30 days."
    out = pipeline._format_static_answer(source, question="What is the travel policy?")
    assert not out.lstrip().startswith("-")
    assert "\n- " not in out


# --------------------------------------------------------------------------
# Structural reformatting of prose
# --------------------------------------------------------------------------


def test_authored_prose_is_not_re_paragraphed(pipeline):
    """It cut a lead after three sentences and broke before "You"/"This"/"We"."""
    text = (
        "We reviewed the three quotes. Two suppliers responded on time. "
        "Pricing is within 4% across all of them. You should confirm the scope "
        "before awarding."
    )
    assert pipeline._apply_structured_formatting(text) == text


def test_authored_paragraphs_and_bullets_survive(pipeline):
    text = "Spend rose 14%.\n\n- Veruca: £12,400\n- Dixon Reynolds: £638"
    assert pipeline._apply_structured_formatting(text) == text


# --------------------------------------------------------------------------
# The retrieval draft and the policy renderer
# --------------------------------------------------------------------------


def test_the_retrieval_draft_carries_no_stock_acknowledgement(pipeline):
    """This draft is shown to the model as the shape to follow — so it spreads."""
    draft = pipeline._build_structured_answer("What is the travel policy?", [], "")
    assert "Sure, I can help you with that" not in draft
    assert "Most definitely" not in draft
    assert "Great! Here is the response" not in draft
    assert "You're in the right place" not in draft
    assert "escalate, refresh the data" not in draft
    # Nor a restatement of the question, nor a lead-in to the facts.
    assert "You asked about" not in draft
    assert "what I found related to your question" not in draft
    assert "These points capture the essentials" not in draft


def test_policy_answers_use_plain_headings(pipeline):
    agent = RAGAgent.__new__(RAGAgent)
    payload = {
        "policy_name": "travel policy",
        "overview": "Travel must be booked through the corporate portal",
        "requirements": ["Book through the portal"],
        "restrictions": ["No first class travel"],
        "spending_limits": ["Hotels are capped at £150 per night"],
        "approval_process": [],
        "operational_notes": [],
    }
    out = agent._render_policy_response(payload, "concise", "what is the travel policy?")
    for emoji in ("✅", "❌", "📎", "➡️"):
        assert emoji not in out
    assert "Book through the portal" in out
    assert "first class travel" in out
    # The headings still have to say which list is which.
    assert "What's allowed" in out
    assert "What's not allowed" in out


def test_policy_answers_do_not_end_on_a_scripted_expense_claim_offer(pipeline):
    """That sign-off was appended to every policy answer, expenses or not."""
    agent = RAGAgent.__new__(RAGAgent)
    payload = {
        "policy_name": "travel policy",
        "overview": "Travel must be booked through the corporate portal",
        "requirements": ["Book through the portal"],
        "restrictions": [],
        "spending_limits": [],
        "approval_process": [],
        "operational_notes": [],
    }
    out = agent._render_policy_response(payload, "concise", "what is the travel policy?")
    assert "expense claim" not in out.lower()
    assert "walk you through it" not in out.lower()


def test_policy_answers_do_not_pad_a_missing_overview(pipeline):
    """With no overview it asserted it had "pulled the relevant guardrails"."""
    agent = RAGAgent.__new__(RAGAgent)
    payload = {
        "policy_name": "travel policy",
        "overview": "",
        "requirements": ["Book through the portal"],
        "restrictions": [],
        "spending_limits": [],
        "approval_process": [],
        "operational_notes": [],
    }
    out = agent._render_policy_response(payload, "concise", "what is the travel policy?")
    assert "guardrails" not in out.lower()
    assert ":." not in out
    assert out.startswith("From Travel Policy:")
