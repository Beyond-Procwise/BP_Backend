"""A follow-up about the last answer must not become a new search.

"put that into a table" carries no retrievable terms, so the literal phrase was
embedded and the nearest neighbours came back — a random invoice line item. The
user asked for the supplier spend they were already looking at and got
INV600263, BESPOK-D89A8BC4 and a tax percentage instead.

Two separate things are being read out of a question here, and conflating them
is what broke it:

* ``is_presentation_directive`` — does this turn need NEW data, or is it asking
  to re-present what is already on screen? "put that into a table" needs no
  retrieval. "show me the top suppliers in a table" does.
* ``requested_format`` — the shape asked for, which applies either way.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services.model_selector import RAGPipeline


def _p() -> RAGPipeline:
    return RAGPipeline.__new__(RAGPipeline)


# --------------------------------------------------------------------------
# Directive detection — does this turn need new data?
# --------------------------------------------------------------------------

DIRECTIVES = [
    "put that into a table",
    "put that in a table",
    "as a table please",
    "can you show that as a table",
    "summarise that in a table",
    "in bullets please",
    "make it shorter",
    "shorten that",
    "just the top 3",
    "show those as a list",
]

NEW_QUESTIONS = [
    "how many invoices do we have?",
    "which suppliers do we spend the most with?",
    "show me the top suppliers in a table",
    "list our contracts expiring this quarter as a table",
    "what is our total spend?",
    "put the Orbis invoices into a table",
]


def test_presentation_directives_are_recognised():
    p = _p()
    for text in DIRECTIVES:
        assert p._is_presentation_directive(text), text


def test_real_questions_are_not_mistaken_for_directives():
    """A question that names its own subject needs retrieval, table or not."""
    p = _p()
    for text in NEW_QUESTIONS:
        assert not p._is_presentation_directive(text), text


def test_a_directive_with_no_previous_turn_is_not_a_directive():
    """Nothing to re-present on the first turn — fall through to a real search."""
    p = _p()
    assert p._is_presentation_directive("put that into a table", has_history=False) is False


# --------------------------------------------------------------------------
# Requested format — orthogonal to the above
# --------------------------------------------------------------------------


def test_requested_format_is_read_from_either_kind_of_question():
    p = _p()
    assert p._requested_format("put that into a table") == "table"
    assert p._requested_format("show me the top suppliers in a table") == "table"
    assert p._requested_format("in bullets please") == "list"
    assert p._requested_format("how many invoices do we have?") is None


def test_format_instruction_names_the_data_source_for_a_directive():
    """The directive prompt must forbid inventing rows the last answer lacked."""
    p = _p()
    instruction = p._directive_instruction("put that into a table", "Prior answer text.")

    lowered = instruction.lower()
    assert "table" in lowered
    # The previous answer is the material, embedded verbatim.
    assert "Prior answer text." in instruction
    # It must be explicit that no new facts may appear.
    assert "do not add" in lowered or "only" in lowered
