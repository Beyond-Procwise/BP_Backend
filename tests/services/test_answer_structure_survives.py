"""Line structure has to survive the trip from model to screen.

The model lays an answer out as a lead sentence, a blank line, then a list. Two
things used to flatten that back into one line before it reached the renderer:

* the request is sent with ``format=json``, and a literal newline is illegal
  inside a JSON string — so the model wrote none at all. Fixed in the persona by
  asking for escaped ``\\n``, because the constraint is the encoding, not the
  writing, and no amount of style instruction moves it.
* ``_remove_placeholders`` ran ``re.sub(r"\\s+", " ")``, which treats a newline
  as whitespace like any other. Every paragraph break and list marker in the
  answer was collapsed to a space on the way out.

With the newlines gone the renderer had one long line to interpret, and read
"The suppliers with the highest spend are:" as a definition term — so a
three-item list rendered as a single <dd> with the bullets run together.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services.model_selector import RAGPipeline

_ANSWER = (
    "The suppliers with the highest spend are:\n"
    "- Orbis Platform Solutions Ltd: £1,116,000.00\n"
    "- Meridian Consulting Group Ltd: £461,454.00\n"
    "- PeopleFirst HR Solutions Ltd: £100,000.00\n"
    "\n"
    "Orbis is the largest by a wide margin."
)


def _pipeline() -> RAGPipeline:
    return RAGPipeline.__new__(RAGPipeline)


def test_placeholder_removal_keeps_line_breaks():
    out = _pipeline()._remove_placeholders(_ANSWER)

    assert out.count("\n") >= 4, repr(out)
    assert "\n- Orbis" in out


def test_placeholder_removal_still_collapses_runs_of_spaces():
    out = _pipeline()._remove_placeholders("Total   spend    is £1.8M")

    assert out == "Total spend is £1.8M"


def test_placeholder_removal_still_removes_the_placeholders():
    out = _pipeline()._remove_placeholders("We have 51 invoices [doc 3] on record.")

    assert "[doc 3]" not in out
    assert "51 invoices" in out


def test_a_bulleted_answer_renders_as_a_list_not_a_definition():
    html = _pipeline()._plain_text_to_html(_ANSWER)

    assert "<ul" in html, html
    assert html.count("<li>") == 3, html
    # The lead line is prose, not a definition term: a trailing colon with
    # nothing after it is not a "term: value" pair.
    assert "<dt>" not in html, html
    assert "£1,116,000.00" in html


def test_the_trailing_colon_lead_line_stays_a_paragraph():
    html = _pipeline()._plain_text_to_html("Top suppliers:\n- Orbis: £1.1M\n- Meridian: £461k")

    assert "Top suppliers:" in html
    assert html.count("<li>") == 2
