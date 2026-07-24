"""The style rules have to be enforced, not merely requested.

Style lives in the governed persona (bp_prompt ask_persona/joshi), but a prompt
is a request — nothing stopped the model returning `## Findings`, `---` rules,
`> quotes` or emoji decoration, and nothing in the post-generation chain looked.
This guard is the enforcement half: presentation markers are removed after
generation so the rendered answer matches the governed style whatever the model
does.

The hard constraint is content preservation. A previous formatting rewriter in
this repo turned "£1,200. Delivery was late" into a numbered item reading
"200. Delivery was late" — it treated a figure as a list marker. So the
invariant tested here is byte-level: every word, digit, and currency figure
survives; only formatting characters may disappear.
"""

import os
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from services.response_style import enforce_response_style

_TOKENS = re.compile(r"[0-9][0-9,.]*|[A-Za-z][A-Za-z'’-]*|[£$€]")


def _content_tokens(text: str):
    """Words, numbers and currency symbols — everything that carries meaning."""

    return _TOKENS.findall(text)


def test_atx_headers_are_removed_but_their_words_survive():
    out = enforce_response_style("## Supplier breakdown\n\nWe have 123 suppliers.")

    assert "#" not in out
    assert "Supplier breakdown" in out
    assert "123" in out


def test_horizontal_rules_are_dropped():
    out = enforce_response_style("Spend is £1.8M.\n\n---\n\nThat is up 13%.")

    assert "---" not in out
    assert "£1.8M" in out and "13%" in out


def test_blockquotes_are_unwrapped_not_deleted():
    out = enforce_response_style("> Payment terms are Net 30.")

    assert not out.lstrip().startswith(">")
    assert "Payment terms are Net 30." in out


def test_whole_line_bold_is_unwrapped():
    """A bolded line on its own is a fake header; inline bold is left alone."""

    out = enforce_response_style("**Total spend**\nWe spent **£1.8M** last year.")

    assert "**Total spend**" not in out
    assert "Total spend" in out
    # The genuinely emphasised figure keeps its emphasis.
    assert "**£1.8M**" in out


def test_decorative_emoji_are_stripped_from_line_edges():
    out = enforce_response_style("🔍 Findings\nThree invoices are flagged 🚩")

    assert "🔍" not in out and "🚩" not in out
    assert "Findings" in out
    assert "Three invoices are flagged" in out


def test_currency_followed_by_a_full_stop_is_not_treated_as_a_list_marker():
    """The exact regression that broke the previous formatter."""

    source = "The invoice totalled £1,200. Delivery was late."
    out = enforce_response_style(source)

    assert out.strip() == source
    assert "£1,200." in out


def test_clause_numbers_in_prose_survive():
    source = "Refer to clause 14. Payment terms are Net 30."
    assert enforce_response_style(source).strip() == source


def test_content_is_byte_identical_apart_from_formatting_markers():
    source = (
        "## Q1 summary\n\n"
        "> Spend reached £1,824,950.43 across 51 invoices.\n\n"
        "---\n\n"
        "**Key risks**\n"
        "- Orbis Platform Solutions Ltd at £1,116,000.00 🚩\n"
        "- 19 of 123 suppliers invoiced\n"
    )
    out = enforce_response_style(source)

    assert _content_tokens(out) == _content_tokens(source)


def test_genuine_lists_are_left_intact():
    source = "1. Coffee Bliss — 7 invoices\n2. Veruca Organic — 7 invoices"
    assert enforce_response_style(source).strip() == source


def test_empty_and_none_are_safe():
    assert enforce_response_style("") == ""
    assert enforce_response_style(None) == ""
