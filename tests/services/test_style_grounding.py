"""The grounding guard.

The system prompt tells the model to invent nothing. It does anyway — reliably, and
fluently enough that the result looks ready to send. These tests cover the enforcement
half.

The two things that matter: an invented fact must not survive, and a supplied fact must
not be mangled. A guard that replaced real figures would be worse than none, because
people would learn to ignore it.
"""

from __future__ import annotations

import pytest

from services.style.grounding import (
    PLACEHOLDER_AMOUNT,
    PLACEHOLDER_CONTACT,
    PLACEHOLDER_DATE,
    PLACEHOLDER_REF,
    ground_draft,
)

TASK = (
    "Ask Meridian Supplies to quote for 25 height-adjustable desks against PO-4471, "
    "delivered by 30 April. Budget is £18,000. The contact is Priya."
)


# --- invented facts are replaced ----------------------------------------------------

def test_an_invented_reference_number_is_replaced():
    """The exact failure observed on the live model three runs running."""
    out = ground_draft("Quote submission — PO-94103", "Hi Priya,\n\nPlease quote.", TASK)
    assert PLACEHOLDER_REF in out.subject
    assert "PO-94103" not in out.subject
    assert out.count == 1


def test_an_invented_deadline_is_replaced():
    out = ground_draft(None, "We need the pricing by 12 April.", TASK)
    assert PLACEHOLDER_DATE in out.body
    assert "12 April" not in out.body


def test_an_invented_amount_is_replaced():
    out = ground_draft(None, "Our budget is £24,500 for this.", TASK)
    assert PLACEHOLDER_AMOUNT in out.body
    assert "24,500" not in out.body


def test_an_invented_contact_name_is_replaced():
    out = ground_draft(None, "Hi Sam,\n\nPlease quote.", TASK)
    assert f"Hi {PLACEHOLDER_CONTACT}," in out.body
    assert "Sam" not in out.body


def test_the_greeting_word_survives_because_it_is_style():
    """structural.greeting is a profile field. Replacing 'Hi' along with the name would
    destroy the habit being reproduced."""
    assert ground_draft(None, "Hi Sam,\n\nx", TASK).body.startswith("Hi ")
    assert ground_draft(None, "Dear Sam,\n\nx", TASK).body.startswith("Dear ")


# --- supplied facts are left alone ---------------------------------------------------

def test_a_reference_from_the_task_survives():
    out = ground_draft("Desks — PO-4471", "Hi Priya,\n\nAgainst PO-4471.", TASK)
    assert "PO-4471" in out.subject
    assert "PO-4471" in out.body
    assert out.clean


def test_a_date_from_the_task_survives():
    assert "30 April" in ground_draft(None, "Delivery by 30 April.", TASK).body


def test_an_amount_from_the_task_survives():
    assert "£18,000" in ground_draft(None, "Budget is £18,000.", TASK).body


def test_a_contact_name_from_the_task_survives():
    out = ground_draft(None, "Hi Priya,\n\nPlease quote.", TASK)
    assert "Hi Priya," in out.body
    assert out.clean


def test_punctuation_differences_do_not_break_grounding():
    """A fact is grounded on whether it was supplied, not on how it was punctuated."""
    assert ground_draft(None, "Against PO 4471.", TASK).clean
    assert ground_draft(None, "Delivery by 30th April.", TASK).clean


def test_ordinary_prose_is_untouched():
    body = "Hi Priya,\n\nWe're refreshing the second floor and need pricing.\n\nJo"
    assert ground_draft(None, body, TASK).body == body


def test_bare_quantities_are_not_treated_as_references():
    """'25 desks' is a quantity from the task, not an identifier."""
    assert ground_draft(None, "We need 25 desks.", TASK).clean


# --- reporting ------------------------------------------------------------------------

def test_every_replacement_is_reported():
    out = ground_draft(
        "Quote — PO-99999",
        "Hi Sam,\n\nWe need this by 12 April at £24,500.",
        TASK,
    )
    assert out.count == 4
    originals = [o for o, _ in out.replacements]
    assert "PO-99999" in originals
    assert "Sam" in originals
    assert any("12 April" in o for o in originals)
    assert any("24,500" in o for o in originals)


def test_a_clean_draft_reports_nothing():
    out = ground_draft("Desks — PO-4471", "Hi Priya,\n\nBy 30 April, budget £18,000.", TASK)
    assert out.clean
    assert out.count == 0


# --- ordering and edge cases -----------------------------------------------------------

def test_money_is_claimed_before_references():
    """'$1,720' must not be read as an identifier."""
    out = ground_draft(None, "The invoice shows $1,720.", TASK)
    assert PLACEHOLDER_AMOUNT in out.body
    assert PLACEHOLDER_REF not in out.body


@pytest.mark.parametrize("body", ["", None])
def test_degenerate_input_does_not_raise(body):
    assert ground_draft(None, body or "", TASK).body == ""


def test_an_empty_task_grounds_nothing_and_replaces_everything():
    """With no task there is nothing to have supplied a fact, so every fact is invented.
    Aggressive, and correct: the bias is to replace."""
    out = ground_draft("PO-4471", "Hi Priya,\n\nBy 30 April.", "")
    assert out.count == 3
