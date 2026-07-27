"""The redactor is the last point at which correspondence can be stopped from reaching
an LLM prompt. These tests are mostly about what survives, and what must not.

The bias under test is over-redaction: removing too much costs signal, removing too
little leaks someone's supplier names and prices.
"""

from __future__ import annotations

import pytest

from services.style.redaction import (
    AMOUNT,
    EMAIL,
    NAME,
    ORG,
    PHONE,
    REF,
    URL,
    redact,
)

REAL_EMAIL = """Hi Sam,

Thanks for turning the racking quote around so quickly. Two things before I take it to
finance: the unit rate on line 3 is £1,240.50 against £1,180 on our last order, and the
lead time has gone from 10 to 15 days. Can you tell me what's driving both?

Quote reference QUT136586 if you need it. Happy to talk Thursday morning.

Kind regards,

Nicholas Geelen
Procurement Lead | Techworld Solutions Ltd
n.geelen@techworld.example.com | 020 7946 0958
www.techworld.example.com

This email and any attachments are confidential and may be legally privileged.
"""


def test_the_shape_survives():
    """What the profile learns must still be there: greeting, paragraphing, sign-off."""
    out = redact(REAL_EMAIL).text
    assert out.startswith("Hi ")
    assert "Kind regards," in out
    assert "\n\n" in out  # paragraph breaks preserved


@pytest.mark.parametrize("leaked", [
    "Sam", "Nicholas", "Geelen", "Techworld",
    "1,240.50", "1,180", "QUT136586",
    "n.geelen@techworld.example.com", "020 7946 0958",
    "www.techworld.example.com",
])
def test_nothing_identifying_survives(leaked):
    assert leaked not in redact(REAL_EMAIL).text


def test_placeholders_are_typed():
    out = redact(REAL_EMAIL).text
    for placeholder in (NAME, AMOUNT, REF):
        assert placeholder in out, placeholder


def test_the_signature_block_goes_but_the_sign_off_habit_stays():
    """'Kind regards, Nick' versus 'Best,' is a style signal. The job title, phone number
    and company under it are not."""
    out = redact(REAL_EMAIL).text
    assert "Kind regards," in out
    assert "Procurement Lead" not in out
    assert PHONE not in out.split("Kind regards,")[0]  # contact detail is gone entirely


def test_the_disclaimer_goes():
    assert "confidential and may be legally privileged" not in redact(REAL_EMAIL).text
    assert "disclaimer" in redact(REAL_EMAIL).blocks_removed


# --- quoted chains -----------------------------------------------------------------

def test_a_quoted_reply_chain_is_dropped_wholesale():
    """A reply chain is someone else's writing sitting inside this one — not this
    writer's style, and the largest single body of content in a typical thread."""
    body = (
        "Hi Dana,\n\nThat works, let's go with the Tuesday slot.\n\nBest,\nNick\n\n"
        "On Tue, 14 Mar 2026 at 09:12, Dana Okonjo <dana@supplier.example> wrote:\n"
        "> We can do Tuesday or Wednesday. Our rate is £940 per pallet position.\n"
        "> Let me know which suits.\n"
    )
    out = redact(body).text
    assert "pallet position" not in out
    assert "940" not in out
    assert "Wednesday" not in out
    assert "Tuesday slot" in out or "Tuesday" in out  # the writer's own words remain


def test_outlook_style_forwarded_headers_are_dropped():
    body = (
        "Hi Priya,\n\nPassing this on.\n\nThanks,\nNick\n\n"
        "-----Original Message-----\n"
        "From: Ade Balogun\nSent: 14 March 2026\nTo: Nick\nSubject: Racking\n\n"
        "Our best price is £8,400 all in.\n"
    )
    out = redact(body).text
    assert "8,400" not in out
    assert "Ade" not in out
    assert "original_message_separator" in redact(body).blocks_removed


def test_bare_quoted_lines_are_dropped():
    out = redact("Hi Sam,\n\nAgreed.\n\n> our price is £500\n> firm for 30 days\n").text
    assert "500" not in out
    assert "firm for 30 days" not in out


# --- what must NOT be redacted -----------------------------------------------------

def test_dates_and_weekdays_survive():
    """date_format is a profile field. Replacing dates would destroy the habit being
    learned."""
    out = redact("Hi Sam,\n\nCan you confirm by Thursday? Otherwise Monday works.\n\nNick").text
    assert "Thursday" in out
    assert "Monday" in out


def test_bare_quantities_survive():
    """'40 chairs' tells you about number_format and identifies nobody."""
    out = redact("Hi Sam,\n\nWe need 40 chairs and 12 desks.\n\nNick").text
    assert "40" in out and "12" in out


def test_procurement_vocabulary_survives():
    """preferred_terms is learned from exactly these words — redacting them would strip
    the vocabulary the profile exists to record."""
    out = redact(
        "Hi Sam,\n\nPlease confirm the unit rate and volume tiers, plus payment terms "
        "and lead time.\n\nNick"
    ).text
    for term in ("unit rate", "volume tiers", "payment terms", "lead time"):
        assert term in out, term


def test_contractions_and_register_survive():
    """contractions is a profile field, so 'what's' must not be mangled."""
    out = redact("Hi Sam,\n\nWhat's driving the increase? I'd rather not guess.\n\nNick").text
    assert "What's" in out and "I'd" in out


# --- entity coverage ---------------------------------------------------------------

@pytest.mark.parametrize("text,placeholder", [
    ("The total came to £1,240.50 net.", AMOUNT),
    ("The total came to 1,240.50 GBP net.", AMOUNT),
    ("The total came to $8,400 net.", AMOUNT),
    ("Please quote against PO-44821 today.", REF),
    ("Please quote against QUT136586 today.", REF),
    ("Reach me on 020 7946 0958 tomorrow.", PHONE),
    ("Mail me at sam.rees@supplier.example tomorrow.", EMAIL),
    ("Terms at https://example.com/terms apply.", URL),
    ("We buy through Dixon Reynolds Ltd normally.", ORG),
    ("We buy through Meridian Holdings normally.", ORG),
])
def test_entity_patterns(text, placeholder):
    assert placeholder in redact(text).text


def test_organisations_are_claimed_before_names():
    """Ordering matters: 'Techworld Ltd' must become [ORG], not '[NAME] Ltd'."""
    out = redact("We moved to Techworld Solutions Ltd last year.").text
    assert ORG in out
    assert "Ltd" not in out


def test_counts_are_reported_for_audit():
    """The tally is safe to log. The input never is."""
    result = redact(REAL_EMAIL)
    assert result.total_replacements > 0
    assert set(result.counts) <= {"name", "org", "amount", "ref", "email", "phone", "url"}


# --- robustness --------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", None, "   ", 12345])
def test_degenerate_input_does_not_raise(value):
    assert redact(value).text.strip() == ""


def test_subject_is_included_when_supplied():
    out = redact("Hi Sam,\n\nAll fine.\n\nNick", subject="Racking quote — QUT136586").text
    assert out.startswith("Subject:")
    assert "QUT136586" not in out


def test_an_email_that_is_only_a_quoted_chain_redacts_to_nothing():
    """The compiler drops these before counting toward min_exemplars."""
    body = "On Tue, 14 Mar 2026 at 09:12, Dana <d@x.example> wrote:\n> our price is £500\n"
    assert redact(body).text.strip() == ""
