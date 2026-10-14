"""A contract's title, governing law and payment terms, read from its own words.

The last three fields of the pattern-less group (audit 2026-10-04): nine, four and
four `canonical_labels` between them, no `patterns`, and NULL on all 7 live
contract rows while the documents state them:

    ## FRAMEWORK AGREEMENT
    This Framework Agreement is governed by the laws of England and Wales.
    Payment shall be made within 30 days of receipt of a valid invoice.

THREE TRAPS, all three present in the live documents and all three tested:

1. "governed by Framework Agreement No. FA-2026-0042" appears on TWO of the seven.
   `governed by` followed by another agreement is not a choice of law — the same
   cited-agreement trap that gave an order form its framework's start date. So
   "the laws of" (or "<Adjective> law") is mandatory.
2. The real Marketing Agreement states "will provide an invoice to the Client
   every 30 days" (an invoicing cadence) and "without amending it within a period
   of 10 business days" (a cure period). Neither is a payment term. So the day
   count must hang off payment wording, not off a bare "within N days".
3. docling DROPPED that document's title: its parsed text begins "## PARTIES".
   A section heading is not a title, so the heading must name a contract-family
   thing — and the document's own first sentence ("This Marketing Agreement ...")
   is the second source, which is the only one that works for that file.

Offline and model-free.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.extraction.engineered.contract_header import (  # noqa: E402
    header_candidates, read_header,
)

#: The real document's parsed opening -- note it begins with a SECTION heading,
#: because docling dropped the title.
REAL = (
    "## PARTIES\n\nThis Marketing Agreement (hereinafter referred to as the "
    "' Agreement') is entered into on June 12, 2025 by and between BrightWave "
    "Digital Ltd. and NexaSpark Marketing Ltd.\n\n"
    "## PAYMENT AND FEES\n\nThe Parties agree that the Marketer will provide an "
    "invoice to the Client every 30 days upon the completion of the Services.\n\n"
    "## TERMINATION\n\nThis Agreement will be terminated immediately if one of the "
    "Parties breaches a condition set forth in this Agreement without amending it "
    "within a period of 10 business days.\n\n"
    "## ALTERNATIVE DISPUTE RESOLUTION\n\nAny dispute shall be submitted to "
    "Arbitration, in accordance with and subject to the laws of England and Wales.\n"
)

#: A synthetic document's parsed opening, where the title DID survive.
SYNTHETIC = (
    "## FRAMEWORK AGREEMENT\n\nFramework Agreement No. FA-2026-0077 "
    "Effective Date: 5 January 2026\n\n"
    "## 3. PAYMENT TERMS\n\nPayment shall be made within 30 days of receipt of a "
    "valid invoice.\n\n"
    "## 4. GOVERNING LAW\n\nThis Framework Agreement is governed by the laws of "
    "England and Wales.\n"
)


# --------------------------------------------------------------------------
# contract_title
# --------------------------------------------------------------------------

def test_the_documents_own_heading_is_the_title():
    assert read_header(SYNTHETIC).title == "Framework Agreement"


def test_a_dropped_title_is_recovered_from_the_documents_first_sentence():
    """THE case for the real file: docling dropped the title, so the heading is
    "PARTIES" -- but the document names itself in its opening sentence."""
    assert read_header(REAL).title == "Marketing Agreement"


def test_a_section_heading_is_not_a_title():
    for heading in ("## PARTIES", "## PAYMENT AND FEES", "## 5. SIGNATURES",
                    "## TERMINATION", "## 1. PURPOSE"):
        h = read_header(f"{heading}\n\nSome clause text that states nothing else.\n")
        assert h.title is None, heading


@pytest.mark.parametrize("heading,title", [
    ("## ORDER FORM", "Order Form"),
    ("## SERVICE AGREEMENT", "Service Agreement"),
    ("## MASTER SERVICES AGREEMENT", "Master Services Agreement"),
    ("## STATEMENT OF WORK", "Statement Of Work"),
    ("## NON-DISCLOSURE AGREEMENT", "Non-Disclosure Agreement"),
    ("# Framework Agreement", "Framework Agreement"),
])
def test_each_contract_family_heading_is_read_and_title_cased(heading, title):
    """ALL CAPS in a heading is typography, not the contract's name;
    proc.bp_contract_master holds Title Case."""
    assert read_header(f"{heading}\n\nThis document is made today.\n").title == title


def test_a_labelled_title_wins_over_the_heading():
    text = "## ORDER FORM\n\nContract Title: SEO Retainer 2026\n"
    assert read_header(text).title == "SEO Retainer 2026"


# --------------------------------------------------------------------------
# governing_law
# --------------------------------------------------------------------------

def test_the_governing_law_is_read():
    assert read_header(SYNTHETIC).governing_law == "England and Wales"


def test_the_law_is_read_from_a_dispute_clause_too():
    assert read_header(REAL).governing_law == "England and Wales"


def test_a_cited_agreement_is_not_a_governing_law():
    """THE trap, live on two of the seven documents. "governed by Framework
    Agreement No. FA-2026-0042" is an incorporation clause, not a choice of law."""
    text = ("## ORDER FORM\n\nThis Order Form is incorporated into and governed by "
            "Framework Agreement No. FA-2026-0042 dated 5 January 2026.\n")
    assert read_header(text).governing_law is None, read_header(text)


@pytest.mark.parametrize("sentence,law", [
    ("This Agreement is governed by the laws of England and Wales.", "England and Wales"),
    ("This Agreement shall be governed by and construed in accordance with the "
     "laws of Scotland.", "Scotland"),
    ("This Agreement is subject to the laws of the State of New York.",
     "the State of New York"),
    ("This Agreement shall be governed by English law.", "English law"),
    ("Governing Law: England and Wales", "England and Wales"),
    ("Applicable Law: Germany", "Germany"),
])
def test_each_law_shape_is_read(sentence, law):
    assert read_header(sentence).governing_law == law, sentence


def test_a_document_that_names_no_law_yields_nothing():
    assert read_header("## ORDER FORM\n\nTotal charges GBP 60,000.\n").governing_law is None


# --------------------------------------------------------------------------
# payment_terms
# --------------------------------------------------------------------------

def test_the_payment_terms_are_read_as_a_day_count():
    """The house format is the document's own phrasing lightly normalised --
    proc.bp_purchase_order_raw holds "Annual in advance, 30 days" and
    proc.bp_invoice_stg holds "30 days - due 30 Jul 2025". "30 days" is the
    comparable core of both."""
    assert read_header(SYNTHETIC).payment_terms == "30 days"


def test_an_invoicing_cadence_is_not_a_payment_term():
    """THE trap on the real document: "will provide an invoice to the Client every
    30 days" is how often it invoices, not when payment falls due. It also states
    a 10-business-day CURE period, which is not a payment term either."""
    assert read_header(REAL).payment_terms is None, read_header(REAL)


def test_a_cure_period_is_not_a_payment_term():
    text = ("This Agreement will terminate if a Party breaches it without amending "
            "the breach within a period of 10 business days.\n")
    assert read_header(text).payment_terms is None


@pytest.mark.parametrize("sentence,terms", [
    ("Payment shall be made within 30 days of receipt of a valid invoice.", "30 days"),
    ("Payment is due within 45 days of the invoice date.", "45 days"),
    ("The Client shall pay within 14 days.", "14 days"),
    ("Invoices are payable within 60 days.", "60 days"),
    ("Payment Terms: Net 30", "Net 30"),
    ("Terms of Payment: 45 days from invoice date", "45 days from invoice date"),
    ("Net Terms: Net 60", "Net 60"),
])
def test_each_payment_shape_is_read(sentence, terms):
    assert read_header(sentence).payment_terms == terms, sentence


def test_a_document_with_no_payment_wording_yields_nothing():
    assert read_header("## ORDER FORM\n\nTotal charges GBP 60,000.\n").payment_terms is None


# --------------------------------------------------------------------------
# candidates and wiring
# --------------------------------------------------------------------------

def test_candidates_carry_the_schemas_field_names():
    by_field = {c.field: c.value for c in header_candidates(SYNTHETIC)}
    assert by_field == {"contract_title": "Framework Agreement",
                        "governing_law": "England and Wales",
                        "payment_terms": "30 days"}, by_field


def test_only_what_the_document_states_is_emitted():
    by_field = {c.field: c.value for c in header_candidates(REAL)}
    assert by_field == {"contract_title": "Marketing Agreement",
                        "governing_law": "England and Wales"}, by_field


def test_nothing_is_emitted_for_a_silent_document():
    assert header_candidates("## ORDER FORM\n\nTotal GBP 60,000.\n") != []  # the title IS stated
    assert header_candidates("Some text with no title, law or payment wording.\n") == []


def test_the_fields_reach_dispatchs_candidate_set():
    from src.services.extraction.dispatch import _contract_party_candidates
    cands, _barred = _contract_party_candidates("contract", SYNTHETIC)
    by_field = {c.field: c.value for c in cands}
    assert by_field.get("contract_title") == "Framework Agreement", by_field
    assert by_field.get("governing_law") == "England and Wales", by_field
    assert by_field.get("payment_terms") == "30 days", by_field


# ---------------------------------------------------------------------------
# Found on a live document 2026-10-04: promoting_signed.pdf took its title from
# its SIGNATURE BLOCK -- "Title: Managing Director Date: 1 March 2026 CLIENT
# Name: Tom Okafor Title: Head of Procurement..." -- because contract.yaml lists
# a bare "Title" among contract_title's canonical_labels, and in a signature
# block "Title:" is the job title. The same word, two meanings, and the
# signature block is where it appears most.
# ---------------------------------------------------------------------------

SIGNED = (
    "## SERVICE AGREEMENT\n\nContract No. SA-2026-0310\n\n"
    "## 5. SIGNATURES\n\nSUPPLIER Name: Priya Raman Title: Managing Director "
    "Date: 1 March 2026 CLIENT Name: Tom Okafor Title: Head of Procurement "
    "Date: 1 March 2026\n"
)


def test_a_signature_blocks_job_title_is_not_the_contract_title():
    h = read_header(SIGNED)
    assert h.title == "Service Agreement", h


def test_a_labelled_title_is_still_read_when_it_is_unambiguous():
    """"Contract Title:" and "Agreement Name:" mean one thing. Only the bare
    "Title:" was dropped."""
    for label in ("Contract Title", "Agreement Title", "Contract Name",
                  "Agreement Name", "Subject"):
        h = read_header(f"## ORDER FORM\n\n{label}: SEO Retainer 2026\n")
        assert h.title == "SEO Retainer 2026", label


def test_a_labelled_title_stops_at_the_next_field():
    """The parser puts a header block on one line, so a title's value has to be
    bounded the same way every other label's is."""
    h = read_header("Contract Title: SEO Retainer 2026 Effective Date: 1 March 2026\n")
    assert h.title == "SEO Retainer 2026", h


# ---------------------------------------------------------------------------
# The backfill decision. Three rules, as for the term and the value: the sweep
# never produced these either, so there is no wrong value of its to clear.
# ---------------------------------------------------------------------------

def _decide(**kw):
    from src.services.extraction.engineered.contract_header import decide_header_correction
    return decide_header_correction(**kw)


def test_an_empty_row_is_filled_from_the_document():
    d = _decide(full_text=SYNTHETIC, stored_title=None, stored_law=None,
                stored_payment=None, provenance_source=None)
    assert (d.title, d.governing_law, d.payment_terms) == (
        "Framework Agreement", "England and Wales", "30 days"), d
    assert d.changed is True


def test_a_row_already_correct_is_left_alone():
    d = _decide(full_text=SYNTHETIC, stored_title="Framework Agreement",
                stored_law="England and Wales", stored_payment="30 days",
                provenance_source="regex")
    assert d.changed is False, d


def test_a_human_confirmed_value_is_never_touched():
    d = _decide(full_text=SYNTHETIC, stored_title="Something Else", stored_law=None,
                stored_payment=None, provenance_source="hitl")
    assert d.changed is False and d.title == "Something Else", d


def test_a_field_the_reader_cannot_find_is_not_cleared():
    """The real contract states no payment terms. A stored value there came from
    somewhere else and this reader has no standing to delete it."""
    d = _decide(full_text=REAL, stored_title=None, stored_law=None,
                stored_payment="45 days", provenance_source="context_layer")
    assert d.payment_terms == "45 days", d


def test_nothing_readable_changes_nothing():
    d = _decide(full_text="Some text with no title, law or payment wording.\n",
                stored_title=None, stored_law=None, stored_payment=None,
                provenance_source=None)
    assert d.changed is False, d
