"""A 'matched' result must show at least one reason for the type that WON.

The per-candidate evidence reservation used to run only for ties, so a page
that repeated its declared type's wording more than twelve times returned
twelve spans, all for the declared type and none for the winner.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.vocabulary import SEED_VOCABULARY  # noqa: E402
from src.services.extraction.type_resolver import resolve_document_type  # noqa: E402

V = SEED_VOCABULARY
NOISY = " ".join(f"Against purchase order PO{i}." for i in range(14)) + "\nINVOICE\n"


def test_matched_disagreement_carries_a_span_for_the_winner():
    r = resolve_document_type(
        declared_concept="doctype.order", full_text=NOISY, vocabulary=V)
    assert (r.status, r.agreement, r.evidence_concept) == (
        "matched", "disagreed", "doctype.invoice")
    assert any(e.concept_code == "doctype.invoice" for e in r.evidence), (
        "a review row must show a reason for the type that won"
    )


def test_matched_evidence_stays_verbatim_and_bounded():
    r = resolve_document_type(
        declared_concept="doctype.order", full_text=NOISY, vocabulary=V)
    assert len(r.evidence) <= 12
    for e in r.evidence:
        assert NOISY[e.start:e.start + len(e.text)] == e.text


def test_agreeing_match_is_unchanged_by_the_reservation():
    page = "TAX INVOICE\nInvoice No: INV-1\nAmount due: 10.00\n"
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=page, vocabulary=V)
    assert r.agreement == "agreed"
    assert r.evidence and all(e.concept_code == "doctype.invoice" for e in r.evidence)
