"""A disagreement or an unknown type reaches the queue a buyer works — and
never blocks the document.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.services.concepts.vocabulary import SEED_VOCABULARY  # noqa: E402
from src.services.extraction.type_resolver import (  # noqa: E402
    resolve_document_type,
    type_resolution_discrepancies,
)

V = SEED_VOCABULARY
INVOICE_PAGE = "TAX INVOICE\nInvoice No: INV-1\nAmount due: 10.00\n"
BLANK_PAGE = "Dear Sir or Madam,\n\nPlease find attached.\n"


def test_agreement_produces_no_review_item():
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=INVOICE_PAGE, vocabulary=V)
    assert type_resolution_discrepancies(r) == []


def test_declared_only_produces_no_review_item():
    """A silent page is not a problem. The uploader said what it is."""
    r = resolve_document_type(
        declared_concept="doctype.invoice", full_text=BLANK_PAGE, vocabulary=V)
    assert type_resolution_discrepancies(r) == []


def test_disagreement_produces_one_non_blocking_item_with_both_readings():
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V)
    items = type_resolution_discrepancies(r)
    assert len(items) == 1
    d = items[0]
    assert d.issue_type == "document_type_disagreement"
    assert d.blocks_promotion is False, (
        "a reporting improvement must not stop live ingestion"
    )
    assert d.raw_value == "doctype.quote"
    assert d.expected_value == "doctype.invoice"
    assert d.evidence_text and d.evidence_text in INVOICE_PAGE


def test_unknown_type_produces_one_non_blocking_item():
    r = resolve_document_type(declared_concept=None, full_text=BLANK_PAGE, vocabulary=V)
    items = type_resolution_discrepancies(r)
    assert len(items) == 1
    assert items[0].issue_type == "unknown_document_type"
    assert items[0].blocks_promotion is False


def test_unresolved_item_names_every_candidate():
    """A tie must reach a human with both options, not with one of them.

    (The plan's version wrapped these asserts in ``if r.status == "unresolved"``
    and used a page that is NOT a tie, so it asserted nothing.)"""
    r = resolve_document_type(
        declared_concept=None, full_text="INVOICE / QUOTE\n", vocabulary=V)
    assert r.status == "unresolved"
    items = type_resolution_discrepancies(r)
    assert len(items) == 1
    assert items[0].issue_type == "unresolved_document_type"
    assert items[0].blocks_promotion is False
    assert set(r.candidates) == {"doctype.invoice", "doctype.quote"}
    for candidate in r.candidates:
        assert candidate in (items[0].computed_value or "")
    # and a span for each side is shown, verbatim
    assert "|" in items[0].evidence_text


def test_every_item_has_a_field_name_so_the_dedup_index_works():
    """The open-row identity is (doc_type, doc_pk_candidate, issue_type,
    field_name). A NULL field_name coalesces to '' and still works, but a
    stable value keeps two different type findings apart."""
    seen = 0
    for declared, page in (("doctype.quote", INVOICE_PAGE), (None, BLANK_PAGE),
                           (None, "INVOICE / QUOTE\n")):
        r = resolve_document_type(
            declared_concept=declared, full_text=page, vocabulary=V)
        for d in type_resolution_discrepancies(r):
            seen += 1
            assert d.field_name == "document_type"
    assert seen == 3


def test_tie_with_a_declared_type_still_reaches_a_human():
    """agreement alone says 'declared_only' here; status says otherwise."""
    r = resolve_document_type(
        declared_concept="doctype.sow", full_text="INVOICE / QUOTE\n", vocabulary=V)
    assert (r.status, r.agreement) == ("unresolved", "declared_only")
    items = type_resolution_discrepancies(r)
    assert [i.issue_type for i in items] == ["unresolved_document_type"]
    assert items[0].raw_value == "doctype.sow"
    assert items[0].blocks_promotion is False


def test_disagreement_evidence_is_the_winners_span_not_the_declared_ones():
    page = " ".join(f"Against purchase order PO{i}." for i in range(14)) + "\nINVOICE\n"
    r = resolve_document_type(
        declared_concept="doctype.order", full_text=page, vocabulary=V)
    (d,) = type_resolution_discrepancies(r)
    assert d.evidence_text.lower() == "invoice"


def test_notes_carry_every_evidence_span_unclipped():
    r = resolve_document_type(
        declared_concept="doctype.quote", full_text=INVOICE_PAGE, vocabulary=V)
    (d,) = type_resolution_discrepancies(r)
    for ev in r.evidence:
        assert repr(ev.text) in d.notes


def test_same_shape_documents_share_one_group_key():
    """Twelve workbooks with the same 'Order Form' cell are ONE decision."""
    keys = set()
    seen_items = []
    for body in ("Order Form\nSupplier: A\n",
                 "Quotation reference 99\nOrder Form\nSupplier: B\nTotal 5\n"):
        r = resolve_document_type(
            declared_concept="doctype.quote", full_text=body, vocabulary=V)
        for d in type_resolution_discrepancies(r):
            seen_items.append(d)
            keys.add((d.issue_type, d.raw_value, d.expected_value,
                      d.notes.split(" Evidence:")[0]))
    assert len(seen_items) == 2, "both documents must produce a finding"
    assert len(keys) == 1, keys
