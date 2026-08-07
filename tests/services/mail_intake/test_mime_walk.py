"""MIME walking, including the two cases that decide whether a real quote is
seen at all: attachments buried in a forwarded chain, and the inline/CID
distinction that separates a signature logo from a document."""
import pathlib

import pytest

from src.services.mail_intake.mime_walk import (
    MalformedMime,
    drop_cid_referenced,
    parse,
)

FIXTURES = pathlib.Path("tests/fixtures/mail_intake")


def _load(name):
    return (FIXTURES / name).read_bytes()


def test_simple_message_yields_one_attachment_and_the_body():
    bundle = parse(_load("simple_pdf.eml"))
    assert bundle.body_text.startswith("Our quote is attached")
    assert len(bundle.parts) == 1
    part = bundle.parts[0]
    assert part.filename == "quote-QT-4471.pdf"
    assert part.declared_mime == "application/pdf"
    assert part.content.startswith(b"%PDF-1.4")
    assert part.nesting_depth == 0
    assert part.disposition == "attachment"


def test_nested_quote_two_levels_down_is_recovered_with_its_depth():
    bundle = parse(_load("forwarded_two_levels.eml"))
    pdfs = [p for p in bundle.parts if p.declared_mime == "application/pdf"]
    assert len(pdfs) == 1
    assert pdfs[0].filename == "quote-QT-9002.pdf"
    assert pdfs[0].nesting_depth == 2


def test_recursion_stops_at_max_depth():
    bundle = parse(_load("forwarded_two_levels.eml"), max_depth=1)
    assert not [p for p in bundle.parts if p.declared_mime == "application/pdf"]


def test_cid_referenced_inline_image_is_dropped():
    bundle = parse(_load("cid_signature_logo.eml"))
    assert any(p.declared_mime == "image/png" for p in bundle.parts)
    kept = drop_cid_referenced(bundle)
    assert not any(p.declared_mime == "image/png" for p in kept)


def test_inline_part_not_referenced_by_cid_is_retained():
    bundle = parse(_load("inline_non_cid_document.eml"))
    kept = drop_cid_referenced(bundle)
    assert [p.filename for p in kept] == ["scan.pdf"]


def test_body_and_attachment_arrive_in_the_same_bundle():
    # Acceptance test 17: validity window in the note, pricing in the file.
    bundle = parse(_load("body_validity_window.eml"))
    assert "valid until 30 September 2026" in bundle.body_text
    assert [p.filename for p in bundle.parts] == ["quote-QT-5510.pdf"]


def test_malformed_mime_raises_rather_than_returning_partial_results():
    with pytest.raises(MalformedMime):
        parse(_load("malformed.eml"))


def test_headers_are_preserved_for_provenance():
    bundle = parse(_load("simple_pdf.eml"))
    assert bundle.headers["from"] == ["Copperleaf Sales <quotes@copperleaf.test>"]
    assert bundle.headers["message-id"] == ["<quote-qt-4471@copperleaf.test>"]
