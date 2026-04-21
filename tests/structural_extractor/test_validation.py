from src.services.structural_extractor.parsing.model import BBox, ParsedDocument, Token
from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.validation import (
    ValidationReport,
    verify_anchors,
    verify_math,
)


def _ev(v):
    return ExtractedValue(value=v, provenance="extracted", source="structural", attempt=1)


# ------------------ anchor tests ------------------

def test_verify_anchors_passes_when_value_matches_source_token():
    tok = Token(text="INV600254", anchor=BBox(1, 0, 0, 100, 20), order=0)
    doc = ParsedDocument(
        source_format="pdf",
        filename="",
        tokens=[tok],
        regions=[],
        tables=[],
        pages_or_sheets=1,
        full_text="INV600254",
        raw_bytes=b"",
    )
    header = {
        "invoice_id": ExtractedValue(
            value="INV600254",
            provenance="extracted",
            anchor_text="INV600254",
            anchor_ref=tok.anchor,
            source="structural",
            attempt=1,
        )
    }
    rep = verify_anchors(header, doc)
    assert rep.passed


def test_verify_anchors_passes_for_derived_values():
    header = {
        "due_date": ExtractedValue(
            value="2019-11-20",
            provenance="derived",
            derivation_trace={"rule_id": "x", "inputs": {}},
            source="derivation_registry",
            attempt=1,
        )
    }
    doc = ParsedDocument(
        source_format="pdf",
        filename="",
        tokens=[],
        regions=[],
        tables=[],
        pages_or_sheets=1,
        full_text="",
        raw_bytes=b"",
    )
    rep = verify_anchors(header, doc)
    assert rep.passed


def test_verify_anchors_fails_when_anchor_text_absent():
    tok = Token(text="FOO", anchor=BBox(1, 0, 0, 100, 20), order=0)
    doc = ParsedDocument(
        source_format="pdf",
        filename="",
        tokens=[tok],
        regions=[],
        tables=[],
        pages_or_sheets=1,
        full_text="Some other text",
        raw_bytes=b"",
    )
    header = {
        "invoice_id": ExtractedValue(
            value="HALLUCINATED",
            provenance="extracted",
            anchor_text="HALLUCINATED",
            anchor_ref=tok.anchor,
            source="structural",
            attempt=1,
        )
    }
    rep = verify_anchors(header, doc)
    assert not rep.passed
    assert any("anchor_text" in f for f in rep.failures)


def test_validation_report_default():
    rep = ValidationReport(passed=True)
    assert rep.failures == []


# ------------------ math tests ------------------

def test_math_reconciles():
    header = {
        "invoice_amount": _ev(8333.0),
        "tax_amount": _ev(1666.60),
        "invoice_total_incl_tax": _ev(9999.60),
    }
    rep = verify_math(header, [])
    assert rep.passed


def test_math_fails_on_mismatch():
    header = {
        "invoice_amount": _ev(8000.0),
        "tax_amount": _ev(1000.0),
        "invoice_total_incl_tax": _ev(9999.60),
    }
    rep = verify_math(header, [])
    assert not rep.passed


def test_line_item_math_reconciles():
    header = {
        "invoice_amount": _ev(1599.80),
        "tax_amount": _ev(0.0),
        "invoice_total_incl_tax": _ev(1599.80),
    }
    items = [
        {"quantity": _ev(10), "unit_price": _ev(79.99), "line_total": _ev(799.90)},
        {"quantity": _ev(10), "unit_price": _ev(79.99), "line_total": _ev(799.90)},
    ]
    rep = verify_math(header, items)
    assert rep.passed
