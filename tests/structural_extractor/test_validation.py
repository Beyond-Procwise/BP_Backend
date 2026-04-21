from src.services.structural_extractor.parsing.model import BBox, ParsedDocument, Token
from src.services.structural_extractor.types import ExtractedValue
from src.services.structural_extractor.validation import (
    ValidationReport,
    verify_anchors,
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
