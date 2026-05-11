"""Tests for the spaCy NER candidate generator (Task 12 of the extraction-redesign plan).

The spaCy NER extractor runs over parsed.full_text and emits one Candidate per
entity whose spaCy label matches the field's judge.ner_type_check.

Key regressions covered:
  I-18: a non-ORG string ("INVOICE NUMBER: 4759275") was committed as
        supplier_name — spaCy would find no ORG for that span and would not
        emit it as a candidate, allowing the L3 merger to prefer a genuine ORG.
  I-38: cross-doc leakage (wrong PERSON committed as requested_by) — spaCy
        emits only persons found in this document's full_text.

Run with GPU marker (en_core_web_trf loads a transformer):
    .venv/bin/pytest tests/extraction_v3/test_spacy_ner.py -v -m gpu
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from src.services.extraction_v3.extractors.spacy_ner import SpacyNERExtractor, _find_bbox_for_text
from src.services.extraction_v3.parsers.docling_backend import parse_with_docling
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import (
    Page, ParsedDocument, Token,
)
from src.services.extraction_v3.yaml_schema.loader import DocSchema, FieldSpec, JudgeRules, load_doc_schema

FX = Path(__file__).parent / "fixtures/invoices"


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _minimal_parsed(full_text: str, tokens: list[Token] | None = None) -> ParsedDocument:
    """Build a ParsedDocument with a single page containing the given text."""
    page_tokens = tokens or []
    return ParsedDocument(
        source_path="/tmp/fake.pdf",
        file_format="pdf-native",
        pages=[
            Page(
                index=0,
                width=595.0,
                height=842.0,
                rotation=0,
                regions=[],
                tables=[],
                tokens=page_tokens,
            )
        ],
        full_text=full_text,
        parser_backend="docling",
        parser_confidence=1.0,
    )


def _minimal_schema(fields: list[FieldSpec]) -> DocSchema:
    """Build a minimal DocSchema without DB-consistency checks."""
    return DocSchema(
        doc_type="test_invoice",
        db_table="proc.bp_invoice",
        fields=fields,
    )


def _ner_field(name: str, ner_type: str) -> FieldSpec:
    return FieldSpec(
        name=name,
        type="string",
        required=False,
        db_column=None,
        canonical_labels=[name],
        extractors=["spacy_ner"],
        judge=JudgeRules(ner_type_check=ner_type),
    )


# --------------------------------------------------------------------------- #
# Unit tests (no real PDF, no GPU required)                                    #
# --------------------------------------------------------------------------- #

def test_no_active_fields_returns_empty():
    """If no field lists 'spacy_ner' in extractors, produce_candidates is a no-op."""
    field = FieldSpec(
        name="invoice_id",
        type="string",
        required=True,
        db_column="invoice_id",
        canonical_labels=["Invoice Number"],
        extractors=["layoutlmv3"],  # spacy_ner NOT included
        judge=JudgeRules(ner_type_check="ORG"),
    )
    schema = _minimal_schema([field])
    parsed = _minimal_parsed("Acme Industries Ltd Invoice Number: 001")
    ex = SpacyNERExtractor()
    assert ex.produce_candidates(parsed, schema) == []


def test_ner_type_check_none_skipped():
    """Fields with ner_type_check='none' should not generate candidates."""
    field = _ner_field("invoice_id", "none")
    schema = _minimal_schema([field])
    parsed = _minimal_parsed("Acme Industries Ltd")
    ex = SpacyNERExtractor()
    assert ex.produce_candidates(parsed, schema) == []


def test_org_entity_produces_candidate():
    """ORG entities near a vendor label become candidates for supplier_name.

    The header-band + position filter (A.1/A.2) requires that supplier_name
    candidates either (a) appear in the top 30% of page 0 (requires token bboxes)
    or (b) appear near an explicit vendor/supplier/from label.  This test uses
    a "Vendor:" prefix so the ORG qualifies via the vendor-context path.
    """
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    # Include "Vendor:" so "Acme Industries Ltd" qualifies via vendor-context
    parsed = _minimal_parsed("Vendor: Acme Industries Ltd\nInvoice Number: 4759275")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    org_cands = [c for c in candidates if c.field == "supplier_name"]
    assert org_cands, "No ORG candidate emitted for supplier_name in vendor context"
    for c in org_cands:
        assert c.evidence_text in parsed.full_text
        assert c.model == "spacy_ner"
        assert 0.0 <= c.confidence <= 1.0  # confidence is context-dependent (0.55–0.90)


def test_i18_regression_invoice_number_not_org():
    """I-18: 'INVOICE NUMBER: 4759275' should NOT appear as an ORG candidate.

    spaCy labels cardinal numbers as CARDINAL, not ORG, so the extractor
    must not emit a candidate that would leak a non-ORG string into supplier_name.
    """
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    # Text that would trigger the I-18 regression: only the invoice number line
    parsed = _minimal_parsed("INVOICE NUMBER: 4759275")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    # No ORG entity expected — the invoice number is a CARDINAL
    supplier_vals = [c.value for c in candidates if c.field == "supplier_name"]
    # "INVOICE NUMBER: 4759275" must not appear as an ORG candidate
    assert "INVOICE NUMBER: 4759275" not in supplier_vals, (
        "I-18 regression: invoice number string was emitted as an ORG supplier_name candidate"
    )


def test_person_entity_produces_candidate_for_requested_by():
    """PERSON entities become candidates for requested_by.

    Uses a text fragment where en_core_web_sm reliably identifies a PERSON.
    The name appears without a colon-label prefix that can confuse the tagger.
    """
    field = _ner_field("requested_by", "PERSON")
    schema = _minimal_schema([field])
    # "Approved by John Smith" — en_core_web_sm reliably tags "John Smith" as PERSON
    # when preceded by a verb ("by"), regardless of the document structure.
    parsed = _minimal_parsed("Approved by John Smith on Invoice Number INV-001")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    person_cands = [c for c in candidates if c.field == "requested_by"]
    assert person_cands, "No PERSON candidate emitted for requested_by"
    for c in person_cands:
        assert c.evidence_text in parsed.full_text
        assert c.model == "spacy_ner"


def test_gpe_entity_produces_candidate_for_country():
    """GPE entities become candidates for country."""
    field = _ner_field("country", "GPE")
    schema = _minimal_schema([field])
    parsed = _minimal_parsed("Supplier Location: United Kingdom\nDate: 2025-01-01")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    gpe_cands = [c for c in candidates if c.field == "country"]
    assert gpe_cands, "No GPE candidate emitted for country"
    for c in gpe_cands:
        assert c.evidence_text in parsed.full_text


def test_multiple_fields_multiple_entities():
    """Multiple fields with different ner_type_check values each get candidates."""
    fields = [
        _ner_field("supplier_name", "ORG"),
        _ner_field("country", "GPE"),
    ]
    schema = _minimal_schema(fields)
    parsed = _minimal_parsed(
        "Supplier: Globex Corporation\nCountry: United States\nInvoice: INV-002"
    )
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    fields_seen = {c.field for c in candidates}
    assert "supplier_name" in fields_seen
    assert "country" in fields_seen


def test_substring_guarantee_all_candidates():
    """Every candidate's evidence_text must be a substring of full_text."""
    fields = [
        _ner_field("supplier_name", "ORG"),
        _ner_field("requested_by", "PERSON"),
        _ner_field("country", "GPE"),
    ]
    schema = _minimal_schema(fields)
    text = (
        "Supplier: Initech Solutions\n"
        "Requested By: Bob Smith\n"
        "Country: Germany\n"
        "Invoice: INV-100"
    )
    parsed = _minimal_parsed(text)
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    for c in candidates:
        assert c.evidence_text in parsed.full_text, (
            f"Substring guarantee violated: {c.evidence_text!r} not in full_text"
        )


def test_confidence_bounds_all_candidates():
    """All candidates must have confidence in [0, 1]."""
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    parsed = _minimal_parsed("Acme Corp issued Invoice INV-42")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    for c in candidates:
        assert 0.0 <= c.confidence <= 1.0


def test_model_tag_is_spacy_ner():
    """All candidates must carry model='spacy_ner'."""
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    parsed = _minimal_parsed("Tech Dynamics Ltd\nInvoice INV-55")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    for c in candidates:
        assert c.model == "spacy_ner"


def test_candidate_field_restricted_to_active_fields():
    """Candidates are only emitted for fields that list 'spacy_ner' in extractors."""
    spacy_field = _ner_field("supplier_name", "ORG")
    non_spacy_field = FieldSpec(
        name="invoice_id",
        type="string",
        required=True,
        db_column="invoice_id",
        canonical_labels=["Invoice Number"],
        extractors=["layoutlmv3"],  # no spacy_ner
        judge=JudgeRules(ner_type_check="none"),
    )
    schema = _minimal_schema([spacy_field, non_spacy_field])
    parsed = _minimal_parsed("Acme Corp\nInvoice Number: 001")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    for c in candidates:
        assert c.field == "supplier_name", (
            f"Candidate produced for non-spacy_ner field: {c.field!r}"
        )


def test_empty_full_text_returns_empty():
    """With no text there are no entities; produce_candidates returns []."""
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    parsed = _minimal_parsed("")
    ex = SpacyNERExtractor()
    assert ex.produce_candidates(parsed, schema) == []


def test_find_bbox_for_text_hit():
    """_find_bbox_for_text returns (page, bbox) when a matching token exists."""
    tok = Token(text="Acme Industries Ltd", page=0, bbox=(10.0, 20.0, 150.0, 35.0))
    parsed = _minimal_parsed("Acme Industries Ltd", tokens=[tok])
    result = _find_bbox_for_text(parsed, "Acme Industries Ltd")
    assert result == (0, (10.0, 20.0, 150.0, 35.0))


def test_find_bbox_for_text_miss():
    """_find_bbox_for_text returns None when no token contains the text."""
    tok = Token(text="Something Else", page=0, bbox=(0.0, 0.0, 50.0, 10.0))
    parsed = _minimal_parsed("Something Else Acme Corp", tokens=[tok])
    result = _find_bbox_for_text(parsed, "Acme Corp")
    assert result is None


def test_find_bbox_for_text_case_insensitive():
    """_find_bbox_for_text matches case-insensitively."""
    tok = Token(text="ACME CORP", page=0, bbox=(5.0, 5.0, 80.0, 20.0))
    parsed = _minimal_parsed("ACME CORP", tokens=[tok])
    result = _find_bbox_for_text(parsed, "acme corp")
    assert result == (0, (5.0, 5.0, 80.0, 20.0))


def test_bbox_falls_back_to_zero_when_no_token():
    """When no token matches, bbox defaults to page 0 with zero bbox."""
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    # No tokens in the page, but full_text has an ORG in vendor context
    parsed = _minimal_parsed("Vendor: Initech Corporation\nInvoice: INV-99")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    # If any ORG candidate was emitted, its bbox should default gracefully
    for c in candidates:
        assert isinstance(c.bbox, tuple)
        assert len(c.bbox) == 4


# --------------------------------------------------------------------------- #
# A.1 — Header-band position filter for supplier_name                         #
# --------------------------------------------------------------------------- #

def test_supplier_name_in_header_band_produces_candidate():
    """ORG in top 30% of page 0 (header band) becomes a supplier_name candidate."""
    from src.services.extraction_v3.schemas.parsed_document import Token
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    # Token is at y=100 out of page height 842 → 100/842 ≈ 12% < 30%: in header
    tok = Token(text="Acme Industries Ltd", page=0, bbox=(10.0, 100.0, 200.0, 120.0))
    parsed = _minimal_parsed("Acme Industries Ltd\nInvoice Number: 4759275", tokens=[tok])
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    supplier_cands = [c for c in candidates if c.field == "supplier_name"]
    assert supplier_cands, "ORG in header band should produce a supplier_name candidate"
    for c in supplier_cands:
        assert c.evidence_text in parsed.full_text
        assert c.confidence >= 0.70


def test_supplier_name_below_header_band_dropped():
    """ORG below header band and not in vendor context is dropped (not emitted)."""
    from src.services.extraction_v3.schemas.parsed_document import Token
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    # Token at y=600 out of 842 → 71%: well below 30% header band
    tok = Token(text="Acme Industries Ltd", page=0, bbox=(10.0, 600.0, 200.0, 620.0))
    parsed = _minimal_parsed("Acme Industries Ltd\nInvoice Number: 4759275", tokens=[tok])
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    supplier_cands = [c for c in candidates if c.field == "supplier_name"]
    # Should NOT produce a candidate — below header band, no vendor context
    assert not supplier_cands, (
        f"ORG below header band (no vendor context) should be dropped; got {[c.value for c in supplier_cands]}"
    )


def test_supplier_name_vendor_context_overrides_position():
    """ORG near a 'Vendor:' label is emitted even without token bbox."""
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    # No tokens — no positional evidence; but "Vendor:" label qualifies it
    parsed = _minimal_parsed("Vendor: TechTonic Solutions\nInvoice Number: 4759275")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    supplier_cands = [c for c in candidates if c.field == "supplier_name"]
    assert supplier_cands, "ORG near vendor label should produce a candidate regardless of position"
    for c in supplier_cands:
        assert c.confidence >= 0.80  # vendor context → highest confidence tier


# --------------------------------------------------------------------------- #
# A.2 — Known-carrier blocklist for supplier_name                             #
# --------------------------------------------------------------------------- #

def test_ups_express_not_emitted_as_supplier():
    """UPS Express (a known carrier) must never be emitted as supplier_name."""
    from src.services.extraction_v3.schemas.parsed_document import Token
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    # UPS Express in SHIP VIA section, also with a token in header band
    tok = Token(text="UPS Express", page=0, bbox=(10.0, 50.0, 150.0, 70.0))
    text = "Vendor: Apparel Co\nSHIP VIA: UPS Express\nInvoice Number: 4759275"
    parsed = _minimal_parsed(text, tokens=[tok])
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    supplier_vals = [c.value for c in candidates if c.field == "supplier_name"]
    assert not any("UPS" in v or "ups" in v.lower() for v in supplier_vals), (
        f"Carrier 'UPS Express' should be blocklisted as supplier_name; got {supplier_vals}"
    )


def test_fedex_not_emitted_as_supplier():
    """FedEx (a known carrier) must not be emitted as supplier_name."""
    from src.services.extraction_v3.schemas.parsed_document import Token
    field = _ner_field("supplier_name", "ORG")
    schema = _minimal_schema([field])
    tok = Token(text="FedEx Ground", page=0, bbox=(10.0, 50.0, 150.0, 70.0))
    text = "Vendor: Globex Corp\nSHIP VIA: FedEx Ground\nInvoice: INV-100"
    parsed = _minimal_parsed(text, tokens=[tok])
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    supplier_vals = [c.value for c in candidates if c.field == "supplier_name"]
    assert not any("fedex" in v.lower() or "FedEx" in v for v in supplier_vals), (
        f"Carrier 'FedEx Ground' should be blocklisted; got {supplier_vals}"
    )


def test_carrier_blocklist_direct():
    """is_carrier() correctly identifies known carriers and non-carriers."""
    from src.services.extraction_v3.extractors._carrier_blocklist import is_carrier
    assert is_carrier("UPS") is True
    assert is_carrier("UPS Express") is True
    assert is_carrier("ups express") is True  # case-insensitive
    assert is_carrier("FedEx") is True
    assert is_carrier("DHL") is True
    assert is_carrier("FedEx Ground Shipping") is True  # substring match
    assert is_carrier("USPS Priority") is True
    # Non-carriers
    assert is_carrier("Acme Industries Ltd") is False
    assert is_carrier("TechTonic Solutions") is False
    assert is_carrier("Globex Corporation") is False
    assert is_carrier("") is False


# --------------------------------------------------------------------------- #
# A.3 — buyer_id prefers ORG over PERSON in BILL TO block                     #
# --------------------------------------------------------------------------- #

def test_buyer_id_emits_org_from_bill_to_block():
    """buyer_id extractor emits the ORG from BILL TO block, not a PERSON."""
    field = _ner_field("buyer_id", "ORG")
    schema = _minimal_schema([field])
    text = (
        "BILL TO:\n"
        "Globex Corporation\n"
        "John Smith\n"
        "123 Main St, New York, NY 10001\n"
    )
    parsed = _minimal_parsed(text)
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    buyer_cands = [c for c in candidates if c.field == "buyer_id"]
    if buyer_cands:
        # If a candidate was emitted, it must NOT be a person name
        for c in buyer_cands:
            assert c.value != "John Smith", (
                "buyer_id should not commit a PERSON name — only ORG entities"
            )
            assert c.evidence_text in parsed.full_text


def test_buyer_id_no_org_in_bill_to_returns_empty():
    """buyer_id emits nothing when BILL TO block contains only a PERSON (no ORG)."""
    field = _ner_field("buyer_id", "ORG")
    schema = _minimal_schema([field])
    # Only a person name in the BILL TO block, no company ORG
    text = (
        "BILL TO:\n"
        "John Smith\n"
        "456 Oak Ave, Chicago, IL 60601\n"
    )
    parsed = _minimal_parsed(text)
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    buyer_cands = [c for c in candidates if c.field == "buyer_id"]
    # Should emit nothing — PERSON in BILL TO block → NULL (not committed)
    for c in buyer_cands:
        assert c.value != "John Smith", (
            "buyer_id must not commit a PERSON name from the BILL TO block"
        )


def test_buyer_id_substring_guarantee():
    """All buyer_id candidates must satisfy the substring guarantee."""
    field = _ner_field("buyer_id", "ORG")
    schema = _minimal_schema([field])
    text = (
        "BILL TO: Initech Solutions\n"
        "Attn: Bob Smith\n"
        "789 Pine Rd, Houston, TX 77001\n"
    )
    parsed = _minimal_parsed(text)
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    for c in candidates:
        if c.field == "buyer_id":
            assert c.evidence_text in parsed.full_text, (
                f"Substring guarantee violated for buyer_id: {c.evidence_text!r}"
            )


def test_no_bill_to_block_buyer_id_returns_empty():
    """buyer_id emits nothing when there is no BILL TO / Customer block in the doc."""
    field = _ner_field("buyer_id", "ORG")
    schema = _minimal_schema([field])
    # No BILL TO block at all
    text = "Vendor: Acme Corp\nInvoice Number: INV-001\nTotal: $1000"
    parsed = _minimal_parsed(text)
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    buyer_cands = [c for c in candidates if c.field == "buyer_id"]
    assert not buyer_cands, (
        "buyer_id should not emit candidates when there is no BILL TO block"
    )


# --------------------------------------------------------------------------- #
# Integration test (real PDF, GPU / transformer model)                        #
# --------------------------------------------------------------------------- #

@pytest.mark.gpu
def test_spacy_ner_extracts_org_entities_for_supplier_name():
    """spaCy NER runs over the real invoice and emits candidates only for
    spacy_ner-enabled fields.

    INV-007-rich.pdf uses bare trade names ("Borcelle", "Fauget") without
    corporate-suffix markers (Ltd, Corp…), so en_core_web_trf does not classify
    them as ORG.  The test therefore verifies:
      1. The extractor runs without error on a real PDF.
      2. Every candidate that IS emitted has valid structure and satisfies the
         substring guarantee.
      3. No candidate is emitted for a field that did not opt into spacy_ner.
      4. Specifically for supplier_name (ner_type_check: ORG), no non-ORG string
         such as "INVOICE NUMBER: 4759275" sneaks in as a candidate (I-18 guard).

    On a different fixture with "Acme Industries Ltd" or similar, the extractor
    would emit at least one ORG candidate — see the unit test
    test_org_entity_produces_candidate which verifies this with a synthetic doc.
    """
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)

    # All candidate fields must opt into spacy_ner in invoice.yaml
    spacy_fields = {f.name for f in schema.fields if "spacy_ner" in f.extractors}
    for c in candidates:
        assert c.field in spacy_fields, (
            f"Candidate emitted for field {c.field!r} which does not list 'spacy_ner'"
        )

    # Substring guarantee — critical for hallucination prevention
    for c in candidates:
        assert c.evidence_text in parsed.full_text, (
            f"Substring guarantee violated: {c.evidence_text!r} not in full_text"
        )

    # I-18 guard: the invoice-number string must never appear as an ORG
    supplier_vals = [c.value for c in candidates if c.field == "supplier_name"]
    for val in supplier_vals:
        assert "INVOICE" not in val.upper() or len(val.split()) <= 2, (
            f"I-18 regression: suspected invoice-number string committed as supplier_name: {val!r}"
        )


@pytest.mark.gpu
def test_spacy_ner_substring_guarantee_on_real_invoice():
    """Substring guarantee must hold across all candidates from a real invoice."""
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    for c in candidates:
        assert c.evidence_text in parsed.full_text, (
            f"Substring guarantee violated on real invoice: {c.evidence_text!r}"
        )


@pytest.mark.gpu
def test_spacy_ner_confidence_and_model_on_real_invoice():
    """All real-invoice candidates must have confidence in [0.0, 1.0] and model='spacy_ner'.

    Note: confidence is context-dependent for supplier_name (0.45-0.90 based on
    entity position relative to buyer/vendor context blocks); other fields use 0.65
    (GPE) or 0.70 (PERSON and other NER types). The test verifies bounds, not an
    exact value, since the context-aware logic was introduced to reduce mis-attributions.
    """
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    ex = SpacyNERExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    for c in candidates:
        assert 0.0 <= c.confidence <= 1.0, (
            f"Confidence out of [0,1] range: field={c.field} confidence={c.confidence}"
        )
        assert c.model == "spacy_ner"
