"""Tests for the vendor-template extractor (Task 14 of the extraction-redesign plan).

The VendorTemplateExtractor wraps extraction_v2's PostgresTemplateStore.
When a v3 ParsedDocument's layout fingerprint matches a stored vendor
template, it emits each field hint as a high-confidence Candidate.

The test injects a template into the store before running, then removes it
afterwards so no state leaks between test runs.

Run (GPU required for parse_with_docling on the PDF fixture):
    .venv/bin/pytest tests/extraction_v3/test_vendor_template_extractor.py -v -m gpu
"""
from pathlib import Path
import pytest
from src.services.extraction_v3.parsers.docling_backend import parse_with_docling
from src.services.extraction_v3.extractors.vendor_template import (
    VendorTemplateExtractor, compute_v3_fingerprint
)
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema
from src.services.extraction_v2.template_store import VendorTemplate, FieldHint
from src.services.extraction_v2.template_store_pg import PostgresTemplateStore
from src.services.db import get_conn


FX = Path(__file__).parent / "fixtures/invoices"


@pytest.fixture
def template_store_with_invoice_template():
    """Insert a vendor template for the test fixture, yield, clean up."""
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    fp = compute_v3_fingerprint(parsed)

    # Pick a value KNOWN to be in parsed.full_text (the substring guarantee
    # depends on this). Use a token we can see — find one programmatically.
    sample_tokens = [t.text.strip() for t in parsed.pages[0].tokens if t.text.strip()]
    seed_value = next((t for t in sample_tokens if len(t) >= 4), "INVOICE")

    store = PostgresTemplateStore(get_conn)
    store.init_schema()
    template = VendorTemplate(
        fingerprint=fp,
        vendor_name="TestVendor",
        doc_type="invoice",
        field_hints={
            "supplier_name": FieldHint(
                field="supplier_name",
                value=seed_value,
                confidence=0.92,
                label="Vendor",
                anchor=None,
            ),
        },
    )
    store.upsert(template)
    yield parsed, fp, seed_value
    # Clean up — remove the test row so it doesn't affect other test runs
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM proc.bp_extraction_template WHERE fingerprint = %s",
                (fp,),
            )


@pytest.mark.gpu
def test_vendor_template_emits_candidates_with_substring_evidence(
    template_store_with_invoice_template,
):
    parsed, fp, seed_value = template_store_with_invoice_template
    schema = load_doc_schema("invoice")
    ex = VendorTemplateExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    # Exactly one candidate from our injected hint
    assert candidates, "VendorTemplate produced no candidates"
    sup_cands = [c for c in candidates if c.field == "supplier_name"]
    assert sup_cands, "no supplier_name candidate from template"
    c = sup_cands[0]
    assert c.value == seed_value
    assert c.evidence_text in parsed.full_text
    assert c.model == "vendor_template"
    assert c.confidence > 0.9


@pytest.mark.gpu
def test_vendor_template_returns_empty_when_no_template_found():
    """When no template is stored for the fingerprint, produce_candidates returns []."""
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    schema = load_doc_schema("invoice")
    fp = compute_v3_fingerprint(parsed)

    # Ensure nothing is in the store for this fingerprint (clean state)
    store = PostgresTemplateStore(get_conn)
    store.init_schema()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM proc.bp_extraction_template WHERE fingerprint = %s",
                (fp,),
            )

    ex = VendorTemplateExtractor()
    candidates = ex.produce_candidates(parsed, schema)
    assert candidates == [], f"expected [] but got {candidates}"


@pytest.mark.gpu
def test_compute_v3_fingerprint_is_stable():
    """The same ParsedDocument must produce the same fingerprint on repeated calls."""
    parsed = parse_with_docling(FX / "INV-007-rich.pdf", file_format="pdf-native")
    fp1 = compute_v3_fingerprint(parsed)
    fp2 = compute_v3_fingerprint(parsed)
    assert fp1 == fp2, "fingerprint is not stable across repeated calls"
    # Fingerprint must be a 16-char hex string
    assert len(fp1) == 16
    assert all(c in "0123456789abcdef" for c in fp1)


@pytest.mark.gpu
def test_vendor_template_drops_hints_not_in_full_text(
    template_store_with_invoice_template,
):
    """Hints whose value is not a substring of parsed.full_text are silently dropped."""
    parsed, fp, seed_value = template_store_with_invoice_template

    # Inject an extra hint with a value that can't possibly be in the doc
    store = PostgresTemplateStore(get_conn)
    ghost_value = "THIS_VALUE_DEFINITELY_NOT_IN_DOC_XYZZY_42"
    assert ghost_value not in parsed.full_text, "ghost value was unexpectedly found in doc"

    template = store.get(fp)
    template.field_hints["invoice_id"] = FieldHint(
        field="invoice_id",
        value=ghost_value,
        confidence=0.92,
        label=None,
        anchor=None,
    )
    store.upsert(template)

    schema = load_doc_schema("invoice")
    ex = VendorTemplateExtractor()
    candidates = ex.produce_candidates(parsed, schema)

    # invoice_id candidate must NOT appear (ghost value not in full_text)
    inv_cands = [c for c in candidates if c.field == "invoice_id"]
    assert not inv_cands, (
        f"ghost-value hint was not dropped; got candidates: {inv_cands}"
    )
