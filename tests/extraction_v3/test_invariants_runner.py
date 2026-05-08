"""Tests for the v3 invariants runner wrapping the v2 ValidatorChain."""
import pytest

from src.services.extraction_v3.binding.invariants_runner import (
    run_invariants,
    InvariantResult,
    _collect_invariant_names,
    _VALIDATORS,
)
from src.services.extraction_v3.yaml_schema.loader import DocSchema, load_doc_schema


def test_runner_runs_document_invariants_for_invoice_schema():
    """All declared invariants run, each result has name + severity."""
    schema = load_doc_schema("invoice")
    header = {
        "invoice_id": "INV-001",
        "invoice_amount": 100.00,
        "tax_amount": 20.00,
        "tax_percent": 20.0,
        "invoice_total_incl_tax": 120.00,
        "currency": "GBP",
        "invoice_date": "2025-10-15",
    }
    line_items = [
        {"line_amount": 100.00, "quantity": 1, "unit_price": 100.00},
    ]
    results = run_invariants(header, line_items, schema)

    # Every result has a name and severity
    for r in results:
        assert r.name, "InvariantResult.name must be non-empty"
        assert r.severity, "InvariantResult.severity must be non-empty"

    # scale_mismatch is in invoice.yaml's document_invariants → must run
    names = {r.name for r in results}
    assert "scale_mismatch" in names


def test_runner_returns_list_of_invariant_result():
    """Return type is always a list of InvariantResult dataclasses."""
    schema = load_doc_schema("invoice")
    results = run_invariants({}, [], schema)
    assert isinstance(results, list)
    for r in results:
        assert isinstance(r, InvariantResult)


def test_runner_unknown_invariant_raises():
    """An invariant name not in the registry must raise ValueError with 'unknown invariant'."""
    schema = DocSchema(
        doc_type="invoice",
        db_table="proc.bp_invoice",
        fields=[],
        line_items=None,
        document_invariants=["totally_made_up_invariant_xyz"],
    )
    with pytest.raises(ValueError, match="unknown invariant"):
        run_invariants({}, [], schema)


def test_runner_scale_mismatch_critical_for_10x():
    """Runner surfaces scale_mismatch CRITICAL for the I-39 TECHWORLD case."""
    schema = load_doc_schema("invoice")
    header = {
        "invoice_amount": 675.00,
        "currency": "GBP",
        "invoice_date": "2025-10-15",
    }
    line_items = [
        {"line_amount": 5000.00},
        {"line_amount": 1750.00},
    ]
    results = run_invariants(header, line_items, schema)
    scale_r = next((r for r in results if r.name == "scale_mismatch"), None)
    assert scale_r is not None, "scale_mismatch result must be present"
    assert "CRITICAL" in scale_r.severity.upper()


def test_runner_no_critical_for_consistent_invoice():
    """A fully consistent invoice should not produce scale_mismatch CRITICAL."""
    schema = load_doc_schema("invoice")
    header = {
        "invoice_amount": 6750.00,
        "tax_amount": 1350.00,
        "tax_percent": 20.0,
        "invoice_total_incl_tax": 8100.00,
        "currency": "GBP",
        "invoice_date": "2025-10-15",
    }
    line_items = [
        {"line_amount": 5000.00, "quantity": 1, "unit_price": 5000.00},
        {"line_amount": 1750.00, "quantity": 1, "unit_price": 1750.00},
    ]
    results = run_invariants(header, line_items, schema)
    scale_r = next((r for r in results if r.name == "scale_mismatch"), None)
    assert scale_r is not None
    assert "CRITICAL" not in scale_r.severity.upper()


def test_collect_invariant_names_deduplicates_across_sections():
    """A name that appears in both document_invariants and a field's invariants
    is collected only once — the cross-section dedup guard fires."""
    from src.services.extraction_v3.yaml_schema.loader import FieldSpec, JudgeRules
    field_with_dup = FieldSpec(
        name="invoice_amount",
        type="money",
        required=True,
        db_column="invoice_amount",
        canonical_labels=["Invoice Amount"],
        extractors=["layoutlmv3"],
        invariants=["scale_mismatch"],  # duplicate of document_invariants entry
    )
    schema = DocSchema(
        doc_type="invoice",
        db_table="proc.bp_invoice",
        fields=[field_with_dup],
        line_items=None,
        document_invariants=["tax_closure", "scale_mismatch"],
    )
    names = _collect_invariant_names(schema)
    # scale_mismatch appears in both document_invariants and the field; collect once
    assert names.count("scale_mismatch") == 1
    assert "tax_closure" in names


def test_registry_contains_all_expected_validators():
    """Spot-check that the registry has all 11 expected entries."""
    expected = {
        "line_arithmetic", "subtotal_closure", "tax_closure",
        "grand_total_closure", "currency_consistency", "date_sanity",
        "vendor_identity", "line_sum_closure", "quantity_sign",
        "round_off_bucket", "scale_mismatch",
    }
    assert expected.issubset(set(_VALIDATORS.keys()))


def test_runner_with_empty_schema_returns_empty_list():
    """Schema with no invariants at all returns an empty results list."""
    schema = DocSchema(
        doc_type="invoice",
        db_table="proc.bp_invoice",
        fields=[],
        line_items=None,
        document_invariants=[],
    )
    results = run_invariants({}, [], schema)
    assert results == []
