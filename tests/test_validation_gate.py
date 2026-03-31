# tests/test_validation_gate.py
"""Tests for the extraction validation gate.

The validation gate scores extraction confidence per-field and per-document,
performs structural and semantic validation, and returns a pass/fail decision.
"""
import pytest
from unittest.mock import MagicMock


def test_structural_validation_passes_complete_invoice():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    result = gate.validate_structural(
        doc_type="Invoice",
        extracted_fields={
            "invoice_id": "INV-001",
            "supplier_name": "Acme Corp",
            "invoice_total_incl_tax": "1500.00",
            "invoice_date": "2026-03-15",
        },
    )
    assert result.passed is True
    assert len(result.missing_required) == 0


def test_structural_validation_fails_missing_required():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    result = gate.validate_structural(
        doc_type="Invoice",
        extracted_fields={
            "supplier_name": "Acme Corp",
            # Missing invoice_id and invoice_total_incl_tax
        },
    )
    assert result.passed is False
    assert "invoice_id" in result.missing_required


def test_semantic_validation_checks_date_format():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    issues = gate.validate_semantic(
        doc_type="Invoice",
        extracted_fields={
            "invoice_id": "INV-001",
            "supplier_name": "Acme Corp",
            "invoice_total_incl_tax": "1500.00",
            "invoice_date": "not-a-date",
        },
    )
    assert any("date" in issue.lower() for issue in issues)


def test_semantic_validation_checks_amount_format():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    issues = gate.validate_semantic(
        doc_type="Invoice",
        extracted_fields={
            "invoice_id": "INV-001",
            "supplier_name": "Acme Corp",
            "invoice_total_incl_tax": "not-a-number",
            "invoice_date": "2026-03-15",
        },
    )
    assert any("amount" in issue.lower() or "total" in issue.lower() for issue in issues)


def test_compute_document_confidence():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    score = gate.compute_document_confidence(
        field_confidences={
            "invoice_id": 0.95,
            "supplier_name": 0.90,
            "invoice_total_incl_tax": 0.85,
            "invoice_date": 0.92,
        },
        structural_passed=True,
        semantic_issues=[],
    )
    assert 0.85 <= score <= 1.0  # High confidence, no issues


def test_compute_document_confidence_penalizes_issues():
    from services.validation_gate import ValidationGate

    gate = ValidationGate()
    high_score = gate.compute_document_confidence(
        field_confidences={"invoice_id": 0.95, "supplier_name": 0.90},
        structural_passed=True,
        semantic_issues=[],
    )
    low_score = gate.compute_document_confidence(
        field_confidences={"invoice_id": 0.95, "supplier_name": 0.90},
        structural_passed=False,
        semantic_issues=["bad date", "bad amount"],
    )
    assert low_score < high_score


def test_gate_decision_passes_high_confidence():
    from services.validation_gate import ValidationGate

    gate = ValidationGate(confidence_threshold=0.70)
    decision = gate.evaluate(
        doc_type="Invoice",
        extracted_fields={
            "invoice_id": "INV-001",
            "supplier_name": "Acme Corp",
            "invoice_total_incl_tax": "1500.00",
            "invoice_date": "2026-03-15",
        },
        field_confidences={
            "invoice_id": 0.95,
            "supplier_name": 0.90,
            "invoice_total_incl_tax": 0.85,
            "invoice_date": 0.92,
        },
    )
    assert decision.passed is True
    assert decision.confidence_score >= 0.70
    assert decision.needs_remediation is False


def test_gate_decision_routes_to_remediation():
    from services.validation_gate import ValidationGate

    gate = ValidationGate(confidence_threshold=0.70)
    decision = gate.evaluate(
        doc_type="Invoice",
        extracted_fields={
            "supplier_name": "Acme Corp",
            # Missing required fields
        },
        field_confidences={
            "supplier_name": 0.40,
        },
    )
    assert decision.passed is False
    assert decision.needs_remediation is True
    assert len(decision.low_confidence_fields) > 0
