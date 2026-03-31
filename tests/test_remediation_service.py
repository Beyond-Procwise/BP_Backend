"""Tests for the extraction remediation service.

Remediation re-extracts low-confidence fields using alternative strategies
and cross-references against existing PostgreSQL data.
"""
import pytest
from unittest.mock import MagicMock, patch


def test_remediate_retries_with_llm():
    from services.remediation_service import RemediationService

    mock_llm = MagicMock(return_value={"invoice_id": "INV-001"})
    service = RemediationService(llm_extract_func=mock_llm)

    result = service.remediate_fields(
        low_confidence_fields=["invoice_id"],
        document_text="Invoice Number: INV-001\nTotal: $1500",
        doc_type="Invoice",
        existing_fields={"supplier_name": "Acme"},
    )
    assert "invoice_id" in result.improved_fields
    mock_llm.assert_called_once()


def test_remediate_cross_references_postgres():
    from services.remediation_service import RemediationService

    mock_llm = MagicMock(return_value={})
    mock_db_lookup = MagicMock(return_value={"supplier_name": "Acme Corporation"})
    service = RemediationService(
        llm_extract_func=mock_llm,
        db_lookup_func=mock_db_lookup,
    )

    result = service.remediate_fields(
        low_confidence_fields=["supplier_name"],
        document_text="Supplier: Acme Corp",
        doc_type="Invoice",
        existing_fields={"supplier_name": "Acme Corp"},
    )
    # Should have tried DB cross-reference
    mock_db_lookup.assert_called()


def test_remediate_returns_improved_confidence():
    from services.remediation_service import RemediationService

    mock_llm = MagicMock(return_value={"invoice_id": "INV-999"})
    service = RemediationService(llm_extract_func=mock_llm)

    result = service.remediate_fields(
        low_confidence_fields=["invoice_id"],
        document_text="Invoice #INV-999",
        doc_type="Invoice",
        existing_fields={},
    )
    assert result.improved_confidences.get("invoice_id", 0) > 0


def test_remediate_handles_no_improvement():
    from services.remediation_service import RemediationService

    mock_llm = MagicMock(return_value={})
    service = RemediationService(llm_extract_func=mock_llm)

    result = service.remediate_fields(
        low_confidence_fields=["invoice_id"],
        document_text="Some random text with no invoice info",
        doc_type="Invoice",
        existing_fields={},
    )
    assert "invoice_id" not in result.improved_fields
