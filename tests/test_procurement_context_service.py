"""Tests for ProcurementContextService."""
import sys
import os
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestProcurementContextService:
    def _make_service(self):
        from services.procurement_context_service import ProcurementContextService
        return ProcurementContextService(MagicMock())

    def test_determine_lifecycle_stage_invoice_with_po(self):
        svc = self._make_service()
        stage = svc.determine_lifecycle_stage("Invoice", {"po_id": "526809"})
        assert stage == "Invoice Matching"

    def test_determine_lifecycle_stage_invoice_without_po(self):
        svc = self._make_service()
        stage = svc.determine_lifecycle_stage("Invoice", {})
        assert stage == "Invoice Received"

    def test_determine_lifecycle_stage_new_quote(self):
        svc = self._make_service()
        stage = svc.determine_lifecycle_stage("Quote", {})
        assert stage == "Quotes Received"

    def test_determine_lifecycle_stage_po(self):
        svc = self._make_service()
        stage = svc.determine_lifecycle_stage("Purchase_Order", {})
        assert stage == "PO Issued"

    def test_determine_lifecycle_stage_contract(self):
        svc = self._make_service()
        stage = svc.determine_lifecycle_stage("Contract", {})
        assert stage == "Supplier Selected"

    def test_build_context_brief_has_required_fields(self):
        svc = self._make_service()
        brief = svc.build_context_brief(
            doc_type="Invoice",
            header={"invoice_id": "INV001", "po_id": "PO526809", "supplier_id": "SupplyX"},
            patterns=[{"pattern_text": "UK invoices have 20% VAT"}],
        )
        assert brief["lifecycle_stage"] == "Invoice Matching"
        assert brief["document_type"] == "Invoice"
        assert "supplier_id" in brief["document_summary"]
        assert len(brief["patterns"]) == 1
        assert brief["next_expected_stage"] == "Payment Approved"

    def test_build_context_brief_includes_related_docs(self):
        svc = self._make_service()
        brief = svc.build_context_brief(
            doc_type="Quote",
            header={"quote_id": "QUT001"},
            related_docs=["PO526809", "INV304056"],
        )
        assert "PO526809" in brief["related_documents"]

    def test_detect_anomalies_invoice_exceeds_po(self):
        svc = self._make_service()
        anomalies = svc.detect_anomalies(
            header={"invoice_total_incl_tax": 15000, "po_id": "PO001"},
            po_total=10000,
        )
        assert len(anomalies) >= 1
        assert any("exceeds" in a.lower() for a in anomalies)

    def test_detect_anomalies_invoice_below_po(self):
        svc = self._make_service()
        anomalies = svc.detect_anomalies(
            header={"invoice_total_incl_tax": 8000},
            po_total=10000,
        )
        assert any("below" in a.lower() for a in anomalies)

    def test_no_anomaly_within_tolerance(self):
        svc = self._make_service()
        anomalies = svc.detect_anomalies(
            header={"invoice_total_incl_tax": 10500},
            po_total=10000,
        )
        assert len(anomalies) == 0

    def test_no_anomaly_when_no_po_total(self):
        svc = self._make_service()
        anomalies = svc.detect_anomalies(
            header={"invoice_total_incl_tax": 15000},
        )
        assert len(anomalies) == 0

    def test_next_stage_returns_correct_stage(self):
        from services.procurement_context_service import ProcurementContextService
        assert ProcurementContextService._next_stage("PO Issued") == "Invoice Received"
        assert ProcurementContextService._next_stage("Invoice Matching") == "Payment Approved"
        assert ProcurementContextService._next_stage("Supplier Reviewed") == ""
