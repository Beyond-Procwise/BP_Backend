# tests/test_training_data_collector.py
"""Tests for training data collection from extraction corrections."""
import pytest
import json
import os
import tempfile
from unittest.mock import MagicMock


def test_collect_extraction_correction():
    from services.training_data_collector import TrainingDataCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrainingDataCollector(output_dir=tmpdir)
        collector.record_correction(
            doc_type="Invoice",
            document_text="Invoice #INV-001\nSupplier: Acme\nTotal: $1500",
            original_fields={"invoice_id": "INV-00", "supplier_name": "Acm"},
            corrected_fields={"invoice_id": "INV-001", "supplier_name": "Acme"},
            correction_source="remediation",
        )
        # Should write to extraction adapter training file
        path = os.path.join(tmpdir, "extraction_corrections.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            line = json.loads(f.readline())
        assert line["doc_type"] == "Invoice"
        assert line["corrected_fields"]["invoice_id"] == "INV-001"


def test_collect_negotiation_example():
    from services.training_data_collector import TrainingDataCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrainingDataCollector(output_dir=tmpdir)
        collector.record_negotiation(
            workflow_id="wf-001",
            supplier_name="Acme Corp",
            round_num=2,
            strategy="competitive",
            counter_offer="We can offer 10% discount",
            outcome="accepted",
        )
        path = os.path.join(tmpdir, "negotiation_examples.jsonl")
        assert os.path.exists(path)
        with open(path) as f:
            line = json.loads(f.readline())
        assert line["outcome"] == "accepted"


def test_get_training_stats():
    from services.training_data_collector import TrainingDataCollector

    with tempfile.TemporaryDirectory() as tmpdir:
        collector = TrainingDataCollector(output_dir=tmpdir)
        collector.record_correction(
            doc_type="Invoice",
            document_text="test",
            original_fields={},
            corrected_fields={"invoice_id": "INV-001"},
            correction_source="remediation",
        )
        collector.record_correction(
            doc_type="Quote",
            document_text="test2",
            original_fields={},
            corrected_fields={"quote_id": "Q-001"},
            correction_source="remediation",
        )
        stats = collector.get_stats()
        assert stats["extraction_corrections"] == 2
