import os
from unittest.mock import patch, MagicMock
import pytest
from src.services.extraction_v3.dispatch import (
    dispatch_document, _normalise_category, _adapt_v3_result,
)
from src.services.extraction_v3.schemas.result import (
    ExtractionResult, CommittedField, ResidualField,
)


def test_normalise_category_invoice():
    """invoice normalises to itself."""
    assert _normalise_category("invoice") == "invoice"


def test_normalise_category_po_synonyms():
    """All PO synonyms normalise to purchase_order."""
    assert _normalise_category("po") == "purchase_order"
    assert _normalise_category("purchase order") == "purchase_order"
    assert _normalise_category("purchase_order") == "purchase_order"


def test_adapt_v3_result_ok():
    result = ExtractionResult(
        doc_type="invoice", doc_pk="INV-1",
        committed=[
            CommittedField(field_path="x", value="y", page=0, bbox=(0, 0, 1, 1),
                            evidence_text="y", model="layoutlmv3", model_confidence=0.9,
                            judge_actions=[], final_confidence=0.85),
        ],
        residuals=[],
        judge_calls=2, pipeline_version="v3.1.0",
    )
    out = _adapt_v3_result(result)
    assert out["status"] == "ok"
    assert out["pk"] == "INV-1"
    assert out["header_persisted"] is True
    assert out["confidence"] == 0.85


def test_adapt_v3_result_partial_on_required_residual():
    result = ExtractionResult(
        doc_type="invoice", doc_pk="INV-1",
        committed=[],
        residuals=[
            ResidualField(field_path="invoice_id", reason="required_field_missing_no_grounding"),
        ],
        judge_calls=1, pipeline_version="v3.1.0",
    )
    out = _adapt_v3_result(result)
    assert out["status"] == "partial"
    assert out["header_persisted"] is False


@patch("src.services.extraction_v3.dispatch.PipelineV3")
@patch("src.services.extraction_v3.dispatch.persist_v3")
def test_dispatch_routes_to_v3(mock_persist, mock_pipeline_cls):
    """dispatch_document always routes to v3 (no flag check needed)."""
    instance = MagicMock()
    result = ExtractionResult(
        doc_type="invoice", doc_pk="INV-X",
        committed=[
            CommittedField(field_path="x", value="y", page=0, bbox=(0, 0, 1, 1),
                            evidence_text="y", model="layoutlmv3", model_confidence=0.9,
                            judge_actions=[], final_confidence=0.85),
        ],
        residuals=[], judge_calls=0, pipeline_version="v3.1.0",
    )
    instance.run.return_value = result
    mock_pipeline_cls.return_value = instance
    out = dispatch_document(None, "/tmp/x.pdf", "invoice")
    assert out["pk"] == "INV-X"
    instance.run.assert_called_once_with("/tmp/x.pdf", "invoice")
    mock_persist.assert_called_once()


@patch("src.services.extraction_v3.dispatch.PipelineV3")
def test_dispatch_v3_failure_returns_error_dict(mock_pipeline_cls):
    instance = MagicMock()
    instance.run.side_effect = RuntimeError("model crash")
    mock_pipeline_cls.return_value = instance
    out = dispatch_document(None, "/tmp/x.pdf", "invoice")
    assert out["status"] == "error"
    assert "model crash" in out.get("error", "")
