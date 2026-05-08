import os
from unittest.mock import patch, MagicMock
import pytest
from src.services.extraction_v3.dispatch import (
    dispatch_document, _flag_for_category, _adapt_v3_result,
)
from src.services.extraction_v3.schemas.result import (
    ExtractionResult, CommittedField, ResidualField,
)


def test_flag_default_is_agentnick():
    """When env var unset, dispatch defaults to agentnick."""
    # Clear any prior env var for this test
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("EXTRACTION_PIPELINE_INVOICE", None)
        assert _flag_for_category("invoice") == "agentnick"


def test_flag_normalises_category_synonyms():
    with patch.dict(os.environ, {"EXTRACTION_PIPELINE_PURCHASE_ORDER": "v3"}):
        assert _flag_for_category("po") == "v3"
        assert _flag_for_category("purchase order") == "v3"
        assert _flag_for_category("purchase_order") == "v3"


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
def test_dispatch_routes_to_v3_when_flag_v3(mock_persist, mock_pipeline_cls):
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
    with patch.dict(os.environ, {"EXTRACTION_PIPELINE_INVOICE": "v3"}):
        out = dispatch_document(None, "/tmp/x.pdf", "invoice")
    assert out["pk"] == "INV-X"
    instance.run.assert_called_once_with("/tmp/x.pdf", "invoice")
    mock_persist.assert_called_once()


@patch("src.services.agent_nick_orchestrator.AgentNickOrchestrator")
def test_dispatch_routes_to_agentnick_when_flag_default(mock_agent_nick_cls):
    nick_instance = MagicMock()
    nick_instance.process_document.return_value = {"status": "ok", "pk": "INV-LEGACY"}
    mock_agent_nick_cls.return_value = nick_instance
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("EXTRACTION_PIPELINE_INVOICE", None)
        out = dispatch_document("settings_obj", "/tmp/x.pdf", "invoice")
    assert out["pk"] == "INV-LEGACY"
    nick_instance.process_document.assert_called_once()


@patch("src.services.extraction_v3.dispatch.PipelineV3")
def test_dispatch_v3_failure_returns_error_dict(mock_pipeline_cls):
    instance = MagicMock()
    instance.run.side_effect = RuntimeError("model crash")
    mock_pipeline_cls.return_value = instance
    with patch.dict(os.environ, {"EXTRACTION_PIPELINE_INVOICE": "v3"}):
        out = dispatch_document(None, "/tmp/x.pdf", "invoice")
    assert out["status"] == "error"
    assert "model crash" in out.get("error", "")
