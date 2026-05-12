from unittest.mock import patch
import json
import pytest
from src.services.extraction_v3.judge.schema_coherence import (
    call_coherence_judge, _parse_response,
)
from src.services.extraction_v3.judge.contracts import CoherenceOutput, InvariantResultSummary

# All tests in this file use the Ollama backend to avoid loading Qwen2.5-VL.
# The env var forces the Ollama code-path so we can mock ollama_generate.
_OLLAMA_ENV = {"EXTRACTION_V3_JUDGE_MODEL": "ollama"}


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.schema_coherence._call_ollama_coherence")
def test_coherent_record(mock_ollama):
    mock_ollama.return_value = CoherenceOutput(verdict="coherent", issues=[])
    result = call_coherence_judge(
        doc_type="invoice",
        extracted_record={
            "invoice_id": "INV-005-41", "supplier_name": "TECHWORLD",
            "invoice_date": "2025-10-15", "invoice_amount": 6750.00,
        },
    )
    assert result is not None
    assert result.verdict == "coherent"
    assert result.issues == []


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.schema_coherence._call_ollama_coherence")
def test_incoherent_record_with_issues(mock_ollama):
    """The I-38 case: requested_by has name from a different doc."""
    from src.services.extraction_v3.judge.contracts import CoherenceIssue
    mock_ollama.return_value = CoherenceOutput(
        verdict="incoherent",
        issues=[CoherenceIssue(field="requested_by", issue="name 'Eleanor Price' does not appear elsewhere in the record")],
    )
    result = call_coherence_judge(
        doc_type="invoice",
        extracted_record={
            "invoice_id": "INV-005-41", "supplier_name": "TECHWORLD",
            "requested_by": "Eleanor Price",  # cross-doc leakage
        },
    )
    assert result is not None
    assert result.verdict == "incoherent"
    assert len(result.issues) == 1
    assert "Eleanor Price" in result.issues[0].issue


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.schema_coherence._call_ollama_coherence")
def test_invalid_verdict_returns_none(mock_ollama):
    mock_ollama.return_value = None
    result = call_coherence_judge("invoice", {"invoice_id": "X"})
    assert result is None


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.schema_coherence._call_ollama_coherence")
def test_invalid_json_returns_none(mock_ollama):
    mock_ollama.return_value = None
    result = call_coherence_judge("invoice", {"invoice_id": "X"})
    assert result is None


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.schema_coherence._call_ollama_coherence")
def test_ollama_failure_returns_none(mock_ollama):
    mock_ollama.return_value = None
    result = call_coherence_judge("invoice", {"invoice_id": "X"})
    assert result is None


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.schema_coherence._call_ollama_coherence")
def test_empty_record_skips_llm(mock_ollama):
    result = call_coherence_judge("invoice", {})
    assert result is None
    mock_ollama.assert_not_called()


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.schema_coherence._call_ollama_coherence")
def test_with_invariant_results(mock_ollama):
    """Invariant results are passed through to the backend (test that backend is called)."""
    mock_ollama.return_value = CoherenceOutput(verdict="coherent", issues=[])
    result = call_coherence_judge(
        "invoice", {"invoice_id": "X"},
        invariant_results=[
            InvariantResultSummary(name="subtotal_closure", passed=True),
            InvariantResultSummary(name="tax_closure", passed=False, message="off by 5p"),
        ],
    )
    assert result is not None
    # Verify the backend was called with the CoherenceInput that contains invariants
    assert mock_ollama.call_count == 1
    called_input = mock_ollama.call_args[0][0]
    inv_names = [r.name for r in called_input.invariant_results]
    assert "subtotal_closure" in inv_names
    assert "tax_closure" in inv_names


def test_parse_response_extracts_from_prose():
    """LLM wraps response in chat-style prose."""
    raw = 'Sure, here is my analysis:\n\n{"verdict": "coherent", "issues": []}\n\nThank you!'
    out = _parse_response(raw)
    assert out is not None and out.verdict == "coherent"
