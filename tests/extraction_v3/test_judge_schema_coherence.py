from unittest.mock import patch
import json
from src.services.extraction_v3.judge.schema_coherence import (
    call_coherence_judge, _parse_response,
)
from src.services.extraction_v3.judge.contracts import CoherenceOutput, InvariantResultSummary


@patch("src.services.extraction_v3.judge.schema_coherence.ollama_generate")
def test_coherent_record(mock_gen):
    mock_gen.return_value = json.dumps({"verdict": "coherent", "issues": []})
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


@patch("src.services.extraction_v3.judge.schema_coherence.ollama_generate")
def test_incoherent_record_with_issues(mock_gen):
    """The I-38 case: requested_by has name from a different doc."""
    mock_gen.return_value = json.dumps({
        "verdict": "incoherent",
        "issues": [
            {"field": "requested_by", "issue": "name 'Eleanor Price' does not appear elsewhere in the record"},
        ],
    })
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


@patch("src.services.extraction_v3.judge.schema_coherence.ollama_generate")
def test_invalid_verdict_returns_none(mock_gen):
    mock_gen.return_value = json.dumps({"verdict": "maybe", "issues": []})
    result = call_coherence_judge("invoice", {"invoice_id": "X"})
    assert result is None


@patch("src.services.extraction_v3.judge.schema_coherence.ollama_generate")
def test_invalid_json_returns_none(mock_gen):
    mock_gen.return_value = "not json"
    result = call_coherence_judge("invoice", {"invoice_id": "X"})
    assert result is None


@patch("src.services.extraction_v3.judge.schema_coherence.ollama_generate")
def test_ollama_failure_returns_none(mock_gen):
    mock_gen.return_value = None
    result = call_coherence_judge("invoice", {"invoice_id": "X"})
    assert result is None


@patch("src.services.extraction_v3.judge.schema_coherence.ollama_generate")
def test_empty_record_skips_llm(mock_gen):
    result = call_coherence_judge("invoice", {})
    assert result is None
    mock_gen.assert_not_called()


@patch("src.services.extraction_v3.judge.schema_coherence.ollama_generate")
def test_with_invariant_results(mock_gen):
    """Invariant results are passed in the prompt (caller can verify by inspecting the call)."""
    mock_gen.return_value = json.dumps({"verdict": "coherent", "issues": []})
    result = call_coherence_judge(
        "invoice", {"invoice_id": "X"},
        invariant_results=[
            InvariantResultSummary(name="subtotal_closure", passed=True),
            InvariantResultSummary(name="tax_closure", passed=False, message="off by 5p"),
        ],
    )
    assert result is not None
    # Verify the prompt included the invariants
    call_args = mock_gen.call_args
    prompt = call_args[0][0] if call_args[0] else call_args[1]["prompt"]
    assert "subtotal_closure" in prompt
    assert "tax_closure" in prompt
    assert "off by 5p" in prompt


def test_parse_response_extracts_from_prose():
    """LLM wraps response in chat-style prose."""
    raw = 'Sure, here is my analysis:\n\n{"verdict": "coherent", "issues": []}\n\nThank you!'
    out = _parse_response(raw)
    assert out is not None and out.verdict == "coherent"
