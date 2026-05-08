"""Tests for the grounded last-resort judge — anti-hallucination guarantee.

Critical coverage:
  - SAFETY CHECK 1: value == evidence_text (no soft-paraphrase)
  - SAFETY CHECK 2: evidence_text in doc_full_text (no fabrication)
  - SAFETY CHECK 3: len(value) <= MAX_VALUE_LENGTH (no runaway)
  - Property test: 100 fabricated LLM outputs all rejected
  - Empty-doc fast path skips Ollama
"""
from unittest.mock import patch
import json
import pytest
from src.services.extraction_v3.judge.grounded_last_resort import (
    call_grounded_last_resort, _parse_response, MAX_VALUE_LENGTH,
)
from src.services.extraction_v3.judge.contracts import GroundedOutput
from src.services.extraction_v3.yaml_schema.loader import FieldSpec, JudgeRules


def _make_field(name="invoice_id", typ="string"):
    return FieldSpec(
        name=name, type=typ, required=True, db_column=name,
        canonical_labels=["Invoice Number", "Invoice No"],
        extractors=["layoutlmv3"], judge=JudgeRules(),
    )


# === Anti-hallucination: SAFETY CHECK 2 (the structural guarantee) ===

@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_REJECTS_value_not_in_doc_text(mock_gen):
    """The CRITICAL anti-hallucination guarantee. If the LLM returns a value
    whose evidence_text is not a substring of doc_full_text, REJECT.

    This test simulates the I-38 cross-doc-leakage case (Eleanor Price into
    a TECHWORLD invoice)."""
    mock_gen.return_value = json.dumps({
        "value": "Eleanor Price",
        "evidence_text": "Eleanor Price",
        "rationale": "guess",
    })
    result = call_grounded_last_resort(
        _make_field("supplier_name"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867. Tax: 1215.",
    )
    assert result is None  # REJECTED


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_accepts_genuine_substring(mock_gen):
    mock_gen.return_value = json.dumps({
        "value": "INV-005-41",
        "evidence_text": "INV-005-41",
        "rationale": "matches doc",
    })
    result = call_grounded_last_resort(
        _make_field("invoice_id"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867. Tax: 1215.",
    )
    assert result is not None
    assert result.value == "INV-005-41"
    assert result.evidence_text == "INV-005-41"
    assert result.evidence_text in "TECHWORLD INV-005-41 for PO405867. Tax: 1215."


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_REJECTS_value_evidence_mismatch(mock_gen):
    """If value != evidence_text, reject (model is trying to soft-paraphrase)."""
    mock_gen.return_value = json.dumps({
        "value": "INV005-41",
        "evidence_text": "INV-005-41",
        "rationale": "stripped dash",
    })
    result = call_grounded_last_resort(
        _make_field("invoice_id"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867",
    )
    assert result is None  # REJECTED — value != evidence


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_REJECTS_value_too_long(mock_gen):
    mock_gen.return_value = json.dumps({
        "value": "X" * 100,
        "evidence_text": "X" * 100,
        "rationale": "very long",
    })
    result = call_grounded_last_resort(
        _make_field(),
        doc_full_text="X" * 200,
    )
    assert result is None  # REJECTED — exceeds MAX_VALUE_LENGTH (64)


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_null_returns_none(mock_gen):
    mock_gen.return_value = json.dumps({
        "value": None,
        "evidence_text": None,
        "rationale": "field not present",
    })
    result = call_grounded_last_resort(_make_field(), doc_full_text="some doc")
    assert result is None


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_invalid_json_returns_none(mock_gen):
    mock_gen.return_value = "not even close to json"
    assert call_grounded_last_resort(_make_field(), doc_full_text="...") is None


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_empty_doc_skips_llm_call(mock_gen):
    """Don't waste an LLM call on empty docs."""
    result = call_grounded_last_resort(_make_field(), doc_full_text="")
    assert result is None
    mock_gen.assert_not_called()


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_whitespace_only_doc_skips_llm_call(mock_gen):
    """Whitespace-only doc is effectively empty — skip Ollama."""
    result = call_grounded_last_resort(_make_field(), doc_full_text="   \n\t  ")
    assert result is None
    mock_gen.assert_not_called()


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_ollama_returns_none(mock_gen):
    """Ollama timeout or connection error → None returned, no exception."""
    mock_gen.return_value = None
    result = call_grounded_last_resort(_make_field(), doc_full_text="TECHWORLD INV-005-41")
    assert result is None


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_candidate_model_is_qa_roberta(mock_gen):
    """Accepted candidates use 'qa_roberta' as the model name (see module docstring)."""
    mock_gen.return_value = json.dumps({
        "value": "INV-005-41",
        "evidence_text": "INV-005-41",
        "rationale": "matches doc",
    })
    result = call_grounded_last_resort(
        _make_field("invoice_id"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867.",
    )
    assert result is not None
    assert result.model == "qa_roberta"


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_candidate_confidence_is_mid(mock_gen):
    """Accepted candidates carry 0.7 mid-confidence."""
    mock_gen.return_value = json.dumps({
        "value": "INV-005-41",
        "evidence_text": "INV-005-41",
        "rationale": "matches doc",
    })
    result = call_grounded_last_resort(
        _make_field("invoice_id"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867.",
    )
    assert result is not None
    assert result.confidence == 0.7


@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_candidate_bbox_is_placeholder(mock_gen):
    """Judge-emitted candidates carry placeholder bbox (0,0,0,0) and page=0."""
    mock_gen.return_value = json.dumps({
        "value": "INV-005-41",
        "evidence_text": "INV-005-41",
        "rationale": "matches doc",
    })
    result = call_grounded_last_resort(
        _make_field("invoice_id"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867.",
    )
    assert result is not None
    assert result.page == 0
    assert result.bbox == (0.0, 0.0, 0.0, 0.0)


# === Property test: 100 random fabricated outputs, none committed ===

@patch("src.services.extraction_v3.judge.grounded_last_resort.ollama_generate")
def test_PROPERTY_no_random_fabrication_committed(mock_gen):
    """Property test: for 100 randomly-fabricated LLM outputs whose values
    are NOT substrings of the doc, NONE should be committed.

    This is the structural anti-hallucination guarantee enforced as a
    quantified safety property."""
    import random
    random.seed(42)
    fake_words = [
        "Eleanor Price", "Jane Doe", "Laura Stevens", "Acme Corp",
        "Global Industries", "Mystery Vendor", "ABC Limited",
        "INV-FAKE-1", "FAKE-PO-99999", "999999",
    ]
    doc_full_text = "TECHWORLD INV-005-41 for PO405867. Tax: 1215."
    rejections = 0
    accepts = 0
    for _ in range(100):
        v = random.choice(fake_words)
        # Vary: sometimes value == evidence, sometimes not
        e = v if random.random() > 0.2 else random.choice(fake_words)
        mock_gen.return_value = json.dumps({
            "value": v, "evidence_text": e, "rationale": "guess",
        })
        result = call_grounded_last_resort(_make_field(), doc_full_text=doc_full_text)
        if result is None:
            rejections += 1
        else:
            accepts += 1
            # If accepted, the value MUST be a substring of doc_full_text
            assert result.evidence_text in doc_full_text, \
                f"HALLUCINATION COMMITTED: {result.evidence_text!r} not in doc text"
    # None of the fake_words are substrings of doc_full_text → all 100 should reject
    assert rejections == 100, f"expected 100 rejections, got {rejections} (accepts={accepts})"


def test_parse_response_strict_extra_forbid():
    """LLM adding extra fields → reject (GroundedOutput has extra='forbid')."""
    raw = '{"value": "x", "evidence_text": "x", "rationale": "y", "extra": "z"}'
    assert _parse_response(raw) is None


def test_parse_response_valid_null_output():
    """Null value/evidence is a valid response (field not in doc)."""
    raw = '{"value": null, "evidence_text": null, "rationale": "not present"}'
    out = _parse_response(raw)
    assert out is not None
    assert out.is_null()


def test_parse_response_handles_prose_wrapper():
    """LLM may wrap JSON in prose — extract the first {...} block."""
    raw = 'Here is my answer:\n{"value": "INV-001", "evidence_text": "INV-001", "rationale": "found"}\nDone.'
    out = _parse_response(raw)
    assert out is not None
    assert out.value == "INV-001"
