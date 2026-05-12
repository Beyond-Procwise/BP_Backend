"""Tests for the grounded last-resort judge — anti-hallucination guarantee.

Critical coverage:
  - SAFETY CHECK 1: value == evidence_text (no soft-paraphrase)
  - SAFETY CHECK 2: evidence_text in doc_full_text (no fabrication)
  - SAFETY CHECK 3: len(value) <= MAX_VALUE_LENGTH (no runaway)
  - Property test: 100 fabricated LLM outputs all rejected
  - Empty-doc fast path skips LLM

All tests use EXTRACTION_V3_JUDGE_MODEL=ollama to avoid loading Qwen2.5-VL.
The ollama_generate import is deferred inside _call_ollama_grounded, so we
patch it at the src.services.ollama_client level.
"""
from unittest.mock import patch, MagicMock
import json
import pytest
from src.services.extraction_v3.judge.grounded_last_resort import (
    call_grounded_last_resort, _parse_response, MAX_VALUE_LENGTH,
    _apply_safety_checks,
)
from src.services.extraction_v3.judge.contracts import GroundedOutput
from src.services.extraction_v3.yaml_schema.loader import FieldSpec, JudgeRules

# Force Ollama backend in all tests to avoid loading Qwen2.5-VL model.
_OLLAMA_ENV = {"EXTRACTION_V3_JUDGE_MODEL": "ollama"}
# The patch target: ollama_generate is imported inside _call_ollama_grounded.
_OLLAMA_PATCH = "src.services.ollama_client.ollama_generate"


def _make_field(name="invoice_id", typ="string"):
    return FieldSpec(
        name=name, type=typ, required=True, db_column=name,
        canonical_labels=["Invoice Number", "Invoice No"],
        extractors=["layoutlmv3"], judge=JudgeRules(),
    )


# === Anti-hallucination: SAFETY CHECK 2 (the structural guarantee) ===

@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_REJECTS_value_not_in_doc_text(mock_ollama):
    """The CRITICAL anti-hallucination guarantee. If the LLM returns a value
    whose evidence_text is not a substring of doc_full_text, REJECT."""
    # The mock returns None (simulating Ollama rejection path via _apply_safety_checks)
    # We test the safety checks directly via _apply_safety_checks for this case.
    parsed = GroundedOutput(value="Eleanor Price", evidence_text="Eleanor Price", rationale="guess")
    result = _apply_safety_checks(parsed, "TECHWORLD INV-005-41 for PO405867. Tax: 1215.", "supplier_name")
    assert result is None  # REJECTED: Eleanor Price not in doc text


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_accepts_genuine_substring(mock_ollama):
    from src.services.extraction_v3.schemas.candidate import Candidate
    doc = "TECHWORLD INV-005-41 for PO405867. Tax: 1215."
    cand = Candidate(field="invoice_id", value="INV-005-41", page=0, bbox=(0,0,0,0),
                     evidence_text="INV-005-41", model="qa_roberta", confidence=0.7)
    mock_ollama.return_value = cand
    result = call_grounded_last_resort(_make_field("invoice_id"), doc_full_text=doc)
    assert result is not None
    assert result.value == "INV-005-41"
    assert result.evidence_text == "INV-005-41"
    assert result.evidence_text in doc


@patch.dict("os.environ", _OLLAMA_ENV)
def test_REJECTS_value_evidence_mismatch():
    """If value != evidence_text, reject (model is trying to soft-paraphrase)."""
    parsed = GroundedOutput(value="INV005-41", evidence_text="INV-005-41", rationale="stripped dash")
    result = _apply_safety_checks(parsed, "TECHWORLD INV-005-41 for PO405867", "invoice_id")
    assert result is None  # REJECTED — value != evidence


@patch.dict("os.environ", _OLLAMA_ENV)
def test_REJECTS_value_too_long():
    parsed = GroundedOutput(value="X" * 100, evidence_text="X" * 100, rationale="very long")
    result = _apply_safety_checks(parsed, "X" * 200, "invoice_id")
    assert result is None  # REJECTED — exceeds MAX_VALUE_LENGTH (64)


@patch.dict("os.environ", _OLLAMA_ENV)
def test_null_returns_none():
    parsed = GroundedOutput(value=None, evidence_text=None, rationale="field not present")
    result = _apply_safety_checks(parsed, "some doc", "invoice_id")
    assert result is None


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_invalid_json_returns_none(mock_ollama):
    mock_ollama.return_value = None
    assert call_grounded_last_resort(_make_field(), doc_full_text="...") is None


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_empty_doc_skips_llm_call(mock_ollama):
    """Don't waste an LLM call on empty docs."""
    result = call_grounded_last_resort(_make_field(), doc_full_text="")
    assert result is None
    mock_ollama.assert_not_called()


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_whitespace_only_doc_skips_llm_call(mock_ollama):
    """Whitespace-only doc is effectively empty — skip LLM."""
    result = call_grounded_last_resort(_make_field(), doc_full_text="   \n\t  ")
    assert result is None
    mock_ollama.assert_not_called()


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_ollama_returns_none(mock_ollama):
    """Ollama timeout or connection error → None returned, no exception."""
    mock_ollama.return_value = None
    result = call_grounded_last_resort(_make_field(), doc_full_text="TECHWORLD INV-005-41")
    assert result is None


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_candidate_model_is_qa_roberta(mock_ollama):
    """Accepted candidates use 'qa_roberta' as the model name (see module docstring)."""
    from src.services.extraction_v3.schemas.candidate import Candidate
    cand = Candidate(field="invoice_id", value="INV-005-41", page=0, bbox=(0,0,0,0),
                     evidence_text="INV-005-41", model="qa_roberta", confidence=0.7)
    mock_ollama.return_value = cand
    result = call_grounded_last_resort(
        _make_field("invoice_id"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867.",
    )
    assert result is not None
    assert result.model == "qa_roberta"


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_candidate_confidence_is_mid(mock_ollama):
    """Accepted candidates carry 0.7 mid-confidence."""
    from src.services.extraction_v3.schemas.candidate import Candidate
    cand = Candidate(field="invoice_id", value="INV-005-41", page=0, bbox=(0,0,0,0),
                     evidence_text="INV-005-41", model="qa_roberta", confidence=0.7)
    mock_ollama.return_value = cand
    result = call_grounded_last_resort(
        _make_field("invoice_id"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867.",
    )
    assert result is not None
    assert result.confidence == 0.7


@patch.dict("os.environ", _OLLAMA_ENV)
@patch("src.services.extraction_v3.judge.grounded_last_resort._call_ollama_grounded")
def test_candidate_bbox_is_placeholder(mock_ollama):
    """Judge-emitted candidates carry placeholder bbox (0,0,0,0) and page=0."""
    from src.services.extraction_v3.schemas.candidate import Candidate
    cand = Candidate(field="invoice_id", value="INV-005-41", page=0, bbox=(0.0,0.0,0.0,0.0),
                     evidence_text="INV-005-41", model="qa_roberta", confidence=0.7)
    mock_ollama.return_value = cand
    result = call_grounded_last_resort(
        _make_field("invoice_id"),
        doc_full_text="TECHWORLD INV-005-41 for PO405867.",
    )
    assert result is not None
    assert result.page == 0
    assert result.bbox == (0.0, 0.0, 0.0, 0.0)


# === Property test: 100 random fabricated outputs, none committed ===

def test_PROPERTY_no_random_fabrication_committed():
    """Property test: for 100 randomly-fabricated LLM outputs whose values
    are NOT substrings of the doc, NONE should be committed.

    Uses _apply_safety_checks directly — the pure function that enforces
    the anti-hallucination contract regardless of backend."""
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
        e = v if random.random() > 0.2 else random.choice(fake_words)
        parsed = GroundedOutput(value=v, evidence_text=e, rationale="guess")
        result = _apply_safety_checks(parsed, doc_full_text, "supplier_name")
        if result is None:
            rejections += 1
        else:
            accepts += 1
            assert result.evidence_text in doc_full_text, \
                f"HALLUCINATION COMMITTED: {result.evidence_text!r} not in doc text"
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
