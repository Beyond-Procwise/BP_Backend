from unittest.mock import patch
import pytest
from src.services.extraction_v3.judge.tiebreaker import (
    call_tiebreaker_judge, _parse_response, _build_prompt,
)
from src.services.extraction_v3.judge.contracts import TiebreakerInput, TiebreakerCandidate
from src.services.extraction_v3.schemas.candidate import Candidate


def _make_candidate(value: str, model="layoutlmv3", confidence=0.7) -> Candidate:
    return Candidate(
        field="supplier_name", value=value, page=0, bbox=(0, 0, 1, 1),
        evidence_text=value, model=model, confidence=confidence,
    )


def test_parse_response_extracts_json():
    raw = '{"chosen_candidate_index": 1, "rationale": "matches address block"}'
    out = _parse_response(raw)
    assert out is not None and out.chosen_candidate_index == 1


def test_parse_response_handles_prose_wrapper():
    raw = 'Sure, here you go:\n{"chosen_candidate_index": 0, "rationale": "x"}\nThanks!'
    out = _parse_response(raw)
    assert out is not None and out.chosen_candidate_index == 0


def test_parse_response_null_choice():
    raw = '{"chosen_candidate_index": null, "rationale": "neither matches"}'
    out = _parse_response(raw)
    assert out is not None and out.chosen_candidate_index is None


def test_parse_response_invalid_json():
    assert _parse_response("not json") is None
    assert _parse_response("") is None
    assert _parse_response('{"chosen_candidate_index": "invalid"}') is None


def test_parse_response_rejects_out_of_schema_fields():
    """extra='forbid' should reject malformed responses."""
    raw = '{"chosen_candidate_index": 0, "rationale": "x", "extra_garbage": "y"}'
    assert _parse_response(raw) is None


@patch("src.services.extraction_v3.judge.tiebreaker.ollama_generate")
def test_call_tiebreaker_picks_candidate(mock_gen):
    mock_gen.return_value = '{"chosen_candidate_index": 0, "rationale": "first matches"}'
    candidates = [
        _make_candidate("Acme Industries Ltd", model="layoutlmv3", confidence=0.81),
        _make_candidate("AcmeIndustries", model="qa_roberta", confidence=0.62),
    ]
    chosen = call_tiebreaker_judge(
        candidates, field="supplier_name", field_type="string",
        parsed_full_text="ACME Industries Ltd\n123 Main St",
    )
    assert chosen is not None
    assert chosen.value == "Acme Industries Ltd"


@patch("src.services.extraction_v3.judge.tiebreaker.ollama_generate")
def test_call_tiebreaker_null_choice_returns_none(mock_gen):
    mock_gen.return_value = '{"chosen_candidate_index": null, "rationale": "neither"}'
    candidates = [
        _make_candidate("Foo", confidence=0.4),
        _make_candidate("Bar", confidence=0.4),
    ]
    chosen = call_tiebreaker_judge(candidates, "x", "string", "")
    assert chosen is None


@patch("src.services.extraction_v3.judge.tiebreaker.ollama_generate")
def test_call_tiebreaker_out_of_range_returns_none(mock_gen):
    """LLM picks an invalid index → reject."""
    mock_gen.return_value = '{"chosen_candidate_index": 999, "rationale": "wat"}'
    candidates = [_make_candidate("a"), _make_candidate("b")]
    chosen = call_tiebreaker_judge(candidates, "x", "string", "")
    assert chosen is None


@patch("src.services.extraction_v3.judge.tiebreaker.ollama_generate")
def test_call_tiebreaker_ollama_failure_returns_none(mock_gen):
    mock_gen.return_value = None  # simulate ollama timeout
    candidates = [_make_candidate("a"), _make_candidate("b")]
    chosen = call_tiebreaker_judge(candidates, "x", "string", "")
    assert chosen is None


def test_call_tiebreaker_short_circuits_single_candidate():
    """Don't call Ollama if there's only 0 or 1 candidate."""
    with patch("src.services.extraction_v3.judge.tiebreaker.ollama_generate") as mock_gen:
        # 0 candidates
        assert call_tiebreaker_judge([], "x", "string", "") is None
        # 1 candidate
        c = _make_candidate("a")
        result = call_tiebreaker_judge([c], "x", "string", "")
        assert result is c
        mock_gen.assert_not_called()
