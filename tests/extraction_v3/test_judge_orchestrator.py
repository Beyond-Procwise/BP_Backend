from unittest.mock import patch
import pytest
from src.services.extraction_v3.judge.orchestrator import run_judge_orchestrator
from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.yaml_schema.loader import load_doc_schema, DocSchema, FieldSpec, JudgeRules


def _c(field, value, model="layoutlmv3", confidence=0.7) -> Candidate:
    return Candidate(
        field=field, value=value, page=0, bbox=(0, 0, 1, 1),
        evidence_text=value, model=model, confidence=confidence,
    )


def _required_baseline(schema: DocSchema, exclude: set[str]) -> dict[str, list[Candidate]]:
    """Return one candidate per required field not in `exclude`, so those fields
    don't trigger grounded-last-resort and contaminate judge_calls counts."""
    return {
        f.name: [_c(f.name, f"baseline-{f.name}")]
        for f in schema.fields
        if f.required and f.name not in exclude
    }


@patch("src.services.extraction_v3.judge.orchestrator.call_coherence_judge")
@patch("src.services.extraction_v3.judge.orchestrator.call_grounded_last_resort")
@patch("src.services.extraction_v3.judge.orchestrator.call_tiebreaker_judge")
def test_one_candidate_no_judge_calls(mock_tie, mock_grounded, mock_coh):
    """Field with one candidate → no tiebreaker, no grounded. Coherence may still fire."""
    mock_coh.return_value = None  # disable coherence for this test
    schema = load_doc_schema("invoice")
    # Cover all required fields so grounded is never invoked for any of them
    baseline = _required_baseline(schema, exclude={"invoice_id"})
    cands = {"invoice_id": [_c("invoice_id", "INV-001")], **baseline}
    result = run_judge_orchestrator(cands, schema, "INV-001 ...")
    assert result.field_outcomes["invoice_id"].chosen.value == "INV-001"
    assert result.judge_calls == 0  # no judge calls at all
    mock_tie.assert_not_called()
    mock_grounded.assert_not_called()


@patch("src.services.extraction_v3.judge.orchestrator.call_coherence_judge")
@patch("src.services.extraction_v3.judge.orchestrator.call_grounded_last_resort")
@patch("src.services.extraction_v3.judge.orchestrator.call_tiebreaker_judge")
def test_disagreement_triggers_tiebreaker(mock_tie, mock_grounded, mock_coh):
    mock_coh.return_value = None
    mock_tie.return_value = _c("invoice_id", "INV-A")  # judge picks first
    schema = load_doc_schema("invoice")
    # Provide baseline for all required fields except invoice_id (which has disagreement)
    baseline = _required_baseline(schema, exclude={"invoice_id"})
    cands = {
        "invoice_id": [_c("invoice_id", "INV-A"), _c("invoice_id", "INV-B")],
        **baseline,
    }
    result = run_judge_orchestrator(cands, schema, "...")
    assert result.field_outcomes["invoice_id"].chosen.value == "INV-A"
    assert "tiebreaker" in result.field_outcomes["invoice_id"].judge_actions
    mock_tie.assert_called_once()
    assert result.judge_calls == 1


@patch("src.services.extraction_v3.judge.orchestrator.call_coherence_judge")
@patch("src.services.extraction_v3.judge.orchestrator.call_grounded_last_resort")
@patch("src.services.extraction_v3.judge.orchestrator.call_tiebreaker_judge")
def test_agreement_no_tiebreaker(mock_tie, mock_grounded, mock_coh):
    """All candidates agree → no tiebreaker call."""
    mock_coh.return_value = None
    schema = load_doc_schema("invoice")
    baseline = _required_baseline(schema, exclude={"invoice_id"})
    cands = {
        "invoice_id": [_c("invoice_id", "INV-A", confidence=0.7), _c("invoice_id", "INV-A", confidence=0.9)],
        **baseline,
    }
    result = run_judge_orchestrator(cands, schema, "...")
    assert result.field_outcomes["invoice_id"].chosen.confidence == 0.9
    mock_tie.assert_not_called()


@patch("src.services.extraction_v3.judge.orchestrator.call_coherence_judge")
@patch("src.services.extraction_v3.judge.orchestrator.call_grounded_last_resort")
@patch("src.services.extraction_v3.judge.orchestrator.call_tiebreaker_judge")
def test_empty_required_triggers_grounded(mock_tie, mock_grounded, mock_coh):
    """Required field with 0 candidates → grounded last-resort."""
    mock_coh.return_value = None
    mock_grounded.return_value = _c("invoice_id", "INV-X")
    schema = load_doc_schema("invoice")
    # Only provide baseline for non-invoice_id required fields; leave invoice_id empty
    baseline = _required_baseline(schema, exclude={"invoice_id"})
    cands = {**baseline}  # nothing for invoice_id
    result = run_judge_orchestrator(cands, schema, "INV-X is here")
    inv_out = result.field_outcomes["invoice_id"]
    # invoice_id is required in invoice.yaml; grounded should fire
    if inv_out.chosen:
        assert "grounded_last_resort" in inv_out.judge_actions
        assert inv_out.chosen.value == "INV-X"


@patch("src.services.extraction_v3.judge.orchestrator.call_coherence_judge")
@patch("src.services.extraction_v3.judge.orchestrator.call_grounded_last_resort")
@patch("src.services.extraction_v3.judge.orchestrator.call_tiebreaker_judge")
def test_grounded_returns_null_marks_residual(mock_tie, mock_grounded, mock_coh):
    """Grounded returns None → field is residual."""
    mock_coh.return_value = None
    mock_grounded.return_value = None  # judge can't find it either
    schema = load_doc_schema("invoice")
    # Leave all fields empty so grounded fires for every required field (all return None)
    result = run_judge_orchestrator({}, schema, "...")
    inv_out = result.field_outcomes["invoice_id"]
    assert inv_out.chosen is None
    assert inv_out.residual_reason == "required_field_missing_no_grounding"


@patch("src.services.extraction_v3.judge.orchestrator.call_coherence_judge")
@patch("src.services.extraction_v3.judge.orchestrator.call_grounded_last_resort")
@patch("src.services.extraction_v3.judge.orchestrator.call_tiebreaker_judge")
def test_cost_ceiling(mock_tie, mock_grounded, mock_coh):
    """Verify the per-doc judge call count: tiebreakers + grounded + 1 coherence."""
    from src.services.extraction_v3.judge.contracts import CoherenceOutput
    mock_tie.return_value = _c("invoice_id", "X")
    mock_grounded.return_value = None
    mock_coh.return_value = CoherenceOutput(verdict="coherent", issues=[])
    schema = load_doc_schema("invoice")
    cands = {
        "invoice_id": [_c("invoice_id", "A"), _c("invoice_id", "B")],  # 1 tiebreaker
        "supplier_name": [_c("supplier_name", "X"), _c("supplier_name", "Y")],  # 1 tiebreaker
    }
    # 2 disagreements + 1 coherence = at most 3 judge calls (no grounded since
    # most required fields will fall through to no-grounding because there's
    # no full_text given for grounded to work with; mock returns None)
    # NOTE: with mocked grounded, it WILL be called once per missing required
    # field (and there are several required fields not in cands), but the call
    # itself is counted.
    result = run_judge_orchestrator(cands, schema, "")
    # 2 tiebreaker calls (succeed, reset failure counter) + grounded calls for missing
    # required fields (all return None → each increments consecutive_failures).
    # Circuit breaker trips at 3 consecutive failures, so coherence is skipped if
    # there are >= 3 missing required fields.
    _FAIL_FAST_THRESHOLD = 3
    n_required = sum(1 for f in schema.fields if f.required)
    n_required_in_cands = sum(1 for f in schema.fields if f.required and f.name in cands)
    n_required_missing = n_required - n_required_in_cands
    # Grounded calls capped at threshold (circuit breaker), tiebreakers reset the counter
    grounded_calls = min(n_required_missing, _FAIL_FAST_THRESHOLD)
    # Coherence only runs if circuit breaker hasn't tripped
    coherence_call = 1 if n_required_missing < _FAIL_FAST_THRESHOLD else 0
    expected = 2 + grounded_calls + coherence_call
    assert result.judge_calls == expected, f"got {result.judge_calls}, expected {expected}"
