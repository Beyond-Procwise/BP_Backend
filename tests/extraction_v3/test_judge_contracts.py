import pytest
from pydantic import ValidationError
from src.services.extraction_v3.judge import (
    TiebreakerInput, TiebreakerOutput, TiebreakerCandidate,
    GroundedInput, GroundedOutput, GroundedConstraints,
    CoherenceInput, CoherenceOutput, CoherenceIssue,
    InvariantResultSummary,
)


# ---- Tiebreaker tests ---------------------------------------------------

def test_tiebreaker_input_valid():
    payload = {
        "field": "supplier_name",
        "field_type": "string",
        "candidates": [
            {"value": "Acme Ltd", "model": "layoutlmv3", "confidence": 0.81,
             "evidence": "Acme Ltd", "page": 0, "bbox": [10, 20, 100, 40]},
        ],
        "context_text": "..."
    }
    obj = TiebreakerInput(**payload)
    assert obj.candidates[0].value == "Acme Ltd"


def test_tiebreaker_input_rejects_extra_fields():
    payload = {
        "field": "x", "field_type": "string", "candidates": [], "context_text": "",
        "unknown_field": "x",
    }
    with pytest.raises(ValidationError):
        TiebreakerInput(**payload)


def test_tiebreaker_candidate_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        TiebreakerCandidate(
            value="x", model="x", confidence=1.5,
            evidence="x", page=0, bbox=(0, 0, 1, 1),
        )


def test_tiebreaker_candidate_rejects_confidence_negative():
    with pytest.raises(ValidationError):
        TiebreakerCandidate(
            value="x", model="x", confidence=-0.1,
            evidence="x", page=0, bbox=(0, 0, 1, 1),
        )


def test_tiebreaker_output_chosen_index():
    o = TiebreakerOutput(chosen_candidate_index=0, rationale="picked")
    assert o.chosen_candidate_index == 0


def test_tiebreaker_output_null_index():
    o = TiebreakerOutput(chosen_candidate_index=None, rationale="cannot decide")
    assert o.chosen_candidate_index is None


def test_tiebreaker_output_rejects_extra_fields():
    with pytest.raises(ValidationError):
        TiebreakerOutput(chosen_candidate_index=0, rationale="x", unknown="y")


# ---- Grounded tests -----------------------------------------------------

def test_grounded_input_default_constraints():
    o = GroundedInput(
        field="invoice_id", field_type="string",
        field_canonical_labels=["Invoice Number"],
        doc_full_text="...",
    )
    assert o.constraints.must_be_verbatim_substring_of_doc_full_text is True
    assert o.constraints.max_length == 64


def test_grounded_input_custom_constraints():
    constraints = GroundedConstraints(max_length=128)
    o = GroundedInput(
        field="invoice_id", field_type="string",
        field_canonical_labels=["Invoice Number"],
        doc_full_text="...",
        constraints=constraints,
    )
    assert o.constraints.max_length == 128


def test_grounded_output_null_for_no_answer():
    o = GroundedOutput(value=None, evidence_text=None, rationale="not present in doc")
    assert o.is_null() is True


def test_grounded_output_value_with_evidence():
    o = GroundedOutput(value="INV-005-41", evidence_text="INV-005-41", rationale="found")
    assert o.is_null() is False


def test_grounded_output_value_only_not_null():
    o = GroundedOutput(value="INV-005-41", evidence_text=None, rationale="found")
    assert o.is_null() is False


def test_grounded_output_evidence_only_not_null():
    o = GroundedOutput(value=None, evidence_text="INV-005-41", rationale="found")
    assert o.is_null() is False


def test_grounded_output_rejects_extra_fields():
    with pytest.raises(ValidationError):
        GroundedOutput(value="x", evidence_text="x", rationale="x", unknown="y")


# ---- Coherence tests ----------------------------------------------------

def test_coherence_output_coherent():
    o = CoherenceOutput(verdict="coherent")
    assert o.verdict == "coherent"
    assert o.issues == []


def test_coherence_output_incoherent():
    o = CoherenceOutput(verdict="incoherent", issues=[
        CoherenceIssue(field="subtotal", issue="negative value"),
    ])
    assert o.verdict == "incoherent"
    assert len(o.issues) == 1


def test_coherence_output_rejects_invalid_verdict():
    with pytest.raises(ValidationError):
        CoherenceOutput(verdict="maybe")


def test_coherence_output_rejects_extra_fields():
    with pytest.raises(ValidationError):
        CoherenceOutput(verdict="coherent", unknown="x")


def test_coherence_input_minimal():
    o = CoherenceInput(
        doc_type="invoice",
        extracted_record={"invoice_id": "X"},
        invariant_results=[],
    )
    assert o.doc_type == "invoice"
    assert len(o.invariant_results) == 0


def test_coherence_input_with_invariants():
    o = CoherenceInput(
        doc_type="invoice",
        extracted_record={"invoice_id": "X", "subtotal": 100.0},
        invariant_results=[
            InvariantResultSummary(name="subtotal_closure", passed=True),
            InvariantResultSummary(
                name="line_sum_match",
                passed=False,
                delta=5.0,
                severity="warning",
                message="sum mismatch",
            ),
        ],
    )
    assert len(o.invariant_results) == 2
    assert o.invariant_results[0].passed is True
    assert o.invariant_results[1].delta == 5.0


def test_invariant_result_summary_optional_fields():
    o = InvariantResultSummary(name="test", passed=True)
    assert o.delta is None
    assert o.severity is None
    assert o.message is None


def test_coherence_issue_basic():
    issue = CoherenceIssue(field="tax_amount", issue="exceeds gross")
    assert issue.field == "tax_amount"
    assert issue.issue == "exceeds gross"


def test_coherence_issue_rejects_extra_fields():
    with pytest.raises(ValidationError):
        CoherenceIssue(field="x", issue="y", unknown="z")
