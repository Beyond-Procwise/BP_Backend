from src.services.structural_extractor.retry.state import AttemptOutput, RetryState


def test_attempt_output_basic():
    o = AttemptOutput(
        attempt=1, source="structural", extracted={}, line_items=None,
        validation_failures=[], residual_unresolved=["invoice_date"],
        latency_ms=100,
    )
    assert o.attempt == 1
    assert o.source == "structural"


def test_retry_state_residual():
    s = RetryState(
        doc=None, doc_type="Invoice",
        target_fields={"invoice_id", "invoice_date"},
        attempts=[], accepted_header={}, accepted_line_items=None,
        unresolved={"invoice_id", "invoice_date"},
    )
    assert "invoice_date" in s.unresolved


def test_retry_state_residual_fields_method():
    s = RetryState(
        doc=None, doc_type="Invoice",
        target_fields={"a", "b", "c"},
        unresolved={"a", "b", "c"},
    )
    # Nothing accepted yet
    assert s.residual_fields() == {"a", "b", "c"}
    # Accept "a"
    from src.services.structural_extractor.types import ExtractedValue
    s.accepted_header["a"] = ExtractedValue(value="x", provenance="extracted")
    assert s.residual_fields() == {"b", "c"}
