from pathlib import Path

from src.services.structural_extractor.discovery.schema import fields_for
from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from src.services.structural_extractor.retry.attempts import run_attempt_1
from src.services.structural_extractor.retry.state import RetryState

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_attempt_1_on_newport_pdf():
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    target = set(fields_for("Invoice"))
    state = RetryState(
        doc=doc, doc_type="Invoice", target_fields=target, unresolved=set(target),
    )
    output = run_attempt_1(state)
    assert output.attempt == 1
    assert output.source == "structural"
    # Should have found at least invoice_id, amounts, currency
    assert "invoice_id" in output.extracted
    assert output.extracted["invoice_id"].value == "INV600254"


def test_attempt_1_residual_is_sorted_list():
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    # Use a target that includes a definitely-missing field
    target = {"invoice_id", "definitely_not_a_field"}
    state = RetryState(
        doc=doc, doc_type="Invoice", target_fields=target, unresolved=set(target),
    )
    output = run_attempt_1(state)
    assert "definitely_not_a_field" in output.residual_unresolved
    assert output.residual_unresolved == sorted(output.residual_unresolved)
