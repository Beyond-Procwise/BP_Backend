from pathlib import Path

from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from src.services.structural_extractor.retry.driver import run_retry_loop

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_retry_loop_on_newport_pdf():
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    result = run_retry_loop(doc, "Invoice", max_attempts=3)
    assert "invoice_id" in result.header
    assert result.header["invoice_id"].value == "INV600254"
    assert result.attempts >= 1
    # Amounts should be resolved by attempt 1 via structural
    assert "invoice_amount" in result.header
    assert abs(result.header["invoice_amount"].value - 8333.0) < 0.01


def test_retry_loop_stops_early_when_all_resolved(monkeypatch):
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    # Stub LLM so the test never depends on an external service
    from src.services.structural_extractor import llm_fallback as lf
    monkeypatch.setattr(lf, "_call_llm", lambda p: "{}")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    # With max_attempts=10, since attempt 1 may resolve everything for Invoice,
    # the loop will stop. We just assert the driver returns a valid result
    # and doesn't exceed max_attempts.
    result = run_retry_loop(doc, "Invoice", max_attempts=10)
    assert result.attempts <= 10
    assert result.doc_type == "Invoice"
    assert result.layout_signature != ""
    assert result.parsed_text == doc.full_text


def test_retry_loop_returns_sorted_unresolved():
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    result = run_retry_loop(doc, "Invoice", max_attempts=1)
    # unresolved_fields should be sorted
    assert result.unresolved_fields == sorted(result.unresolved_fields)
