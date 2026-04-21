from pathlib import Path

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_extract_end_to_end_on_newport_pdf(monkeypatch):
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    from src.services.structural_extractor import extract
    # Stub LLM to avoid real Ollama calls during tests
    from src.services.structural_extractor import llm_fallback
    monkeypatch.setattr(llm_fallback, "_call_llm", lambda prompt: "{}")
    result = extract(FIX.read_bytes(), "INV600254.pdf", "Invoice")
    assert result.header["invoice_id"].value == "INV600254"
    assert abs(result.header["invoice_amount"].value - 8333.0) < 0.01
    assert result.attempts >= 1
