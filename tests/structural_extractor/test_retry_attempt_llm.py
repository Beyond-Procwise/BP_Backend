from src.services.structural_extractor.parsing.model import BBox, ParsedDocument, Token
from src.services.structural_extractor.retry.attempts import run_attempt_llm
from src.services.structural_extractor.retry.state import RetryState


def test_llm_attempt_grounds_values(monkeypatch):
    from src.services.structural_extractor import llm_fallback as lf
    monkeypatch.setattr(
        lf, "_call_llm",
        lambda prompt: '{"invoice_id": {"value": "INV-999", "anchor": "Invoice INV-999"}}',
    )
    toks = [
        Token(text="Invoice", anchor=BBox(1, 0, 0, 0, 0), order=0),
        Token(text="INV-999", anchor=BBox(1, 50, 0, 0, 0), order=1),
    ]
    doc = ParsedDocument(
        source_format="pdf", filename="", tokens=toks, regions=[], tables=[],
        pages_or_sheets=1, full_text="Invoice INV-999", raw_bytes=b"",
    )
    state = RetryState(
        doc=doc, doc_type="Invoice",
        target_fields={"invoice_id"}, unresolved={"invoice_id"},
    )
    output = run_attempt_llm(state, attempt_no=5)
    assert output.attempt == 5
    assert output.source == "llm_fallback"
    assert "invoice_id" in output.extracted
    assert output.extracted["invoice_id"].value == "INV-999"


def test_llm_attempt_short_circuits_when_no_residual(monkeypatch):
    # If unresolved set is empty, no LLM call should happen
    called = {"n": 0}
    from src.services.structural_extractor import llm_fallback as lf

    def _fail(prompt):
        called["n"] += 1
        return "{}"

    monkeypatch.setattr(lf, "_call_llm", _fail)
    doc = ParsedDocument(
        source_format="pdf", filename="", tokens=[], regions=[], tables=[],
        pages_or_sheets=1, full_text="", raw_bytes=b"",
    )
    state = RetryState(
        doc=doc, doc_type="Invoice",
        target_fields={"invoice_id"}, unresolved=set(),
    )
    output = run_attempt_llm(state, attempt_no=5)
    assert called["n"] == 0
    assert output.source == "llm_fallback"
    assert output.extracted == {}


def test_llm_attempt_drops_ungrounded(monkeypatch):
    from src.services.structural_extractor import llm_fallback as lf
    monkeypatch.setattr(
        lf, "_call_llm",
        lambda prompt: '{"invoice_id": {"value": "FAKE123", "anchor": "x"}}',
    )
    doc = ParsedDocument(
        source_format="pdf", filename="", tokens=[], regions=[], tables=[],
        pages_or_sheets=1, full_text="nothing useful here", raw_bytes=b"",
    )
    state = RetryState(
        doc=doc, doc_type="Invoice",
        target_fields={"invoice_id"}, unresolved={"invoice_id"},
    )
    output = run_attempt_llm(state, attempt_no=6)
    assert "invoice_id" not in output.extracted
    assert "invoice_id" in output.residual_unresolved
