from pathlib import Path

from src.services.structural_extractor.parsing.pdf_parser import parse_pdf
from src.services.structural_extractor.retry.attempts import run_attempt_nlu
from src.services.structural_extractor.retry.state import RetryState

FIX = Path(__file__).parent / "fixtures/docs/INV600254.pdf"


def test_attempt_2_runs_without_nlu_model_load(monkeypatch):
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    state = RetryState(
        doc=doc, doc_type="Invoice",
        target_fields={"invoice_id"}, unresolved={"invoice_id"},
    )
    # Monkey-patch NER to avoid loading real model
    from src.services.structural_extractor.nlu import ner
    monkeypatch.setattr(ner, "run", lambda text: [], raising=False)
    output = run_attempt_nlu(state, attempt_no=2)
    assert output.attempt == 2
    assert output.source == "nlu_ner"


def test_attempt_3_source_is_table():
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    state = RetryState(
        doc=doc, doc_type="Invoice",
        target_fields={"invoice_id"}, unresolved={"invoice_id"},
    )
    output = run_attempt_nlu(state, attempt_no=3)
    assert output.attempt == 3
    assert output.source == "nlu_table"


def test_attempt_4_source_is_layout():
    if not FIX.exists():
        import pytest
        pytest.skip("fixture missing")
    doc = parse_pdf(FIX.read_bytes(), "INV600254.pdf")
    state = RetryState(
        doc=doc, doc_type="Invoice",
        target_fields={"invoice_id"}, unresolved={"invoice_id"},
    )
    output = run_attempt_nlu(state, attempt_no=4)
    assert output.attempt == 4
    assert output.source == "nlu_layout"
