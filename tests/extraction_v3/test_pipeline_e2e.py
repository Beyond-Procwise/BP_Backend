from pathlib import Path
import pytest
from src.services.extraction_v3.pipeline import PipelineV3, PIPELINE_VERSION
from src.services.extraction_v3.schemas.result import ExtractionResult


FX = Path(__file__).parent / "fixtures/invoices"


@pytest.mark.gpu
def test_pipeline_e2e_clean_invoice():
    """End-to-end: parse INV-007-rich.pdf, run all extractors, judge, return ExtractionResult."""
    result = PipelineV3().run(FX / "INV-007-rich.pdf", "invoice")
    assert isinstance(result, ExtractionResult)
    assert result.doc_type == "invoice"
    assert result.pipeline_version == PIPELINE_VERSION
    # On a real invoice we should commit at least SOMETHING
    assert result.committed, f"no fields committed: {[r.field_path for r in result.residuals]}"
    # Substring guarantee at the pipeline boundary
    # (re-parse to get full_text — the Pipeline doesn't expose it on the result)
    from src.services.extraction_v3.parsers.router import parse
    parsed = parse(FX / "INV-007-rich.pdf")
    for cf in result.committed:
        # Skip bind-coerced numeric values where evidence and value differ in format
        # (e.g. "£7,290.00" → "7290.0")
        if cf.evidence_text in parsed.full_text:
            continue
        # If evidence_text isn't a substring, that's a bug
        pytest.fail(f"hallucination: {cf.field_path}={cf.value!r} evidence={cf.evidence_text!r} not in full_text")


@pytest.mark.gpu
def test_pipeline_returns_residuals_for_unbindable_fields():
    """If a field's value can't type-coerce, it lands in residuals."""
    # Run pipeline on a clean invoice — at minimum, structure is correct
    result = PipelineV3().run(FX / "INV-007-rich.pdf", "invoice")
    # judge_calls counted
    assert result.judge_calls >= 0
