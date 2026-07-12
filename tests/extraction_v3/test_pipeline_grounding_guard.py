"""Integration test: PipelineV3 demotes hallucinated header fields (FINDINGS.md F1).

Stubs only the VLM extractor (which needs a GPU) so the real pipeline —
parse → judge → grounding guard → result — runs end to end. Confirms that a
value absent from the document is routed to residuals with reason
``ungrounded_value`` while a value present in the document is committed.
"""
from pathlib import Path

import pytest

from src.services.extraction_v3.schemas.candidate import Candidate
from src.services.extraction_v3.schemas.parsed_document import ParsedDocument
import src.services.extraction_v3.extractors.vlm as vlm_mod
import src.services.extraction_v3.pipeline as pipeline_mod
from src.services.extraction_v3.pipeline import PipelineV3

_FULL_TEXT = (
    "ACME Industries Ltd\n"
    "Invoice Number: INV-77001\n"
    "Invoice Date: 15 Feb 2024\n"
    "Grand Total: £1,200.00\n"
)


@pytest.fixture
def _stub(monkeypatch):
    # Parse returns a fixed ParsedDocument regardless of the (dummy) path.
    def fake_parse(path):
        return ParsedDocument(
            source_path=str(path), file_format="pdf-native", pages=[],
            full_text=_FULL_TEXT, parser_backend="stub", parser_confidence=1.0,
        )
    monkeypatch.setattr(pipeline_mod, "parse_document", fake_parse)

    def fake_vlm(parsed, schema, source_file=""):
        return [
            # Grounded: appears verbatim in the document
            Candidate(field="invoice_id", value="INV-77001", page=1, bbox=(0.0, 0.0, 1.0, 1.0),
                      evidence_text="Invoice Number: INV-77001", model="qwen_vlm", confidence=0.9),
            # Hallucinated: this supplier appears nowhere in the document
            Candidate(field="supplier_name", value="Globex Corporation", page=1, bbox=(0.0, 0.0, 1.0, 1.0),
                      evidence_text="Globex Corporation", model="qwen_vlm", confidence=0.9),
        ]
    monkeypatch.setattr(vlm_mod, "extract_with_vlm", fake_vlm)


def test_pipeline_demotes_hallucinated_field(_stub, tmp_path):
    doc = tmp_path / "dummy_invoice.pdf"
    doc.write_bytes(b"%PDF-1.4 dummy")

    result = PipelineV3().run(doc, "invoice")

    committed = {cf.field_path: cf for cf in result.committed}
    residual_reasons = {r.field_path: r.reason for r in result.residuals}

    # grounded field survives
    assert "invoice_id" in committed
    # hallucinated field is blocked and routed to review
    assert "supplier_name" not in committed
    assert residual_reasons.get("supplier_name") == "ungrounded_value"
