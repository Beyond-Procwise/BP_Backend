"""End-to-end test: template store closes the supplier_name gap.

Demonstrates the V2 learning loop:
    1. First pass — supplier_name abstains (no consensus)
    2. User-provided correction is recorded into the template store
    3. Second pass — supplier_name commits via template + consensus
"""
from __future__ import annotations

import pytest
from pathlib import Path

from src.services.extraction_v2.fingerprint import compute_fingerprint
from src.services.extraction_v2.pipeline import ExtractionPipelineV2
from src.services.extraction_v2.template_store import InMemoryTemplateStore
from src.services.structural_extractor.parsing import parse


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "structural_extractor" / "fixtures" / "docs"
AQUARIUS = FIXTURE_DIR / "AQUARIUS INV-25-050 for PO508084 .pdf"


@pytest.mark.skipif(not AQUARIUS.exists(), reason="AQUARIUS fixture missing")
class TestTemplateLearningLoop:
    def test_supplier_abstains_on_first_pass(self):
        store = InMemoryTemplateStore()
        pipeline = ExtractionPipelineV2(template_store=store)
        doc = parse(AQUARIUS.read_bytes(), AQUARIUS.name)

        result = pipeline.extract(doc, "Invoice")

        assert "supplier_name" not in result.committed
        # The doc still produced a fingerprint; template_used = False
        assert result.fingerprint is not None
        assert result.template_used is False

    def test_recorded_correction_commits_supplier_on_replay(self):
        store = InMemoryTemplateStore()
        pipeline = ExtractionPipelineV2(template_store=store)
        doc = parse(AQUARIUS.read_bytes(), AQUARIUS.name)

        # First pass — produce the fingerprint and observe the gap
        first = pipeline.extract(doc, "Invoice")
        fingerprint = first.fingerprint
        assert "supplier_name" not in first.committed

        # User corrects the gap via the review queue / onboarding UI
        store.record_correction(
            fingerprint=fingerprint,
            field="supplier_name",
            value="Aquarius Marketing Ltd",
            confidence=0.95,
            label="From",
            doc_type="Invoice",
            vendor_name="Aquarius",
        )

        # Second pass on a doc with the same layout — supplier commits
        second = pipeline.extract(doc, "Invoice")
        assert second.template_used is True
        assert "supplier_name" in second.committed
        assert second.committed["supplier_name"].value == "Aquarius Marketing Ltd"
        assert second.committed["supplier_name"].confidence >= 0.75

    def test_record_success_increments_template_count(self):
        store = InMemoryTemplateStore()
        pipeline = ExtractionPipelineV2(template_store=store)
        doc = parse(AQUARIUS.read_bytes(), AQUARIUS.name)

        # Bootstrap a template via correction
        first = pipeline.extract(doc, "Invoice")
        store.record_correction(
            fingerprint=first.fingerprint,
            field="supplier_name", value="Aquarius",
            confidence=0.95, doc_type="Invoice",
        )

        # Run twice more — each should increment success_count
        pipeline.extract(doc, "Invoice")
        pipeline.extract(doc, "Invoice")

        t = store.get(first.fingerprint)
        assert t is not None
        assert t.success_count >= 2
