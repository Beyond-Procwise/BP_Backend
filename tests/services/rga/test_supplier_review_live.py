"""The supplier review against the real corpus: it measures, draws and passes its checks.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/rga/test_supplier_review_live.py
"""
from __future__ import annotations

import os

import pytest

from src.services.analytics.models import Confidence
from src.services.rga import postcheck
from src.services.rga.factpack import build_fact_pack
from src.services.rga.models import (ChartBlock, ChartSeries, MetricBlock, NarrativeBlock,
                                     ReportAST, Section, TableBlock)
from src.services.rga.render import html as page_renderer
from src.services.rga.render import pptx as deck_renderer
from src.services.rga.style import resolve_style_brief

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

SCOPE = {"period_label": "2026 Q2", "period_start": "2026-04-01",
         "period_end": "2026-06-30", "currency": "GBP"}


@pytest.fixture(scope="module")
def pack():
    import src.services.rga  # noqa: F401
    return build_fact_pack("supplier_criticality_review", scope=SCOPE, as_of="2026-09-24",
                           emit_audit=False)


def test_it_measures_the_quarter(pack):
    by = {f.label: f for f in pack.facts}
    assert by["Suppliers transacted with"].value > 0
    assert by["Supplier criticality rating"].confidence is Confidence.UNASSESSED
    named = [f for f in pack.facts if f.label.startswith("Spend with ")]
    assert 1 <= len(named) <= 5
    assert pack.hash == build_fact_pack("supplier_criticality_review", scope=SCOPE,
                                        as_of="2026-09-24", emit_audit=False).hash


def test_a_report_naming_suppliers_with_digits_passes_its_checks(pack):
    """Names like 'Ashcroft Associates 10' ride on figure labels, which the check reads as
    references -- the review would otherwise be blocked for naming its own suppliers."""
    top = [f.fact_id for f in pack.facts if f.label.startswith("Spend with ")]
    ast = ReportAST(sections=[
        Section(id="summary", title="Summary", blocks=[
            MetricBlock(fact_ref="F0002", emphasis="primary"), MetricBlock(fact_ref="F0001"),
            NarrativeBlock(text="Spend of {{F0002}} went to {{F0001}} suppliers; the largest "
                                "took {{F0003}} of it.", fact_refs=["F0002", "F0001", "F0003"])]),
        Section(id="top_suppliers", title="Largest suppliers", blocks=[
            *[MetricBlock(fact_ref=r) for r in top],
            ChartBlock(chart_type="bar", series=[ChartSeries(label="Spend", fact_refs=top)])]),
        Section(id="gaps", title="What is not measured", blocks=[
            TableBlock(columns=["Measure", "Value"],
                       rows=[["Criticality rating", "F0007"], ["Single sourcing", "F0008"],
                             ["Contract coverage", "F0009"]])]),
    ])
    brief = resolve_style_brief("supplier_criticality_review")
    for renderer in (deck_renderer, page_renderer):
        art = renderer.render(ast, pack, brief, title="Supplier criticality review")
        result = postcheck.run(art, pack, ast, brief, emit_audit=False)
        assert result.passed, (renderer.__name__, [f.detail for f in result.blocking])
