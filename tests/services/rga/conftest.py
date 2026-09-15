"""Fixtures for the RGA suite. No database, no model, no GPU.

Every fixture here is synthetic on purpose. The Fact Pack builder's live queries
are exercised by ``test_exec_summary_live.py``, which is gated on
``PROCWISE_TEST_LIVE_DB=1``; everything else has to run on a laptop with nothing
switched on, or it will not be run.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pytest

from src.services.analytics.models import Confidence
from src.services.rga.models import (
    ChartBlock,
    ChartSeries,
    FactEntry,
    FactPack,
    Finding,
    FindingCode,
    FindingListBlock,
    FormatHint,
    MetricBlock,
    NarrativeBlock,
    Origin,
    ReportAST,
    Section,
    Severity,
    TableBlock,
)
from src.services.rga.style import resolve_style_brief


def make_fact(
    number: int,
    label: str,
    value: Optional[Decimal],
    hint: FormatHint,
    confidence: Confidence,
    *,
    currency: Optional[str] = None,
    unit: Optional[str] = None,
    origin: Origin = Origin.OBSERVED,
) -> FactEntry:
    return FactEntry(
        fact_id=f"F{number:04d}", label=label, value=value, currency=currency,
        unit=unit, format_hint=hint, confidence=confidence, origin=origin,
        provenance_id=f"bp_agent_actions:FP-fixture#F{number:04d}",
        derivation=f"fixture.q{number}",
    )


@pytest.fixture
def facts() -> list[FactEntry]:
    """A measured money figure, a count, a rate, and one that could not be taken."""
    return [
        make_fact(1, "Invoiced spend (GBP)", Decimal("5833817.90"),
                  FormatHint.MONEY, Confidence.CORROBORATED, currency="GBP"),
        make_fact(2, "Deals in period", Decimal(376), FormatHint.INT,
                  Confidence.ASSERTED, unit="deals"),
        make_fact(3, "Three-way match rate", Decimal("29.52"), FormatHint.PCT,
                  Confidence.CORROBORATED),
        make_fact(4, "Realised savings (GBP)", None, FormatHint.TEXT,
                  Confidence.UNASSESSED),
    ]


def make_pack(facts: list[FactEntry], *, pack_id: str = "FP-fixture0001",
              findings: Optional[list[Finding]] = None) -> FactPack:
    return FactPack(
        pack_id=pack_id, report_type_id="exec_procurement_summary",
        scope={"period_label": "2026 Q1", "period_start": "2026-01-01",
               "period_end": "2026-03-31", "currency": "GBP"},
        as_of="2026-03-31",
        generated_at=datetime(2026, 9, 12, tzinfo=timezone.utc),
        generated_by="tests",
        facts=facts,
        findings=findings or [Finding(
            finding_id="FP-fixture0001-FND001",
            code=FindingCode.MEASURE_UNAVAILABLE, severity=Severity.MEDIUM,
            detail="Realised savings: not captured anywhere in the corpus",
            blocks_release=False, fact_id="F0004")],
    )


@pytest.fixture
def pack(facts) -> FactPack:
    return make_pack(facts)


@pytest.fixture
def brief():
    return resolve_style_brief("exec_procurement_summary")


@pytest.fixture
def ast() -> ReportAST:
    """A hand-written tree exercising every block kind."""
    return ReportAST(sections=[
        Section(id="exec_summary", title="Executive summary", blocks=[
            MetricBlock(fact_ref="F0001", emphasis="primary"),
            MetricBlock(fact_ref="F0002"),
            NarrativeBlock(
                text="Invoiced spend was {{F0001}} across {{F0002}} deals.",
                fact_refs=["F0001", "F0002"]),
        ]),
        Section(id="coverage", title="Control coverage", blocks=[
            TableBlock(columns=["Measure", "Value"],
                       rows=[["Three-way match rate", "F0003"],
                             ["Realised savings", "F0004"]]),
            ChartBlock(chart_type="bar", series=[
                ChartSeries(label="Spend", fact_refs=["F0001"]),
                ChartSeries(label="Deals", fact_refs=["F0002"])]),
            FindingListBlock(finding_refs=[]),
        ]),
    ])
