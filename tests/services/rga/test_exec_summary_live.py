"""The exec summary against the real corpus. The Phase 1 exit test.

    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/services/rga/test_exec_summary_live.py

Skipped by default: the suite runs against an in-memory fake database, and
these assertions are about figures that only exist in Postgres. A silent skip is
the convention here (``tests/services/test_canonical_masters.py``) — the
alternative is a suite that fails on a laptop with nothing switched on.

WHICH CORPUS. The brief's exit test names "Assurity Group Q2". Assurity Ltd is
this platform's own buyer identity, not a dataset, and no such dataset exists —
see docs/rga/discovery.md §3.6. These run against the corpus that does exist
(Beyond Procurement Group plc, 5,042 deals), and the report says so on its face.
"""

from __future__ import annotations

import os

import pytest

from src.services.analytics.models import Confidence
from src.services.rga.factpack import build_fact_pack
from src.services.rga.models import (
    FactPack,
    FindingListBlock,
    MetricBlock,
    NarrativeBlock,
    ReportAST,
    Section,
    TableBlock,
)
from src.services.rga.render import pptx as renderer
from src.services.rga import postcheck
from src.services.rga.style import resolve_style_brief

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in (
    "1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

SCOPE = {"period_label": "2026 Q1", "period_start": "2026-01-01",
         "period_end": "2026-03-31", "currency": "GBP"}
AS_OF = "2026-03-31"


@pytest.fixture(scope="module")
def live_pack():
    import src.services.rga  # noqa: F401  registers the builders

    return build_fact_pack("exec_procurement_summary", scope=SCOPE, as_of=AS_OF,
                           emit_audit=False)


@pytest.fixture(scope="module")
def live_ast():
    """Hand-written. No model was involved in producing this tree."""
    return ReportAST(sections=[
        Section(id="exec_summary", title="Executive summary", blocks=[
            MetricBlock(fact_ref="F0003", emphasis="primary"),
            MetricBlock(fact_ref="F0001"),
            MetricBlock(fact_ref="F0002"),
            NarrativeBlock(
                text="Invoiced spend for the period was {{F0003}}, across "
                     "{{F0001}} deals with {{F0002}} suppliers.",
                fact_refs=["F0003", "F0001", "F0002"]),
        ]),
        Section(id="coverage", title="Control coverage", blocks=[
            TableBlock(columns=["Measure", "Value"],
                       rows=[["Three-way match rate", "F0004"],
                             ["Average quote-to-PO cycle", "F0005"]]),
        ]),
        Section(id="opportunities", title="Opportunities", blocks=[
            MetricBlock(fact_ref="F0006", emphasis="primary"),
            TableBlock(columns=["Measure", "Value"],
                       rows=[["Identified value", "F0007"],
                             ["Realised savings", "F0008"]]),
            FindingListBlock(finding_refs=[]),
        ]),
    ])


class TestTheFactsAreMeasured:
    def test_every_fact_carries_provenance_confidence_and_origin(self, live_pack):
        assert live_pack.facts, "the builder produced no facts"
        for entry in live_pack.facts:
            assert entry.provenance_id.strip()
            assert entry.derivation.strip()
            assert entry.confidence in Confidence
            assert entry.origin is not None

    def test_spend_is_stated_in_one_currency_or_not_at_all(self, live_pack):
        """The corpus holds five currencies. A native sum would be arithmetic on
        unlike units, so the figure is either converted or absent."""
        spend = [f for f in live_pack.facts if f.label.startswith("Invoiced spend")]
        assert len(spend) == 1
        entry = spend[0]
        if entry.value is not None:
            assert entry.currency == "GBP"
            assert entry.confidence is Confidence.CORROBORATED
            assert "rates as of" in entry.derivation
        else:
            assert entry.confidence is Confidence.UNASSESSED

    def test_an_unmeasurable_figure_is_unassessed_and_never_zero(self, live_pack):
        """Realised savings are captured nowhere in this corpus (0 of 308)."""
        realised = live_pack.fact_by_label("Realised savings (GBP)")
        assert realised is not None
        assert realised.value is None
        assert realised.confidence is Confidence.UNASSESSED
        assert realised.display == "—"

    def test_a_measured_zero_is_not_the_same_as_unmeasured(self, live_pack):
        """No opportunity was detected in Q1 2026. The COUNT is a real zero;
        the value it would have had is not."""
        count = live_pack.fact_by_label("Opportunities identified")
        value = live_pack.fact_by_label("Identified value (GBP)")

        assert count.value == 0 and count.confidence is Confidence.ASSERTED
        assert value.value is None and value.confidence is Confidence.UNASSESSED


class TestPhase1ExitTest:
    def test_the_pack_is_deterministic(self, live_pack):
        again = build_fact_pack("exec_procurement_summary", scope=SCOPE,
                                as_of=AS_OF, emit_audit=False)
        assert again.hash == live_pack.hash
        assert again.pack_id == live_pack.pack_id

    def test_the_report_renders_and_the_post_check_passes(self, live_pack, live_ast):
        brief = resolve_style_brief("exec_procurement_summary")
        artefact = renderer.render(live_ast, live_pack, brief)

        result = postcheck.run(artefact, live_pack, live_ast, brief,
                               emit_audit=False)

        assert result.passed, [f"{f.code.value}: {f.detail}" for f in result.blocking]

    def test_it_regenerates_byte_identically_from_stored_inputs(
            self, live_pack, live_ast):
        """DoD12, with no model anywhere in the path."""
        brief = resolve_style_brief("exec_procurement_summary")
        original = renderer.render(live_ast, live_pack, brief)

        reloaded = FactPack.from_stored(live_pack.stored())
        again = renderer.render(
            ReportAST.model_validate(live_ast.model_dump(mode="json")),
            reloaded, brief)

        assert again.content == original.content
