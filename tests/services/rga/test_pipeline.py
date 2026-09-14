"""The state machine end to end: what runs, in what order, and what stops it.

No database and no model. A stub builder stands in for the deterministic
queries so the whole path — scope, pack, style, compose, render, post-check,
release — can be exercised on a laptop with nothing switched on.

The event-ordering test is the point of the file. §7 lists the events a run must
emit; a run that quietly stopped emitting one would still produce a correct
report, and nobody would notice until they needed the trail.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest

from src.services.analytics.models import Confidence
from src.services.rga import audit, factpack
from src.services.rga.factpack import FactBuilder, register
from src.services.rga.models import (
    FormatHint,
    MetricBlock,
    NarrativeBlock,
    ReportAST,
    Section,
    TableBlock,
)
from src.services.rga.pipeline import ReportRun, generate_report

SCOPE = {"period_label": "2026 Q1", "period_start": "2026-01-01",
         "period_end": "2026-03-31", "currency": "GBP"}


@pytest.fixture
def stub_type():
    """A report type whose facts come from nowhere but this function."""
    name = "_pipeline_test_type"

    @register(name)
    def _build(fb: FactBuilder) -> None:
        fb.add(label="Invoiced spend (GBP)", value=Decimal("5833817.90"),
               derivation="stub.spend", confidence=Confidence.CORROBORATED,
               format_hint=FormatHint.MONEY, currency="GBP")
        fb.add(label="Deals in period", value=Decimal(376),
               derivation="stub.deals", confidence=Confidence.ASSERTED,
               format_hint=FormatHint.INT, unit="deals")

    yield name
    factpack._BUILDERS.pop(name, None)


@pytest.fixture
def hand_written():
    return ReportAST(sections=[
        Section(id="exec_summary", title="Executive summary", blocks=[
            MetricBlock(fact_ref="F0001", emphasis="primary"),
            NarrativeBlock(text="Invoiced spend was {{F0001}} across {{F0002}} deals.",
                           fact_refs=["F0001", "F0002"]),
        ]),
    ])


class Recorder:
    def __init__(self, explode_on=None):
        self.rows = []
        self.explode_on = explode_on

    def __call__(self, **kwargs):
        if kwargs.get("action_type") == self.explode_on:
            raise RuntimeError("the audit spine is unreachable")
        self.rows.append(kwargs)

    @property
    def order(self):
        return [r["action_type"] for r in self.rows]


def run(stub_type, ast, **kwargs) -> ReportRun:
    return generate_report(stub_type, scope=SCOPE, as_of="2026-03-31", ast=ast,
                           **kwargs)


class TestTheHappyPath:
    def test_a_clean_report_reaches_release(self, stub_type, hand_written):
        result = run(stub_type, hand_written, emit_audit=False)

        assert result.released is True
        assert result.blocked is False
        assert result.stage_reached == "RELEASE"
        assert result.artefact is not None
        assert result.result.passed

    def test_the_run_id_is_the_pack_id(self, stub_type, hand_written):
        """A run and the snapshot it was computed from are the same thing."""
        result = run(stub_type, hand_written, emit_audit=False)

        assert result.run_id == result.pack.pack_id


class TestEventsEndToEnd:
    def test_every_stage_emits_its_event_in_order(self, stub_type, hand_written):
        writer = Recorder()
        run(stub_type, hand_written, writer=writer)

        assert writer.order == [
            audit.FACTPACK_BUILT,     # emitted inside build_fact_pack
            audit.SCOPE_RESOLVED,
            audit.STYLEBRIEF_RESOLVED,
            audit.COMPOSED,
            audit.RENDERED,
            audit.POSTCHECK_PASSED,
            audit.RELEASED,
        ]

    def test_every_event_carries_the_run_id_and_the_pack_hash(
            self, stub_type, hand_written):
        writer = Recorder()
        result = run(stub_type, hand_written, writer=writer)

        for row in writer.rows:
            if row["action_type"] == audit.FACTPACK_BUILT:
                continue  # factpack writes its own shape, tested separately
            assert row["trace_id"] == result.run_id
            assert row["details"]["run_id"] == result.run_id
            assert row["details"]["pack_hash"] == result.pack.hash

    def test_a_run_that_used_no_model_says_so(self, stub_type, hand_written):
        writer = Recorder()
        run(stub_type, hand_written, writer=writer)

        composed = [r for r in writer.rows if r["action_type"] == audit.COMPOSED][0]
        assert composed["details"]["model"] is None
        assert "no model" in composed["summary"]

    def test_the_style_event_records_what_it_could_not_resolve(
            self, stub_type, hand_written):
        writer = Recorder()
        run(stub_type, hand_written, writer=writer)

        row = [r for r in writer.rows
               if r["action_type"] == audit.STYLEBRIEF_RESOLVED][0]
        assert "business_unit" in row["details"]["unresolved_scopes"]
        assert row["details"]["style_version"]


class TestItFailsClosed:
    def test_a_blocked_report_is_not_released(self, stub_type):
        """A fabricated figure in a column header — author text, so the type
        allows it and the post-check on the rendered bytes is what catches it."""
        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            MetricBlock(fact_ref="F0001"),
            TableBlock(columns=["Target £9.9M"], rows=[["F0001"]])])])

        result = run(stub_type, ast, emit_audit=False)

        assert result.released is False
        assert result.blocked is True
        assert result.stage_reached == "POST_CHECK"
        assert any("9.9" in r for r in result.blocking_reasons())

    def test_a_blocked_run_still_produces_an_artefact_to_look_at(self, stub_type):
        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            MetricBlock(fact_ref="F0001"),
            TableBlock(columns=["Target £9.9M"], rows=[["F0001"]])])])

        result = run(stub_type, ast, emit_audit=False)

        assert result.artefact is not None
        assert len(result.artefact.content) > 0

    def test_a_blocked_run_emits_no_release_event(self, stub_type):
        ast = ReportAST(sections=[Section(id="s", title="S", blocks=[
            MetricBlock(fact_ref="F0001"),
            TableBlock(columns=["Target £9.9M"], rows=[["F0001"]])])])
        writer = Recorder()

        run(stub_type, ast, writer=writer)

        assert audit.POSTCHECK_FAILED in writer.order
        assert audit.RELEASED not in writer.order

    def test_a_renderer_fault_blocks_the_report_rather_than_raising(
            self, stub_type, hand_written):
        """generate_report promises never to raise for a report that merely
        failed. Live, a degenerate table took python-pptx down with a
        ZeroDivisionError and the whole call went with it."""
        class Exploding:
            @staticmethod
            def render(*a, **k):
                raise ZeroDivisionError("integer division or modulo by zero")

        writer = Recorder()
        result = run(stub_type, hand_written, renderer=Exploding, writer=writer)

        assert result.released is False
        assert result.stage_reached == "RENDER"
        assert result.artefact is None
        assert any("ZeroDivisionError" in r for r in result.blocking_reasons())
        assert audit.RELEASED not in writer.order

    def test_a_failed_composition_stops_before_rendering(self, stub_type):
        def explode(prompt, *, schema=None, temperature=0.0):
            raise ConnectionError("ollama is not listening")

        writer = Recorder()
        result = generate_report(stub_type, scope=SCOPE, as_of="2026-03-31",
                                 generate=explode, writer=writer)

        assert result.released is False
        assert result.stage_reached == "COMPOSE"
        assert result.artefact is None
        assert audit.RENDERED not in writer.order
        assert audit.RELEASED not in writer.order

    def test_a_release_whose_audit_cannot_be_written_does_not_happen(
            self, stub_type, hand_written):
        """The one event that must raise rather than swallow. An unauditable
        release is not a release."""
        writer = Recorder(explode_on=audit.RELEASED)

        with pytest.raises(RuntimeError, match="audit spine is unreachable"):
            run(stub_type, hand_written, writer=writer)

        assert audit.RELEASED not in writer.order


class TestRegeneration:
    def test_rerunning_produces_an_identical_artefact_never_a_patch(
            self, stub_type, hand_written):
        first = run(stub_type, hand_written, emit_audit=False)
        second = run(stub_type, hand_written, emit_audit=False)

        assert second.pack.hash == first.pack.hash
        assert second.artefact.content == first.artefact.content
        # Frozen all the way down: nothing could have been edited in place.
        with pytest.raises(Exception):
            first.pack.facts[0].value = Decimal("1")

    def test_a_different_scope_is_a_different_run(self, stub_type, hand_written):
        first = run(stub_type, hand_written, emit_audit=False)
        second = generate_report(
            stub_type, scope=dict(SCOPE, period_label="2025 Q1",
                                  period_start="2025-01-01"),
            as_of="2026-03-31", ast=hand_written, emit_audit=False)

        assert second.run_id != first.run_id
