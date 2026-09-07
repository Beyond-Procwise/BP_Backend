"""The evaluation record has to outlive the process that made it.

Every `evaluate` call builds a full `EvaluationRecord` and emits it. Until this
was wired, the default sink was `MemoryAuditSink` -- a 5,000-entry ring buffer --
and `DbAuditSink` was installed nowhere, so nothing ever reached
`proc.bp_agent_actions`. The records were correct and then thrown away.

ADR 0002 D6 explains why the database is not the *default*: a formula run over
pairs would flood the audit table. That reasoning holds for the default; it is
not an argument against installing the durable sink in the API process, where
the measured volume is a handful of evaluations per request (benchmark's
per-deal loop averages 4.5 quote lines, p95 of 9, worst observed 49).

Two properties matter more than the wiring itself, and both get a test that has
been watched failing:

  * installing twice must not stack, because a re-entrant lifespan would
    otherwise bury the original sink and make restoration impossible;
  * a database failure must never take down the calculation being audited.
"""
from __future__ import annotations

from datetime import date

import pytest

from src.services.formulas import (
    DbAuditSink, GoldenVector, MemoryAuditSink, Output, Term, evaluate, formula,
    get_audit_sink, set_audit_sink,
)
from src.services.formulas.audit import install_db_audit_sink
from src.services.formulas.contract import RATIO
from src.services.formulas.registry import REGISTRY

_FROM = date(2026, 9, 5)


@pytest.fixture
def restore_sink():
    previous = get_audit_sink()
    yield
    set_audit_sink(previous)


@pytest.fixture
def clean_registry():
    before = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(before)


@pytest.fixture
def a_formula(clean_registry):
    @formula(
        "t.sink", version="1.0.0", owner="tests", purpose="fixture",
        effective_from=_FROM,
        inputs=[Term("x", RATIO, minimum=0.0, maximum=1.0)],
        output=Output("float", RATIO, "x doubled"),
        golden=[GoldenVector(inputs={"x": 0.5}, expected=1.0)],
    )
    def _double(x):
        return x * 2

    return "t.sink"


@pytest.fixture
def written(monkeypatch):
    """Capture what DbAuditSink hands to the action recorder."""
    calls = []

    def _record_action(**kw):
        calls.append(kw)

    monkeypatch.setattr(
        "src.services.agent_actions.record_action", _record_action
    )
    return calls


class TestInstalling:
    def test_the_durable_sink_is_installed(self, restore_sink):
        install_db_audit_sink()
        assert isinstance(get_audit_sink(), DbAuditSink)

    def test_it_reports_that_it_installed(self, restore_sink):
        assert install_db_audit_sink() is True

    def test_installing_twice_does_not_stack(self, restore_sink):
        install_db_audit_sink()
        first = get_audit_sink()

        assert install_db_audit_sink() is False, "second install should be a no-op"
        assert get_audit_sink() is first

    def test_it_replaces_the_in_memory_default(self, restore_sink):
        set_audit_sink(MemoryAuditSink())
        install_db_audit_sink()
        assert isinstance(get_audit_sink(), DbAuditSink)


class TestRecordsReachTheAuditTable:
    def test_an_evaluation_writes_one_action_row(
        self, restore_sink, a_formula, written
    ):
        install_db_audit_sink()
        evaluate(a_formula, {"x": 0.5})
        assert len(written) == 1

    def test_the_row_is_typed_as_a_formula_evaluation(
        self, restore_sink, a_formula, written
    ):
        install_db_audit_sink()
        evaluate(a_formula, {"x": 0.5})
        assert written[0]["action_type"] == "formula_eval"
        assert written[0]["field_name"] == a_formula
        assert written[0]["status"] == "ok"

    def test_the_full_record_rides_in_details(
        self, restore_sink, a_formula, written
    ):
        install_db_audit_sink()
        evaluate(a_formula, {"x": 0.5})
        details = written[0]["details"]
        assert details["inputs"] == {"x": 0.5}
        assert details["output"] == 1.0
        assert details["qualified_version"].startswith("1.0.0+")

    def test_a_refused_evaluation_is_audited_as_unassessed(
        self, restore_sink, a_formula, written
    ):
        """A refusal is the case you most need on the record, not the one to drop."""
        install_db_audit_sink()
        result = evaluate(a_formula, {"x": 99.0})

        assert result.unassessed, "premise: 99.0 is outside the declared range"
        assert len(written) == 1
        assert written[0]["status"] == "unassessed"


class TestAnAuditFailureNeverBreaksTheCalculation:
    """Defence in depth, and this test names the depth rather than one layer.

    Two guards stand between a dead database and a failed calculation:
    `DbAuditSink.write` catches, and `emit` catches again around the sink. This
    test was checked against both — removing either one on its own leaves it
    green, and it goes red (`RuntimeError: bp_agent_actions is unreachable`)
    only when both are broken. So it is a test of the *property*, not of either
    guard; a single-layer regression will not be caught here, which is the
    honest thing to record rather than to imply otherwise.
    """

    def test_the_evaluation_still_returns_its_value(
        self, restore_sink, a_formula, monkeypatch
    ):
        def _explode(**kw):
            raise RuntimeError("bp_agent_actions is unreachable")

        monkeypatch.setattr(
            "src.services.agent_actions.record_action", _explode
        )
        install_db_audit_sink()

        result = evaluate(a_formula, {"x": 0.5})

        assert not result.unassessed
        assert result.value == 1.0
