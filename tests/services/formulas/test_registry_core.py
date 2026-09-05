"""The registry's own guarantees.

Several of these tests exist to prove a guard actually fails. A guard that has
only ever been observed passing is not evidence of anything --- so where this
module claims something is rejected, it breaks it on purpose and watches it go
red.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timezone

import pytest

from src.services.formulas import (
    UNASSESSED,
    Confidence,
    FormulaError,
    GoldenVector,
    GoldenVectorFailure,
    MemoryAuditSink,
    Provenance,
    Term,
    UnassessedError,
    evaluate,
    evaluate_many,
    formula,
    set_audit_sink,
)
from src.services.formulas.contract import COUNT, GBP, LABEL, RATIO, Output
from src.services.formulas.registry import REGISTRY, FormulaKind, compute_version_hash

_FROM = date(2026, 9, 5)


@pytest.fixture
def sink():
    s = MemoryAuditSink()
    previous = set_audit_sink(s)
    yield s
    set_audit_sink(previous)


@pytest.fixture
def clean_registry():
    """Register throwaway formulas without polluting the real registry."""
    before = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(before)


def _define(name, **over):
    kwargs = dict(
        version="1.0.0", owner="tests", purpose="fixture",
        effective_from=_FROM,
        inputs=[Term("x", RATIO, minimum=0.0, maximum=1.0)],
        output=Output("float", RATIO, "x doubled"),
        golden=[GoldenVector(inputs={"x": 0.5}, expected=1.0)],
    )
    kwargs.update(over)

    @formula(name, **kwargs)
    def _fn(x):
        return x * 2

    return _fn


# --------------------------------------------------------------- UNASSESSED


class TestUnassessed:
    def test_is_not_falsy_it_refuses_to_be_tested(self):
        with pytest.raises(UnassessedError):
            bool(UNASSESSED)
        with pytest.raises(UnassessedError):
            if UNASSESSED:  # noqa: SIM103 - the point of the test
                pass

    def test_arithmetic_raises_rather_than_coercing(self):
        for op in (lambda: UNASSESSED + 1, lambda: 1 + UNASSESSED,
                   lambda: UNASSESSED * 2, lambda: float(UNASSESSED),
                   lambda: UNASSESSED > 0):
            with pytest.raises(UnassessedError):
                op()

    def test_the_common_default_idiom_cannot_silently_zero_it(self):
        # `result or 0` is exactly how "unknown" becomes "zero" in the wild.
        with pytest.raises(UnassessedError):
            _ = UNASSESSED or 0

    def test_identity_comparison_is_the_supported_test(self):
        assert UNASSESSED is UNASSESSED
        assert (UNASSESSED == UNASSESSED) is True
        assert (UNASSESSED == 0) is False
        assert (UNASSESSED == None) is False  # noqa: E711 - explicit


class TestConfidence:
    def test_ladder_is_ordered(self):
        assert Confidence.OBSERVED > Confidence.ASSERTED > Confidence.UNVERIFIED

    def test_weakest_of_nothing_is_unverified_not_observed(self):
        from src.services.formulas import weakest

        assert weakest([]) is Confidence.UNVERIFIED

    def test_weakest_takes_the_floor(self):
        from src.services.formulas import weakest

        assert weakest([Confidence.OBSERVED, Confidence.ASSERTED]) is Confidence.ASSERTED


# --------------------------------------------------------------- registration


class TestRegistration:
    def test_a_formula_without_golden_vectors_is_refused(self, clean_registry):
        with pytest.raises(FormulaError, match="golden vector"):
            _define("t.no_vectors", golden=[])

    def test_a_wrong_golden_vector_fails_the_import(self, clean_registry):
        """The gate, broken on purpose."""
        with pytest.raises(GoldenVectorFailure, match="does not reproduce"):

            @formula(
                "t.wrong", version="1.0.0", owner="tests", purpose="fixture",
                effective_from=_FROM,
                inputs=[Term("x", RATIO)],
                output=Output("float", RATIO, ""),
                golden=[GoldenVector(inputs={"x": 0.5}, expected=999.0)],
            )
            def _fn(x):
                return x * 2

        assert "t.wrong" not in REGISTRY

    def test_a_raising_golden_vector_fails_the_import(self, clean_registry):
        with pytest.raises(GoldenVectorFailure, match="ZeroDivisionError"):

            @formula(
                "t.boom", version="1.0.0", owner="tests", purpose="fixture",
                effective_from=_FROM,
                inputs=[Term("x", RATIO)],
                output=Output("float", RATIO, ""),
                golden=[GoldenVector(inputs={"x": 0.0}, expected=0.0)],
            )
            def _fn(x):
                return 1 / x

    def test_duplicate_names_are_refused(self, clean_registry):
        _define("t.dupe")
        with pytest.raises(FormulaError, match="already registered"):
            _define("t.dupe")

    def test_duplicate_terms_are_refused(self, clean_registry):
        with pytest.raises(FormulaError, match="duplicate input term"):
            _define("t.dup_term",
                    inputs=[Term("x", RATIO), Term("x", COUNT)])

    def test_version_hash_changes_when_the_maths_changes(self):
        from src.services.formulas.contract import Contract

        contract = Contract(inputs=(Term("x", RATIO),), output=Output("float", RATIO, ""))
        vectors = (GoldenVector(inputs={"x": 1.0}, expected=2.0),)

        def a(x):
            return x * 2

        def b(x):
            return x * 2.0000001

        assert compute_version_hash(a, contract, vectors) != compute_version_hash(
            b, contract, vectors
        )

    def test_version_hash_changes_when_the_contract_changes(self):
        from src.services.formulas.contract import Contract

        def a(x):
            return x * 2

        vectors = (GoldenVector(inputs={"x": 1.0}, expected=2.0),)
        loose = Contract(inputs=(Term("x", RATIO),), output=Output("float", RATIO, ""))
        tight = Contract(
            inputs=(Term("x", RATIO, minimum=0.0, maximum=1.0),),
            output=Output("float", RATIO, ""),
        )
        assert compute_version_hash(a, loose, vectors) != compute_version_hash(
            a, tight, vectors
        )

    def test_version_hash_changes_when_a_golden_vector_changes(self):
        from src.services.formulas.contract import Contract

        def a(x):
            return x * 2

        contract = Contract(inputs=(Term("x", RATIO),), output=Output("float", RATIO, ""))
        assert compute_version_hash(
            a, contract, (GoldenVector(inputs={"x": 1.0}, expected=2.0),)
        ) != compute_version_hash(
            a, contract, (GoldenVector(inputs={"x": 2.0}, expected=4.0),)
        )


# --------------------------------------------------------------- validation


class TestContractValidation:
    def test_missing_required_input_is_unassessed_not_zero(self, clean_registry, sink):
        _define("t.required")
        r = evaluate("t.required", {})
        assert r.value is UNASSESSED
        assert r.unassessed
        assert r.findings[0].code == "missing_required_input"
        assert sink.records[-1].status == "unassessed"

    def test_out_of_range_is_unassessed(self, clean_registry, sink):
        _define("t.range")
        r = evaluate("t.range", {"x": 42.0})
        assert r.value is UNASSESSED
        assert r.findings[0].code == "out_of_range"
        assert "above the declared maximum" in r.findings[0].detail

    def test_the_zero_to_one_hundred_scale_collision_is_caught(self, clean_registry):
        """A 0-100 risk score handed to a 0-1 contract is refused, not scaled.

        This is the live confusion the contract exists to catch: risk_score is
        consumed on both scales in this codebase and nothing could complain.
        """
        _define("t.risk", inputs=[Term("x", RATIO, minimum=0.0, maximum=1.0)])
        assert evaluate("t.risk", {"x": 0.75}).value == 1.5
        assert evaluate("t.risk", {"x": 75.0}).value is UNASSESSED

    def test_unit_mismatch_is_refused(self, clean_registry):
        from src.services.formulas import Quantity

        _define("t.unit", inputs=[Term("x", GBP)])
        r = evaluate("t.unit", {"x": Quantity(10.0, COUNT)})
        assert r.value is UNASSESSED
        assert r.findings[0].code == "unit_mismatch"

    def test_a_quantity_in_the_right_unit_is_unwrapped(self, clean_registry):
        from src.services.formulas import Quantity

        _define("t.unit_ok", inputs=[Term("x", GBP)])
        assert evaluate("t.unit_ok", {"x": Quantity(10.0, GBP)}).value == 20.0

    def test_an_undeclared_key_is_refused_rather_than_dropped(self, clean_registry):
        _define("t.extra")
        r = evaluate("t.extra", {"x": 0.5, "risk": 1.0})
        assert r.value is UNASSESSED
        assert any(f.code == "undeclared_input" for f in r.findings)

    def test_nan_is_refused_against_a_declared_range(self, clean_registry):
        _define("t.nan")
        r = evaluate("t.nan", {"x": float("nan")})
        assert r.value is UNASSESSED
        assert r.findings[0].code == "not_a_number"

    def test_an_optional_term_still_evaluates(self, clean_registry):
        @formula(
            "t.optional", version="1.0.0", owner="tests", purpose="fixture",
            effective_from=_FROM,
            inputs=[Term("x", RATIO, required=False)],
            output=Output("str", LABEL, ""),
            golden=[GoldenVector(inputs={"x": None}, expected="absent")],
        )
        def _fn(x=None):
            return "absent" if x is None else "present"

        assert evaluate("t.optional", {}).value == "absent"

    def test_a_body_that_raises_becomes_unassessed_not_a_crash(self, clean_registry):
        @formula(
            "t.raiser", version="1.0.0", owner="tests", purpose="fixture",
            effective_from=_FROM,
            inputs=[Term("x", RATIO, required=False)],
            output=Output("float", RATIO, ""),
            golden=[GoldenVector(inputs={"x": 1.0}, expected=1.0)],
        )
        def _fn(x=None):
            if x is None:
                raise ValueError("no x")
            return x

        r = evaluate("t.raiser", {})
        assert r.value is UNASSESSED
        assert r.findings[-1].code == "evaluation_error"


class TestResult:
    def test_unwrap_raises_on_unassessed(self, clean_registry):
        from src.services.formulas import EvaluationRefused

        _define("t.unwrap")
        with pytest.raises(EvaluationRefused):
            evaluate("t.unwrap", {}).unwrap()

    def test_or_else_makes_the_fallback_visible(self, clean_registry):
        _define("t.orelse")
        assert evaluate("t.orelse", {}).or_else(0.0) == 0.0
        assert evaluate("t.orelse", {"x": 0.5}).or_else(0.0) == 1.0


# --------------------------------------------------------------- audit


class TestAuditRecord:
    def test_every_evaluation_writes_one_record(self, clean_registry, sink):
        _define("t.audit")
        evaluate("t.audit", {"x": 0.5})
        assert len(sink.records) == 1
        rec = sink.records[0]
        assert rec.formula == "t.audit"
        assert rec.version == "1.0.0"
        assert rec.version_hash == REGISTRY["t.audit"].version_hash
        assert rec.inputs == {"x": 0.5}
        assert rec.output == 1.0
        assert rec.status == "ok"
        assert isinstance(rec.evaluated_at, datetime)
        assert rec.to_dict()["qualified_version"].startswith("1.0.0+")

    def test_provenance_ids_and_confidence_are_carried(self, clean_registry, sink):
        _define("t.prov")
        r = evaluate(
            "t.prov", {"x": 0.5},
            provenance={"x": Provenance("doc-1#unit_price", Confidence.ASSERTED, "llm")},
        )
        assert r.confidence is Confidence.ASSERTED
        assert sink.records[0].provenance_ids == ["doc-1#unit_price"]

    def test_result_confidence_never_exceeds_its_weakest_input(self, clean_registry):
        _define("t.cap", inputs=[Term("x", RATIO, minimum=0.0, maximum=1.0)])
        r = evaluate(
            "t.cap", {"x": 0.5},
            provenance={
                "a": Provenance("a", Confidence.OBSERVED),
                "b": Provenance("b", Confidence.ASSERTED),
            },
        )
        assert r.confidence is Confidence.ASSERTED

    def test_llm_sourced_inputs_cap_the_result_at_asserted(self, clean_registry):
        _define("t.llm")
        r = evaluate(
            "t.llm", {"x": 0.5},
            provenance={"x": Provenance("judge-1", Confidence.ASSERTED, "l3_judge")},
        )
        assert r.confidence is not Confidence.OBSERVED
        assert r.confidence is Confidence.ASSERTED

    def test_unstated_provenance_is_unverified_not_observed(self, clean_registry):
        _define("t.silent")
        assert evaluate("t.silent", {"x": 0.5}).confidence is Confidence.UNVERIFIED

    def test_audit_can_be_suppressed_for_a_hot_loop(self, clean_registry, sink):
        _define("t.quiet")
        evaluate("t.quiet", {"x": 0.5}, audit=False)
        assert sink.records == []


# --------------------------------------------------------------- batch


class TestEvaluateMany:
    def test_scalar_batch_writes_exactly_one_record(self, clean_registry, sink):
        _define("t.batch")
        batch = evaluate_many("t.batch", [{"x": 0.1}, {"x": 0.2}, {"x": 0.3}])
        assert len(sink.records) == 1
        assert sink.records[0].batch_size == 3
        assert batch.values == pytest.approx([0.2, 0.4, 0.6])

    def test_one_bad_context_does_not_poison_a_scalar_batch(self, clean_registry):
        _define("t.batch_mixed")
        batch = evaluate_many("t.batch_mixed", [{"x": 0.1}, {"x": 99.0}, {"x": 0.3}])
        assert batch.values[0] == pytest.approx(0.2)
        assert batch.values[1] is UNASSESSED
        assert batch.values[2] == pytest.approx(0.6)
        assert batch.any_unassessed
        assert len(batch.assessed) == 2

    def test_a_set_formula_refuses_scalar_evaluation(self, clean_registry):
        @formula(
            "t.set", version="1.0.0", kind=FormulaKind.SET, owner="tests",
            purpose="fixture", effective_from=_FROM,
            inputs=[Term("v", RATIO, minimum=0.0, maximum=1000.0)],
            output=Output("list[float]", RATIO, ""),
            golden=[GoldenVector(inputs=[{"v": 1.0}, {"v": 3.0}], expected=[0.25, 0.75])],
        )
        def _fn(rows):
            total = sum(r["v"] for r in rows)
            return [r["v"] / total for r in rows]

        r = evaluate("t.set", {"v": 1.0})
        assert r.value is UNASSESSED
        assert "SET formula" in r.findings[-1].detail

    def test_a_set_formula_sees_the_whole_population(self, clean_registry, sink):
        @formula(
            "t.share", version="1.0.0", kind=FormulaKind.SET, owner="tests",
            purpose="fixture", effective_from=_FROM,
            inputs=[Term("v", RATIO, minimum=0.0, maximum=1000.0)],
            output=Output("list[float]", RATIO, ""),
            golden=[GoldenVector(inputs=[{"v": 1.0}, {"v": 3.0}], expected=[0.25, 0.75])],
        )
        def _fn(rows):
            total = sum(r["v"] for r in rows)
            return [r["v"] / total for r in rows]

        batch = evaluate_many("t.share", [{"v": 2.0}, {"v": 2.0}, {"v": 4.0}])
        assert batch.values == pytest.approx([0.25, 0.25, 0.5])
        assert len(sink.records) == 1
        assert sink.records[0].batch_size == 3

    def test_one_bad_context_refuses_the_whole_set(self, clean_registry):
        """Fail-closed: a population statistic from a partly-invalid population
        is not a smaller truth, it is a wrong one."""

        @formula(
            "t.set_strict", version="1.0.0", kind=FormulaKind.SET, owner="tests",
            purpose="fixture", effective_from=_FROM,
            inputs=[Term("v", RATIO, minimum=0.0, maximum=10.0)],
            output=Output("list[float]", RATIO, ""),
            golden=[GoldenVector(inputs=[{"v": 1.0}, {"v": 3.0}], expected=[0.25, 0.75])],
        )
        def _fn(rows):
            total = sum(r["v"] for r in rows)
            return [r["v"] / total for r in rows]

        batch = evaluate_many("t.set_strict", [{"v": 1.0}, {"v": 9999.0}])
        assert all(v is UNASSESSED for v in batch.values)

    def test_shared_parameters_are_validated_once(self, clean_registry, sink):
        @formula(
            "t.shared", version="1.0.0", kind=FormulaKind.SET, owner="tests",
            purpose="fixture", effective_from=_FROM,
            inputs=[Term("v", RATIO, minimum=0.0, maximum=100.0),
                    Term("scale", RATIO, minimum=0.0, maximum=10.0, shared=True)],
            output=Output("list[float]", RATIO, ""),
            golden=[GoldenVector(inputs=[{"v": 1.0}], shared={"scale": 2.0},
                                 expected=[2.0])],
        )
        def _fn(rows, scale):
            return [r["v"] * scale for r in rows]

        ok = evaluate_many("t.shared", [{"v": 1.0}, {"v": 2.0}], shared={"scale": 3.0})
        assert ok.values == pytest.approx([3.0, 6.0])
        assert sink.records[-1].inputs["shared"] == {"scale": 3.0}

        bad = evaluate_many("t.shared", [{"v": 1.0}], shared={"scale": 999.0})
        assert bad.values[0] is UNASSESSED
        assert any("shared:" in f.detail for f in bad[0].findings)

    def test_a_length_changing_set_formula_is_a_hard_error(self, clean_registry):
        @formula(
            "t.set_bad_len", version="1.0.0", kind=FormulaKind.SET, owner="tests",
            purpose="fixture", effective_from=_FROM,
            inputs=[Term("v", RATIO, minimum=0.0, maximum=10.0)],
            output=Output("list[float]", RATIO, ""),
            golden=[GoldenVector(inputs=[{"v": 1.0}], expected=[1.0])],
        )
        def _fn(rows):
            return [r["v"] for r in rows][:1]

        with pytest.raises(RuntimeError, match="length-preserving"):
            evaluate_many("t.set_bad_len", [{"v": 1.0}, {"v": 2.0}])
