"""BATNA assessment, salvaged from the deleted NegotiationStrategyEngine.

The engine derived leverage from two fields and then threw the distinction
away: `int(task.get("alternative_quotes", 0) or 0)` coerced "we never looked"
into "we looked and found none", and no-alternatives-no-history selected the
*most* aggressive posture it had. These tests pin the salvaged ladder and the
correction.
"""
import pytest

from src.services.formulas.unassessed import UNASSESSED, Confidence, UnassessedError
from src.services.negotiation.leverage import BatnaAssessment, assess_batna


class TestSalvagedLadder:
    """The four rungs the engine's _build_batna and select_strategy encoded."""

    def test_two_or_more_alternatives_is_strong(self):
        out = assess_batna(alternative_quotes=2, supplier_history_count=0)
        assert out.strength == "strong"
        assert out.score == 1.0

    def test_more_alternatives_do_not_exceed_the_ceiling(self):
        out = assess_batna(alternative_quotes=9, supplier_history_count=0)
        assert out.strength == "strong"
        assert out.score == 1.0

    def test_one_alternative_is_moderate(self):
        out = assess_batna(alternative_quotes=1, supplier_history_count=0)
        assert out.strength == "moderate"
        assert 0.0 < out.score < 1.0

    def test_no_alternatives_but_a_relationship_is_weak(self):
        out = assess_batna(alternative_quotes=0, supplier_history_count=7)
        assert out.strength == "weak"

    def test_no_alternatives_and_no_history_is_none_not_strong(self):
        """The engine's most dangerous rung.

        Zero alternatives and zero history selected STRATEGY_ANCHORING, whose
        target_discount was 0.15 -- the largest of the six. Having no
        alternative supplier is the *weakest* position at the table, and it
        must not read as licence to anchor hardest.
        """
        out = assess_batna(alternative_quotes=0, supplier_history_count=0)
        assert out.strength == "none"
        assert out.score == 0.0

    def test_strength_is_monotonic_in_alternatives(self):
        scores = [
            assess_batna(alternative_quotes=n, supplier_history_count=0).score
            for n in (0, 1, 2)
        ]
        assert scores == sorted(scores)


class TestFailsClosed:
    """Absent evidence is not evidence of absence."""

    def test_both_inputs_missing_is_unassessed(self):
        out = assess_batna(alternative_quotes=None, supplier_history_count=None)
        assert out.strength is UNASSESSED
        assert out.score is UNASSESSED
        assert out.confidence is Confidence.UNVERIFIED

    def test_unassessed_score_cannot_be_used_as_a_number(self):
        out = assess_batna(alternative_quotes=None, supplier_history_count=None)
        with pytest.raises(UnassessedError):
            _ = out.score * 2

    def test_unassessed_score_is_not_falsy(self):
        """`score or 0.0` must not silently become a real zero."""
        out = assess_batna(alternative_quotes=None, supplier_history_count=None)
        with pytest.raises(UnassessedError):
            bool(out.score)

    def test_missing_alternatives_alone_is_unassessed(self):
        """History alone cannot establish a BATNA.

        A BATNA is an *alternative*; how many orders we placed with this
        supplier says nothing about who else could supply.
        """
        out = assess_batna(alternative_quotes=None, supplier_history_count=7)
        assert out.strength is UNASSESSED

    def test_missing_history_alone_still_assesses(self):
        """Alternatives alone are enough: they are the BATNA."""
        out = assess_batna(alternative_quotes=3, supplier_history_count=None)
        assert out.strength == "strong"

    def test_a_finding_names_what_was_missing(self):
        out = assess_batna(alternative_quotes=None, supplier_history_count=None)
        assert out.findings
        assert any("alternative_quotes" in f for f in out.findings)

    def test_negative_counts_are_rejected_not_clamped(self):
        out = assess_batna(alternative_quotes=-1, supplier_history_count=0)
        assert out.strength is UNASSESSED

    def test_non_numeric_is_unassessed_not_zero(self):
        out = assess_batna(alternative_quotes="two", supplier_history_count=None)
        assert out.strength is UNASSESSED


class TestConfidence:
    def test_counts_from_a_system_of_record_are_observed(self):
        out = assess_batna(alternative_quotes=2, supplier_history_count=4,
                           source=Confidence.OBSERVED)
        assert out.confidence is Confidence.OBSERVED

    def test_default_source_is_unverified_not_observed(self):
        """Silence about provenance is not a claim of provenance."""
        out = assess_batna(alternative_quotes=2, supplier_history_count=4)
        assert out.confidence is Confidence.UNVERIFIED

    def test_a_count_read_out_of_a_message_is_asserted(self):
        out = assess_batna(alternative_quotes=2, supplier_history_count=0,
                           source=Confidence.ASSERTED)
        assert out.confidence is Confidence.ASSERTED


class TestNarrative:
    """The engine's BATNA prose was its only user-visible output; keep the substance."""

    def test_strong_batna_says_alternatives_are_ready(self):
        out = assess_batna(alternative_quotes=3, supplier_history_count=0)
        assert "3" in out.narrative
        assert out.reasons

    def test_narrative_never_states_a_count_it_does_not_have(self):
        out = assess_batna(alternative_quotes=None, supplier_history_count=None)
        assert "0" not in out.narrative
        assert "no alternative" not in out.narrative.lower()

    def test_assessment_is_frozen(self):
        out = assess_batna(alternative_quotes=2, supplier_history_count=0)
        assert isinstance(out, BatnaAssessment)
        with pytest.raises(Exception):
            out.strength = "weak"  # type: ignore[misc]
