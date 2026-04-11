"""Tests for services.pattern_service using the in-memory storage backend."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import pytest
from services.pattern_service import PatternService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def svc():
    """Return a fresh PatternService backed by in-memory storage."""
    return PatternService(storage="memory")


# ---------------------------------------------------------------------------
# record_pattern — basic insert
# ---------------------------------------------------------------------------

class TestRecordPattern:
    def test_returns_dict_with_expected_fields(self, svc):
        row = svc.record_pattern(
            pattern_type="negotiation",
            pattern_text="Cooperative strategy yields 10% discount with repeat suppliers",
            category="procurement",
            confidence=0.8,
        )
        assert row["pattern_type"] == "negotiation"
        assert row["pattern_text"] == "Cooperative strategy yields 10% discount with repeat suppliers"
        assert row["category"] == "procurement"
        assert row["confidence"] == 0.8
        assert row["source_count"] == 1
        assert row["deprecated"] is False
        assert "id" in row
        assert "last_validated" in row

    def test_confidence_clamped_to_1(self, svc):
        row = svc.record_pattern("t", "text", confidence=1.5)
        assert row["confidence"] == 1.0

    def test_confidence_clamped_to_0(self, svc):
        row = svc.record_pattern("t", "text", confidence=-0.5)
        assert row["confidence"] == 0.0

    def test_empty_pattern_type_raises(self, svc):
        with pytest.raises(ValueError):
            svc.record_pattern("", "some text")

    def test_empty_pattern_text_raises(self, svc):
        with pytest.raises(ValueError):
            svc.record_pattern("negotiation", "")


# ---------------------------------------------------------------------------
# record_pattern — UPSERT behaviour
# ---------------------------------------------------------------------------

class TestRecordPatternUpsert:
    def test_second_insert_increments_source_count(self, svc):
        svc.record_pattern("negotiation", "Bulk order discount pattern", confidence=0.6)
        row2 = svc.record_pattern("negotiation", "Bulk order discount pattern", confidence=0.6)
        assert row2["source_count"] == 2

    def test_second_insert_increases_confidence(self, svc):
        svc.record_pattern("negotiation", "Bulk order discount pattern", confidence=0.6)
        row2 = svc.record_pattern("negotiation", "Bulk order discount pattern", confidence=0.6)
        assert row2["confidence"] > 0.6

    def test_upsert_confidence_does_not_exceed_1(self, svc):
        svc.record_pattern("t", "txt", confidence=0.99)
        row2 = svc.record_pattern("t", "txt", confidence=0.99)
        assert row2["confidence"] <= 1.0

    def test_different_text_creates_new_row(self, svc):
        svc.record_pattern("negotiation", "Pattern A", confidence=0.5)
        svc.record_pattern("negotiation", "Pattern B", confidence=0.5)
        patterns = svc.get_patterns()
        texts = {p["pattern_text"] for p in patterns}
        assert "Pattern A" in texts
        assert "Pattern B" in texts


# ---------------------------------------------------------------------------
# get_patterns — filtering
# ---------------------------------------------------------------------------

class TestGetPatterns:
    def test_returns_all_active_patterns(self, svc):
        svc.record_pattern("negotiation", "Pattern A", category="goods")
        svc.record_pattern("pricing", "Pattern B", category="services")
        results = svc.get_patterns()
        assert len(results) == 2

    def test_filter_by_pattern_type(self, svc):
        svc.record_pattern("negotiation", "Pattern A", category="goods")
        svc.record_pattern("pricing", "Pattern B", category="services")
        results = svc.get_patterns(pattern_type="negotiation")
        assert len(results) == 1
        assert results[0]["pattern_type"] == "negotiation"

    def test_filter_by_category(self, svc):
        svc.record_pattern("negotiation", "Pattern A", category="goods")
        svc.record_pattern("pricing", "Pattern B", category="services")
        results = svc.get_patterns(category="goods")
        assert len(results) == 1
        assert results[0]["category"] == "goods"

    def test_filter_by_min_confidence(self, svc):
        svc.record_pattern("negotiation", "Low confidence pattern", confidence=0.3)
        svc.record_pattern("negotiation", "High confidence pattern", confidence=0.9)
        results = svc.get_patterns(min_confidence=0.7)
        assert len(results) == 1
        assert results[0]["pattern_text"] == "High confidence pattern"

    def test_results_sorted_by_confidence_desc(self, svc):
        svc.record_pattern("t", "Low", confidence=0.3)
        svc.record_pattern("t", "High", confidence=0.9)
        svc.record_pattern("t", "Mid", confidence=0.6)
        results = svc.get_patterns()
        confidences = [r["confidence"] for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_deprecated_patterns_excluded(self, svc):
        svc.record_pattern("negotiation", "Old pattern", confidence=0.8)
        svc.deprecate_pattern("negotiation", "Old pattern")
        results = svc.get_patterns()
        assert len(results) == 0

    def test_combined_filters(self, svc):
        svc.record_pattern("negotiation", "Pattern A", category="goods", confidence=0.8)
        svc.record_pattern("negotiation", "Pattern B", category="services", confidence=0.9)
        svc.record_pattern("pricing", "Pattern C", category="goods", confidence=0.7)
        results = svc.get_patterns(pattern_type="negotiation", category="goods", min_confidence=0.5)
        assert len(results) == 1
        assert results[0]["pattern_text"] == "Pattern A"


# ---------------------------------------------------------------------------
# reinforce_pattern
# ---------------------------------------------------------------------------

class TestReinforcePattern:
    def test_reinforce_increases_confidence(self, svc):
        svc.record_pattern("negotiation", "Some pattern", confidence=0.5)
        updated = svc.reinforce_pattern("negotiation", "Some pattern", delta=0.1)
        assert updated is not None
        assert updated["confidence"] >= 0.6

    def test_reinforce_increments_source_count(self, svc):
        svc.record_pattern("negotiation", "Some pattern", confidence=0.5)
        updated = svc.reinforce_pattern("negotiation", "Some pattern", delta=0.1)
        assert updated["source_count"] == 2

    def test_reinforce_clamps_at_1(self, svc):
        svc.record_pattern("t", "txt", confidence=0.99)
        updated = svc.reinforce_pattern("t", "txt", delta=0.5)
        assert updated["confidence"] <= 1.0

    def test_reinforce_returns_none_for_missing(self, svc):
        result = svc.reinforce_pattern("negotiation", "Nonexistent pattern")
        assert result is None

    def test_reinforce_negative_delta_reduces_confidence(self, svc):
        svc.record_pattern("negotiation", "Some pattern", confidence=0.8)
        updated = svc.reinforce_pattern("negotiation", "Some pattern", delta=-0.2)
        assert updated["confidence"] < 0.8

    def test_reinforce_clamps_at_0(self, svc):
        svc.record_pattern("t", "txt", confidence=0.05)
        updated = svc.reinforce_pattern("t", "txt", delta=-0.5)
        assert updated["confidence"] >= 0.0


# ---------------------------------------------------------------------------
# deprecate_pattern
# ---------------------------------------------------------------------------

class TestDeprecatePattern:
    def test_deprecate_returns_true_when_found(self, svc):
        svc.record_pattern("negotiation", "Pattern to retire")
        result = svc.deprecate_pattern("negotiation", "Pattern to retire")
        assert result is True

    def test_deprecate_returns_false_when_not_found(self, svc):
        result = svc.deprecate_pattern("negotiation", "Nonexistent pattern")
        assert result is False

    def test_deprecated_pattern_hidden_from_get(self, svc):
        svc.record_pattern("negotiation", "Pattern to retire", confidence=0.9)
        svc.deprecate_pattern("negotiation", "Pattern to retire")
        results = svc.get_patterns()
        assert not any(p["pattern_text"] == "Pattern to retire" for p in results)


# ---------------------------------------------------------------------------
# learn_from_outcome
# ---------------------------------------------------------------------------

class TestLearnFromOutcome:
    def test_positive_outcome_records_pattern(self, svc):
        outcome = {
            "strategy": "Cooperative",
            "expected": {"discount": 0.1},
            "actual": {"discount": 0.12},
            "category": "goods",
        }
        recorded = svc.learn_from_outcome(outcome)
        assert len(recorded) == 1
        assert recorded[0]["pattern_type"] == "outcome_positive"

    def test_negative_outcome_records_caution_pattern(self, svc):
        outcome = {
            "strategy": "Aggressive",
            "expected": {"discount": 0.15},
            "actual": {"discount": 0.05},
            "category": "services",
        }
        recorded = svc.learn_from_outcome(outcome)
        assert len(recorded) == 1
        assert recorded[0]["pattern_type"] == "outcome_caution"

    def test_multiple_metrics_records_multiple_patterns(self, svc):
        outcome = {
            "strategy": "Collaborative",
            "expected": {"discount": 0.1, "lead_time_reduction": 5.0},
            "actual": {"discount": 0.12, "lead_time_reduction": 4.0},
            "category": "logistics",
        }
        recorded = svc.learn_from_outcome(outcome)
        assert len(recorded) == 2

    def test_missing_actual_metric_skipped(self, svc):
        outcome = {
            "strategy": "Standard",
            "expected": {"discount": 0.1, "volume_bonus": 0.05},
            "actual": {"discount": 0.12},
        }
        recorded = svc.learn_from_outcome(outcome)
        # Only 'discount' has actual value
        assert len(recorded) == 1

    def test_repeated_outcomes_upsert_same_pattern(self, svc):
        outcome = {
            "strategy": "Cooperative",
            "expected": {"discount": 0.1},
            "actual": {"discount": 0.12},
            "category": "goods",
        }
        svc.learn_from_outcome(outcome)
        svc.learn_from_outcome(outcome)
        results = svc.get_patterns(pattern_type="outcome_positive")
        assert len(results) == 1
        assert results[0]["source_count"] == 2

    def test_outcome_confidence_higher_for_over_performance(self, svc):
        outcome_good = {
            "strategy": "Strategy A",
            "expected": {"savings": 1000.0},
            "actual": {"savings": 1500.0},
        }
        outcome_bad = {
            "strategy": "Strategy B",
            "expected": {"savings": 1000.0},
            "actual": {"savings": 500.0},
        }
        good_rows = svc.learn_from_outcome(outcome_good)
        bad_rows = svc.learn_from_outcome(outcome_bad)
        assert good_rows[0]["confidence"] > bad_rows[0]["confidence"]

    def test_zero_expected_value_skipped(self, svc):
        outcome = {
            "strategy": "Test",
            "expected": {"metric": 0.0},
            "actual": {"metric": 5.0},
        }
        recorded = svc.learn_from_outcome(outcome)
        assert len(recorded) == 0

    def test_returns_empty_list_for_empty_outcome(self, svc):
        recorded = svc.learn_from_outcome({})
        assert recorded == []

    def test_category_propagated_to_pattern(self, svc):
        outcome = {
            "strategy": "Cooperative",
            "expected": {"discount": 0.1},
            "actual": {"discount": 0.15},
            "category": "electronics",
        }
        recorded = svc.learn_from_outcome(outcome)
        assert recorded[0]["category"] == "electronics"
