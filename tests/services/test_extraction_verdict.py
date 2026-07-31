# tests/services/test_extraction_verdict.py
"""What a human decided about a value, joined to the reader that produced it.

'dismiss' is the subtle one: dismissing a finding says the finding was wrong, which means
the extracted VALUE was right. Recording it as a negative would teach the system to distrust
exactly the readers people keep agreeing with.
"""
import pytest

from src.services.extraction_feedback.verdict import record_verdict, verdict_for


class _Cur:
    def __init__(self, provenance=None):
        self.rows, self._prov = [], provenance
        self.description = None
        self._result = []

    def execute(self, sql, params=()):
        if "FROM proc.bp_extraction_provenance" in sql:
            self._result = [self._prov] if self._prov else []
            return
        self.rows.append((sql, params))
        self._result = []

    def fetchone(self):
        return self._result[0] if self._result else None


def test_dismiss_is_a_vote_FOR_the_extracted_value():
    assert verdict_for("dismiss", resolved_value=None, extracted_value="GBP") == "rejected"


def test_replacing_the_value_is_the_strongest_negative():
    assert verdict_for("apply_value", resolved_value="CAD", extracted_value="USD") == "corrected"


def test_applying_the_same_value_back_is_a_confirmation_not_a_correction():
    # A human who retypes what was already there has agreed with it.
    assert verdict_for("apply_value", resolved_value="USD", extracted_value="USD") == "confirmed"


def test_clearing_a_value_is_a_correction():
    assert verdict_for("keep_null", resolved_value=None, extracted_value="USD") == "corrected"


def test_an_action_that_decides_nothing_has_no_verdict():
    assert verdict_for("flag", resolved_value=None, extracted_value="USD") is None
    assert verdict_for(None, resolved_value=None, extracted_value="USD") is None


def test_the_verdict_carries_the_producer_it_is_about():
    cur = _Cur(provenance=("regex", "dollar_symbol", 0.58))
    got = record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                         action="apply_value", resolved_value="CAD", extracted_value="USD",
                         resolved_by="ap@example.com")
    assert got == "corrected"
    sql, params = cur.rows[0]
    assert "INSERT INTO proc.bp_extraction_verdict" in sql
    assert "regex" in params and "dollar_symbol" in params and "corrected" in params


def test_a_value_with_no_provenance_is_still_recorded():
    # Documents extracted before Task 1 shipped have no provenance. The verdict is still
    # worth keeping — it just cannot be attributed to a reader.
    cur = _Cur(provenance=None)
    assert record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                          action="apply_value", resolved_value="CAD", extracted_value="USD",
                          resolved_by=None) == "corrected"
    _, params = cur.rows[0]
    assert None in params


def test_no_verdict_writes_no_row():
    cur = _Cur()
    assert record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                          action="flag", resolved_value=None, extracted_value="USD",
                          resolved_by=None) is None
    assert cur.rows == []
