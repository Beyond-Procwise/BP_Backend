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


def test_the_dispatch_snapshot_supplies_the_reader_when_the_table_has_no_row_yet():
    """This is the ORDINARY case in production, not a fallback for odd documents.

    proc.bp_extraction_provenance is written inside promotion.promote(). A document that
    raised a blocking discrepancy never reached promote() — that is precisely why a human
    is looking at it — so at the moment the verdict is recorded the table has nothing to
    say about it. The `_field_provenance` map dispatch froze into _raw.parser_snapshot at
    extraction time carries the same three values and is always there. Without it every
    real verdict would name a NULL reader, and a NULL reader matches nothing in
    apply_observed or _compute_accuracy_score — the loop would never close.
    """
    cur = _Cur(provenance=None)
    got = record_verdict(
        cur, doc_type="invoice", doc_pk="INV-500", field_name="currency",
        action="apply_value", resolved_value="CAD", extracted_value="USD",
        resolved_by="ap@example.com",
        snapshot={"currency": {"source": "regex", "pattern_name": "dollar_symbol",
                               "confidence": 0.58}},
    )
    assert got == "corrected"
    _, params = cur.rows[0]
    assert "regex" in params and "dollar_symbol" in params and 0.58 in params


def test_the_provenance_table_still_wins_when_it_does_have_a_row():
    # A re-promoted document may have been read by a different reader than the snapshot
    # remembers; the table is the later, authoritative record.
    cur = _Cur(provenance=("ner", None, 0.4))
    record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                   action="apply_value", resolved_value="CAD", extracted_value="USD",
                   resolved_by="ap@example.com",
                   snapshot={"currency": {"source": "regex",
                                          "pattern_name": "dollar_symbol",
                                          "confidence": 0.58}})
    _, params = cur.rows[0]
    assert "ner" in params and "dollar_symbol" not in params


def test_a_value_nothing_can_attribute_is_still_recorded():
    # Neither the table nor the snapshot knows this field. The human's judgement is still
    # worth keeping — it just cannot be charged to any reader.
    cur = _Cur(provenance=None)
    assert record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                          action="apply_value", resolved_value="CAD", extracted_value="USD",
                          resolved_by=None, snapshot={"invoice_amount": {}}) == "corrected"
    _, params = cur.rows[0]
    assert None in params


def test_a_bulk_machine_dismissal_is_not_counted_as_a_human_agreeing():
    """'dismiss' means a person looked and decided nothing was wrong — which
    accuracy._AGREES scores as the reader having been RIGHT. Two processes already write
    resolved/dismiss rows in bulk (dedup-migration: 41 live rows, 20 of them blocking;
    session_postprocess: 5). Eight of those crosses MIN_SAMPLE on its own and would
    manufacture a perfect score for a reader no human ever endorsed — or bury a demotion
    that real corrections had earned."""
    for principal in ("dedup-migration", "session_postprocess", "SESSION_POSTPROCESS"):
        cur = _Cur(provenance=("regex", "dollar_symbol", 0.58))
        assert record_verdict(cur, doc_type="invoice", doc_pk="INV-1",
                              field_name="currency", action="dismiss",
                              resolved_value=None, extracted_value="USD",
                              resolved_by=principal) is None
        assert cur.rows == [], f"{principal} wrote a verdict row"


def test_a_person_dismissing_a_finding_still_counts():
    # The guard must key on WHO, not on the action — a human dismissal is real evidence.
    cur = _Cur(provenance=("regex", "dollar_symbol", 0.58))
    assert record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                          action="dismiss", resolved_value=None, extracted_value="USD",
                          resolved_by="ap@example.com") == "rejected"
    assert len(cur.rows) == 1


def test_the_latest_claim_is_found_by_id_not_by_attempt():
    """`attempt` is numbered per raw_id, not per document: a re-extraction gets a NEW
    raw_id and starts again at attempt=1, so ORDER BY attempt DESC would hand back the
    OLDER raw_id's attempt=2 row in preference to the newer claim. id is a monotonic
    BIGSERIAL over the whole table and is the only correct 'most recent'."""
    from src.services.extraction_feedback.verdict import _PROVENANCE_SQL
    assert "ORDER BY id DESC" in _PROVENANCE_SQL
    assert "attempt" not in _PROVENANCE_SQL


def test_no_verdict_writes_no_row():
    cur = _Cur()
    assert record_verdict(cur, doc_type="invoice", doc_pk="INV-1", field_name="currency",
                          action="flag", resolved_value=None, extracted_value="USD",
                          resolved_by=None) is None
    assert cur.rows == []
