"""How often each reader has actually been right.

The number that matters is not "how confident was the pattern" — that was hand-written in a
YAML file — but "how often did a human let this reader's answer stand". Two rules protect it
from being noise: nothing is scored until there is enough evidence to mean anything, and a
key with too little evidence is ABSENT rather than zero, so callers fall back to the static
prior instead of treating silence as failure.
"""
from src.services.extraction_feedback.accuracy import (
    MIN_SAMPLE, observed_accuracy,
)


def _v(verdict, field="currency", pattern="dollar_symbol", doc_type="invoice"):
    return {"doc_type": doc_type, "field_name": field, "pattern_name": pattern,
            "source": "regex", "verdict": verdict}


def test_a_reader_humans_keep_agreeing_with_scores_high():
    rows = [_v("confirmed")] * 9 + [_v("rejected")]
    acc = observed_accuracy(rows)
    assert acc[("invoice", "currency", "dollar_symbol")] == 1.0


def test_a_reader_humans_keep_correcting_scores_low():
    rows = [_v("corrected")] * 8 + [_v("confirmed")] * 2
    assert observed_accuracy(rows)[("invoice", "currency", "dollar_symbol")] == 0.2


def test_rejected_counts_AS_agreement():
    # Dismissing a finding says the value was fine. Counting it against the reader would
    # invert the whole signal.
    rows = [_v("rejected")] * MIN_SAMPLE
    assert observed_accuracy(rows)[("invoice", "currency", "dollar_symbol")] == 1.0


def test_too_little_evidence_is_absent_not_zero():
    rows = [_v("corrected")] * (MIN_SAMPLE - 1)
    assert ("invoice", "currency", "dollar_symbol") not in observed_accuracy(rows)


def test_the_sample_floor_is_adjustable_for_a_caller_that_wants_to_be_stricter():
    rows = [_v("confirmed")] * 10
    assert ("invoice", "currency", "dollar_symbol") not in observed_accuracy(rows, min_sample=20)


def test_readers_are_scored_separately_not_pooled():
    rows = ([_v("confirmed", pattern="anchored_currency_iso")] * MIN_SAMPLE
            + [_v("corrected", pattern="dollar_symbol")] * MIN_SAMPLE)
    acc = observed_accuracy(rows)
    assert acc[("invoice", "currency", "anchored_currency_iso")] == 1.0
    assert acc[("invoice", "currency", "dollar_symbol")] == 0.0


def test_the_ai_layer_is_scored_under_its_source_since_it_has_no_pattern():
    rows = [{"doc_type": "invoice", "field_name": "currency", "pattern_name": None,
             "source": "context_layer", "verdict": "confirmed"}] * MIN_SAMPLE
    assert observed_accuracy(rows)[("invoice", "currency", "context_layer")] == 1.0


def test_doc_types_are_scored_separately():
    rows = ([_v("confirmed")] * MIN_SAMPLE
            + [_v("corrected", doc_type="quote")] * MIN_SAMPLE)
    acc = observed_accuracy(rows)
    assert acc[("invoice", "currency", "dollar_symbol")] == 1.0
    assert acc[("quote", "currency", "dollar_symbol")] == 0.0


# ---------------------------------------------------------------------------
# The cache: one measurement, shared between extraction and promotion
# ---------------------------------------------------------------------------
#
# load_accuracy() is not cheap. src.services.db.get_conn has no pool — every call is a
# fresh psycopg2.connect — and _LOAD_SQL filters on decided_at, which has no index, so an
# uncached call is a TCP connection plus a sequential scan of the whole verdict table. Both
# callers (dispatch, once per document; promote, once per promoted document) were paying it
# per document.

import pytest  # noqa: E402

from src.services.extraction_feedback import accuracy as acc_mod  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_cache():
    acc_mod.clear_cache()
    yield
    acc_mod.clear_cache()


class _Conn:
    """Counts how many times the load query is actually run."""

    def __init__(self, rows, autocommit=True):
        self.rows, self.autocommit, self.queries = rows, autocommit, 0
        self._cur = self

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        if "bp_extraction_verdict" in sql:
            self.queries += 1
        self.description = [("doc_type",), ("field_name",), ("pattern_name",),
                            ("source",), ("verdict",)]

    def fetchall(self):
        return self.rows


def test_the_second_document_does_not_pay_for_the_query_again():
    conn = _Conn([("invoice", "currency", "dollar_symbol", "regex", "corrected")] * MIN_SAMPLE)
    first = acc_mod.cached_accuracy(conn)
    second = acc_mod.cached_accuracy(conn)
    assert conn.queries == 1
    assert first == second == {("invoice", "currency", "dollar_symbol"): 0.0}


def test_an_expired_entry_is_refreshed():
    conn = _Conn([("invoice", "currency", "dollar_symbol", "regex", "corrected")] * MIN_SAMPLE)
    acc_mod.cached_accuracy(conn)
    acc_mod.cached_accuracy(conn, ttl_seconds=-1)
    assert conn.queries == 2


def test_a_failed_refresh_keeps_the_last_good_map_and_retries():
    # It must NOT pin an empty map as "fresh" for 15 minutes, and it must not raise into
    # the promotion transaction it is running inside.
    good = _Conn([("invoice", "currency", "dollar_symbol", "regex", "confirmed")] * MIN_SAMPLE)
    acc_mod.cached_accuracy(good)
    warm = dict(acc_mod.cached_accuracy(good))

    class _Broken(_Conn):
        def execute(self, sql, params=None):
            raise RuntimeError('relation "proc.bp_extraction_verdict" does not exist')

    broken = _Broken([])
    assert acc_mod.cached_accuracy(broken, ttl_seconds=-1) == warm
    assert acc_mod.cached_accuracy(broken, ttl_seconds=-1) == warm  # retried, not pinned


def test_a_refresh_on_a_transactional_connection_is_savepoint_isolated():
    """promote() calls this mid-transaction. A failed read must roll back to a savepoint
    rather than leaving Postgres with an aborted transaction, which would take the whole
    promotion down at the next commit."""
    issued: list[str] = []

    class _Txn(_Conn):
        def __init__(self):
            super().__init__([], autocommit=False)

        def execute(self, sql, params=None):
            issued.append(sql)
            if "bp_extraction_verdict" in sql:
                raise RuntimeError('column "verdict" does not exist')

    acc_mod.cached_accuracy(_Txn(), ttl_seconds=-1)
    assert any("SAVEPOINT" in s.upper() for s in issued)
    assert any("ROLLBACK TO SAVEPOINT" in s.upper() for s in issued)
