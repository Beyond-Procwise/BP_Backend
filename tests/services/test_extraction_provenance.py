"""What produced each value we persisted.

Without this, no human correction can be attributed to anything: "the currency was wrong"
is useless unless we know whether a regex, the NER gap-filler or the AI layer said it.

The tests below `test_record_is_a_no_op_without_a_primary_key` cover the fix-round-1
addition: promote() — the one funnel every promotion path (dispatch's inline call, the
HITL NOTIFY listener, promote_pending) goes through — never has the live `candidates`
list in scope, only a snapshot of producer_of() frozen at dispatch time and threaded
through parser_snapshot, plus (on the HITL path) the set of fields a human just fixed.
"""
from src.services.extraction.provenance import producer_of, record, snapshot
from src.services.extraction.types import Candidate


def _cand(field, value, source="regex", pattern_name="anchored_currency_iso", confidence=0.92):
    return Candidate(field=field, value=value, span=None, source=source,
                     pattern_name=pattern_name, confidence=confidence)


def test_a_value_a_pattern_produced_is_attributed_to_that_pattern():
    cands = [_cand("currency", "GBP"), _cand("currency", "USD", pattern_name="dollar_symbol",
                                             confidence=0.58)]
    assert producer_of("currency", "GBP", cands) == ("regex", "anchored_currency_iso", 0.92)


def test_a_value_no_candidate_offered_is_attributed_to_the_ai_layer():
    # context_layer is the authoritative gate and routinely writes a value no regex found.
    # Recording that honestly is the point: an unattributed value is not a regex win.
    assert producer_of("currency", "CAD", [_cand("currency", "GBP")]) == ("context_layer", None, None)


def test_matching_is_on_the_persisted_value_not_the_field_alone():
    cands = [_cand("currency", "GBP"), _cand("invoice_id", "INV-1", pattern_name="id_anchor")]
    assert producer_of("invoice_id", "INV-1", cands)[1] == "id_anchor"


def test_comparison_tolerates_the_shapes_a_column_arrives_in():
    # The candidate carries the captured STRING; the column may hold a Decimal or int.
    from decimal import Decimal
    cands = [_cand("invoice_amount", "1,234.50", pattern_name="total_labelled")]
    assert producer_of("invoice_amount", Decimal("1234.50"), cands)[1] == "total_labelled"
    assert producer_of("quantity", 5, [_cand("quantity", "5", pattern_name="qty")])[1] == "qty"


def test_record_writes_one_row_per_non_null_column():
    class _Cur:
        def __init__(self): self.rows = []
        def execute(self, sql, params): self.rows.append(params)
    cur = _Cur()
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": "GBP", "invoice_amount": 100, "buyer_id": None},
               candidates=[_cand("currency", "GBP")])
    assert n == 2                                  # the NULL column is not provenance
    fields = {p[2] for p in cur.rows}
    assert fields == {"currency", "invoice_amount"}


def test_record_is_a_no_op_without_a_primary_key():
    class _Cur:
        def __init__(self): self.rows = []
        def execute(self, sql, params): self.rows.append(params)
    cur = _Cur()
    assert record(cur, parent_table="proc.bp_invoice_stg", parent_pk="",
                  columns={"currency": "GBP"}, candidates=[]) == 0
    assert cur.rows == []


class _Cur:
    def __init__(self): self.rows = []
    def execute(self, sql, params): self.rows.append(params)


def test_snapshot_freezes_producer_of_for_every_non_null_column():
    # promote() has no live `candidates` — it reads this back from parser_snapshot.
    snap = snapshot(
        columns={"currency": "GBP", "invoice_amount": 100, "buyer_id": None},
        candidates=[_cand("currency", "GBP")],
    )
    assert snap["currency"] == {"source": "regex", "pattern_name": "anchored_currency_iso",
                                 "confidence": 0.92}
    assert snap["invoice_amount"] == {"source": "context_layer", "pattern_name": None,
                                       "confidence": None}
    assert "buyer_id" not in snap                    # NULL column has no producer


def test_record_attributes_from_a_frozen_snapshot_not_live_candidates():
    # This is the shape promote() actually calls with: no candidates in scope, only
    # the snapshot recorded at dispatch time.
    cur = _Cur()
    snap = {"currency": {"source": "regex", "pattern_name": "anchored_currency_iso",
                          "confidence": 0.92}}
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": "GBP", "invoice_amount": 100}, snapshot=snap)
    assert n == 2
    by_field = {p[2]: p for p in cur.rows}
    assert by_field["currency"][3] == "regex"          # from the snapshot
    assert by_field["invoice_amount"][3] == "context_layer"  # absent from snapshot


def test_hitl_fields_override_the_snapshot():
    # A human just corrected `currency` — attribute it to them, not to the regex that
    # produced the value BEFORE the correction. This is exactly the population the
    # per-reader accuracy loop (tasks 2-4) needs to see honestly.
    cur = _Cur()
    snap = {"currency": {"source": "regex", "pattern_name": "anchored_currency_iso",
                          "confidence": 0.92}}
    record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
           columns={"currency": "USD"}, snapshot=snap, hitl_fields={"currency"})
    assert cur.rows[0][3] == "hitl"
    assert cur.rows[0][4] is None                      # no pattern_name for a human fix


def test_record_still_works_with_neither_snapshot_nor_candidates():
    # Legacy _raw rows written before this fix have no _field_provenance in their
    # parser_snapshot — honest fallback is the same "no reader we track offered it"
    # bucket producer_of() already uses.
    cur = _Cur()
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": "GBP"})
    assert n == 1
    assert cur.rows[0][3] == "context_layer"
