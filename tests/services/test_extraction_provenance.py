"""What produced each value we persisted.

Without this, no human correction can be attributed to anything: "the currency was wrong"
is useless unless we know whether a regex, the NER gap-filler or the AI layer said it.

The tests below `test_record_is_a_no_op_without_a_primary_key` cover the fix-round-1
addition: promote() — the one funnel every promotion path (dispatch's inline call, the
HITL NOTIFY listener, promote_pending) goes through — never has the live `candidates`
list in scope, only a snapshot of producer_of() frozen at dispatch time and threaded
through parser_snapshot, plus (on the HITL path) the set of fields a human just fixed.

The tests from `test_hitl_correction_writes_both_the_rejected_reader_and_the_hitl_row`
onward cover fix-round-2: a HITL-corrected field must leave the reader that got it wrong
VISIBLE in bp_extraction_provenance itself (no join to _raw required — see the provenance.py
module docstring for the exact read contract), plus two Minor findings in the same code:
_comparable() crashing on a bool, and date columns never matching a differently-formatted
date candidate string.
"""
from src.services.extraction.provenance import producer_of, record, snapshot, _comparable
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
        def executemany(self, sql, seq): self.rows.extend(seq)
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
        def executemany(self, sql, seq): self.rows.extend(seq)
    cur = _Cur()
    assert record(cur, parent_table="proc.bp_invoice_stg", parent_pk="",
                  columns={"currency": "GBP"}, candidates=[]) == 0
    assert cur.rows == []


class _Cur:
    # record() batches its inserts into one executemany (a dozen round trips per document
    # was the wrong price for evidence nobody reads synchronously); execute stays here so
    # a single-row write would still be seen if the implementation ever went back to it.
    def __init__(self): self.rows = []
    def execute(self, sql, params): self.rows.append(params)
    def executemany(self, sql, seq): self.rows.extend(seq)


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


def test_hitl_correction_writes_both_the_rejected_reader_and_the_hitl_row():
    # A human just corrected `currency`. The reader that got it wrong (regex /
    # anchored_currency_iso) must stay VISIBLE in this table — that's the whole point,
    # per the module docstring's read contract — alongside a source='hitl' row for what
    # is actually stored now (the human's USD, not the regex's wrong claim).
    cur = _Cur()
    snap = {"currency": {"source": "regex", "pattern_name": "anchored_currency_iso",
                          "confidence": 0.92}}
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": "USD"}, snapshot=snap, hitl_fields={"currency"})
    assert n == 2
    by_source = {p[3]: p for p in cur.rows}
    assert set(by_source) == {"regex", "hitl"}
    rejected = by_source["regex"]
    assert rejected[2] == "currency" and rejected[4] == '"anchored_currency_iso"'
    assert rejected[5] == 0.92
    hitl_row = by_source["hitl"]
    assert hitl_row[2] == "currency" and hitl_row[4] is None and hitl_row[5] is None
    # A straight filter answers "which reader produced the value a human rejected" with
    # no join to _raw.parser_snapshot — exactly the contract the module docstring states.
    assert [p for p in cur.rows if p[3] != "hitl"][0][3] == "regex"


def test_hitl_correction_with_no_prior_claim_writes_only_the_hitl_row():
    # A human FILLED A GAP (the field was NULL pre-correction) rather than fixing a
    # wrong value — there is no reader to blame, so no rejected-producer row is invented.
    cur = _Cur()
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": "USD"}, snapshot={}, hitl_fields={"currency"})
    assert n == 1
    assert cur.rows[0][3] == "hitl"


def test_hitl_keep_null_still_records_the_rejected_reader_and_the_correction():
    # keep_null clears a noisy value to NULL — the column is absent from `columns` here
    # exactly as promote() would pass it (raw_data[field] is None post-fix), but the
    # correction event and what it rejected are still real and must not vanish silently.
    cur = _Cur()
    snap = {"currency": {"source": "regex", "pattern_name": "dollar_symbol", "confidence": 0.58}}
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": None}, snapshot=snap, hitl_fields={"currency"})
    assert n == 2
    sources = {p[3] for p in cur.rows}
    assert sources == {"regex", "hitl"}


def test_a_field_never_touched_by_hitl_still_gets_exactly_one_row():
    # Sanity check that the hitl-field sweep doesn't leak into ordinary fields.
    cur = _Cur()
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": "GBP", "invoice_amount": 100}, snapshot={},
               hitl_fields={"currency"})
    # currency: gap-fill hitl (no snapshot entry) -> 1 row. invoice_amount: ordinary -> 1 row.
    assert n == 2
    by_field = {}
    for p in cur.rows:
        by_field.setdefault(p[2], []).append(p[3])
    assert by_field["currency"] == ["hitl"]
    assert by_field["invoice_amount"] == ["context_layer"]


def test_comparable_does_not_crash_on_a_bool():
    # bool is a subclass of int — isinstance(True, (int, float, Decimal)) is True — so
    # the numeric branch must not be allowed to swallow it (Decimal(str(True)) raises).
    assert _comparable(True) == "true"
    assert _comparable(False) == "false"


def test_comparable_normalises_a_python_date_and_a_month_name_candidate_string():
    # The column holds a real date/datetime (post type-binder); the candidate holds
    # whatever text the regex captured. "15 January 2024" is unambiguous (a name can't
    # be a day-of-month) so it's a real match, not a guess.
    import datetime as dt
    assert _comparable(dt.date(2024, 1, 15)) == "2024-01-15"
    assert _comparable(dt.datetime(2024, 1, 15, 9, 30)) == "2024-01-15"
    assert _comparable("15 January 2024") == "2024-01-15"
    assert _comparable("Jan 15, 2024") == "2024-01-15"
    cands = [_cand("invoice_date", "15 January 2024", pattern_name="date_full_month")]
    assert producer_of("invoice_date", dt.date(2024, 1, 15), cands)[1] == "date_full_month"


def test_comparable_normalises_an_unambiguous_numeric_date_but_not_an_ambiguous_one():
    import datetime as dt
    # 25 cannot be a month -> unambiguous D/M/Y.
    assert _comparable("25/01/2024") == _comparable(dt.date(2024, 1, 25))
    # Both parts <= 12 -> genuinely ambiguous (2 Jan vs 1 Feb) -> left unmatched, not guessed.
    assert _comparable("01/02/2024") != _comparable(dt.date(2024, 2, 1))
    assert _comparable("01/02/2024") != _comparable(dt.date(2024, 1, 2))


def test_record_still_works_with_neither_snapshot_nor_candidates():
    # Legacy _raw rows written before this fix have no _field_provenance in their
    # parser_snapshot — honest fallback is the same "no reader we track offered it"
    # bucket producer_of() already uses.
    cur = _Cur()
    n = record(cur, parent_table="proc.bp_invoice_stg", parent_pk="INV-1",
               columns={"currency": "GBP"})
    assert n == 1
    assert cur.rows[0][3] == "context_layer"
