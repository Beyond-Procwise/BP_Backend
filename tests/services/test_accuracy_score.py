"""Complete and correct are different words.

A document can have every field filled and every one of them wrong. confidence_score cannot
tell you that — it counts filled fields. This can, once there are verdicts behind it.
"""
from decimal import Decimal

from src.services.extraction.promotion import _compute_accuracy_score
from src.services.extraction.types import Candidate


def _cand(field, value, pattern):
    return Candidate(field=field, value=value, span=None, source="regex",
                     pattern_name=pattern, confidence=0.9)


def test_no_measurements_means_no_score_not_a_zero():
    # Every row will say this until the loop has run for a while. Zero would read as
    # "we know this document is wrong", which is a different and false claim.
    assert _compute_accuracy_score("invoice", {"currency": "GBP"},
                                   [_cand("currency", "GBP", "iso_code_in_text")], {}) is None


def test_it_averages_the_readers_that_actually_produced_this_document():
    acc = {("invoice", "currency", "iso_code_in_text"): 1.0,
           ("invoice", "invoice_amount", "total_labelled"): 0.5}
    got = _compute_accuracy_score(
        "invoice", {"currency": "GBP", "invoice_amount": 100},
        [_cand("currency", "GBP", "iso_code_in_text"),
         _cand("invoice_amount", "100", "total_labelled")], acc)
    assert got == Decimal("75.00")


def test_an_unmeasured_reader_is_skipped_not_counted_as_zero():
    acc = {("invoice", "currency", "iso_code_in_text"): 1.0}
    got = _compute_accuracy_score(
        "invoice", {"currency": "GBP", "invoice_amount": 100},
        [_cand("currency", "GBP", "iso_code_in_text"),
         _cand("invoice_amount", "100", "unmeasured_pattern")], acc)
    assert got == Decimal("100.00")


def test_a_document_whose_readers_keep_being_corrected_scores_low():
    acc = {("invoice", "currency", "dollar_symbol"): 0.2}
    got = _compute_accuracy_score("invoice", {"currency": "USD"},
                                  [_cand("currency", "USD", "dollar_symbol")], acc)
    assert got == Decimal("20.00")


def test_null_columns_contribute_nothing():
    acc = {("invoice", "currency", "iso_code_in_text"): 1.0}
    got = _compute_accuracy_score("invoice", {"currency": "GBP", "buyer_id": None},
                                  [_cand("currency", "GBP", "iso_code_in_text")], acc)
    assert got == Decimal("100.00")
