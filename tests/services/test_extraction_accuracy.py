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
