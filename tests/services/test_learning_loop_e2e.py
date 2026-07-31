"""The whole loop, in one test: extract -> human corrects -> the reader is trusted less.

This is the claim the feature makes. Everything else is machinery.
"""
from src.services.extraction.pattern_registry import PatternRegistry
from src.services.extraction_feedback.accuracy import MIN_SAMPLE, observed_accuracy
from src.services.extraction_feedback.verdict import verdict_for


def test_repeated_corrections_demote_the_reader_that_keeps_being_wrong():
    # 1. A reader produces a value; a human replaces it. Ten times.
    verdicts = []
    for _ in range(MIN_SAMPLE + 2):
        v = verdict_for("apply_value", resolved_value="CAD", extracted_value="USD")
        assert v == "corrected"
        verdicts.append({"doc_type": "invoice", "field_name": "currency",
                         "pattern_name": "dollar_symbol", "source": "regex", "verdict": v})

    # 2. That becomes a measured rate.
    acc = observed_accuracy(verdicts)
    assert acc[("invoice", "currency", "dollar_symbol")] == 0.0

    # 3. Which the extractor then believes over its hand-set prior.
    reg = PatternRegistry("invoice")
    before = {p.name: p.prior_confidence for p in reg._by_field["currency"]}
    assert before["dollar_symbol"] > 0
    reg.apply_observed(acc)
    after = {p.name: p.prior_confidence for p in reg._by_field["currency"]}
    assert after["dollar_symbol"] == 0.0
    assert after["dollar_symbol"] < before["dollar_symbol"]

    # 4. And it is now the reader of last resort rather than a trusted one.
    assert [p.name for p in reg._by_field["currency"]][-1] == "dollar_symbol"


def test_agreement_leaves_a_good_reader_exactly_where_it_was():
    verdicts = [{"doc_type": "invoice", "field_name": "currency",
                 "pattern_name": "anchored_currency_iso", "source": "regex",
                 "verdict": "confirmed"}] * (MIN_SAMPLE + 2)
    reg = PatternRegistry("invoice")
    before = {p.name: p.prior_confidence for p in reg._by_field["currency"]}
    reg.apply_observed(observed_accuracy(verdicts))
    after = {p.name: p.prior_confidence for p in reg._by_field["currency"]}
    assert after == before
