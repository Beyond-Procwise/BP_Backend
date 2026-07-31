"""The hand-set prior is a starting guess. A measurement beats it.

Two safety rules, both non-negotiable: a measured rate may never raise a reader ABOVE its
hand-set prior (learning must not quietly promote documents that used to be reviewed), and a
reader with no measurement keeps exactly the prior it always had.
"""
from src.services.extraction.pattern_registry import PatternRegistry


def _priors(reg, field):
    return {p.name: p.prior_confidence for p in reg._by_field.get(field, [])}


def test_a_reader_humans_keep_correcting_is_trusted_less():
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")["dollar_symbol"]
    changed = reg.apply_observed({("invoice", "currency", "dollar_symbol"): 0.20})
    assert changed == 1
    assert _priors(reg, "currency")["dollar_symbol"] == 0.20
    assert _priors(reg, "currency")["dollar_symbol"] < before


def test_a_measurement_never_raises_a_reader_above_its_hand_set_prior():
    # Learning may make the pipeline more cautious. It must never make it bolder than a
    # human intended — that would promote documents that used to stop for review.
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")["dollar_symbol"]
    reg.apply_observed({("invoice", "currency", "dollar_symbol"): 1.0})
    assert _priors(reg, "currency")["dollar_symbol"] == before


def test_a_reader_with_no_measurement_is_untouched():
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")
    reg.apply_observed({("invoice", "invoice_id", "some_other_pattern"): 0.1})
    assert _priors(reg, "currency") == before


def test_an_empty_map_changes_nothing():
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")
    assert reg.apply_observed({}) == 0
    assert _priors(reg, "currency") == before


def test_the_preference_order_follows_the_measurement():
    # Order IS the preference between readers: the extractor takes the highest prior first.
    # A demoted reader must actually fall behind the ones that outperform it.
    reg = PatternRegistry("invoice")
    reg.apply_observed({("invoice", "currency", "anchored_currency_iso"): 0.10})
    order = [p.name for p in reg._by_field["currency"]]
    assert order[-1] == "anchored_currency_iso"


def test_another_doc_type_s_measurement_does_not_leak():
    reg = PatternRegistry("invoice")
    before = _priors(reg, "currency")
    reg.apply_observed({("quote", "currency", "dollar_symbol"): 0.05})
    assert _priors(reg, "currency") == before
