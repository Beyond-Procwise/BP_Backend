"""The hand-set prior is a starting guess. A measurement beats it.

Two safety rules, both non-negotiable: a measured rate may never raise a reader ABOVE its
hand-set prior (learning must not quietly promote documents that used to be reviewed), and a
reader with no measurement keeps exactly the prior it always had.
"""
import pytest

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


def test_reapplying_the_same_rate_to_an_already_demoted_pattern_is_a_no_op():
    # dispatch.py calls apply_observed on the SAME process-wide registry singleton on
    # every document, not once per accuracy refresh. A demoted prior must not keep
    # sliding downward each time the (unchanged) measurement is re-applied to it.
    reg = PatternRegistry("invoice")
    accuracy = {("invoice", "currency", "dollar_symbol"): 0.20}
    first = reg.apply_observed(accuracy)
    once = _priors(reg, "currency")["dollar_symbol"]
    second = reg.apply_observed(accuracy)
    twice = _priors(reg, "currency")["dollar_symbol"]
    assert first == 1
    assert second == 0
    assert once == twice == 0.20


def test_a_reader_whose_rate_recovers_comes_back_up_to_its_hand_set_prior():
    """Every application is computed against the YAML baseline, not against whatever the
    last one left behind.

    This registry is a process-wide singleton and the accuracy map behind it is refreshed
    on a 15-minute timer. Comparing a fresh rate against an already-demoted prior would
    ratchet: 0.55 is not < 0.20, so a reader knocked down by one bad window could never
    recover until the process restarted — and two workers started at different times would
    read the same document differently. It still may never rise ABOVE the prior a human
    wrote; recovery stops exactly there."""
    reg = PatternRegistry("invoice")
    baseline = _priors(reg, "currency")["dollar_symbol"]

    reg.apply_observed({("invoice", "currency", "dollar_symbol"): 0.20})
    assert _priors(reg, "currency")["dollar_symbol"] == 0.20

    # People stop correcting it; the measured rate climbs back above the prior.
    changed = reg.apply_observed({("invoice", "currency", "dollar_symbol"): 0.95})
    assert changed == 1
    assert _priors(reg, "currency")["dollar_symbol"] == baseline

    # A partial recovery lands on the measurement, still below the prior.
    reg.apply_observed({("invoice", "currency", "dollar_symbol"): 0.40})
    assert _priors(reg, "currency")["dollar_symbol"] == 0.40


def test_the_measurement_disappearing_restores_the_prior_it_replaced():
    # A rate falls out of the 180-day window, or drops back below MIN_SAMPLE. "No
    # measurement" means "use the hand-set prior" — it must not leave the last demotion
    # frozen in place forever.
    reg = PatternRegistry("invoice")
    baseline = _priors(reg, "currency")["dollar_symbol"]
    reg.apply_observed({("invoice", "currency", "dollar_symbol"): 0.20})
    assert reg.apply_observed({}) == 1
    assert _priors(reg, "currency")["dollar_symbol"] == baseline


def test_apply_observed_runs_before_run_pattern_extractor_reads_the_registry(monkeypatch):
    """Pins the ORDER, not just the effect.

    get_registry(doc_type) is a process-wide singleton. dispatch.run_pattern_extractor
    resolves that same instance and reads registry.patterns_for(field) to pick which
    reader wins per field for THIS document. If learning is applied any later than
    immediately before that first read, the demotion has no effect on the document in
    front of us — it would only take hold starting with the next document. This test
    fails if apply_observed is moved back to after candidate production, even though
    that broken ordering still eventually mutates the singleton (and would pass a test
    that only checks the prior ends up lowered)."""
    from src.services.extraction import dispatch as dispatch_mod
    from src.services.extraction.pattern_registry import clear_cache, get_registry

    class _StopHere(Exception):
        pass

    clear_cache()
    seen: dict = {}

    def _fake_run_pattern_extractor(parsed, doc_type):
        reg = get_registry(doc_type)
        pat = next(p for p in reg._by_field["currency"] if p.name == "dollar_symbol")
        seen["prior_at_l1_call_time"] = pat.prior_confidence
        raise _StopHere("stop before any downstream (persistence/DB) side effects run")

    monkeypatch.setattr(dispatch_mod, "parse_document", lambda file_path: object())
    monkeypatch.setattr(dispatch_mod, "run_pattern_extractor", _fake_run_pattern_extractor)
    monkeypatch.setattr(
        dispatch_mod, "_cached_accuracy",
        lambda: {("invoice", "currency", "dollar_symbol"): 0.05},
    )

    try:
        with pytest.raises(_StopHere):
            dispatch_mod.dispatch_document(
                process_monitor_id=None, file_path="unused", doc_type="invoice",
            )
        # Default (unlearned) prior_confidence for dollar_symbol is 0.75 (loader default).
        # If apply_observed ran after run_pattern_extractor (the bug), this would still
        # read 0.75 at call time — the demotion would land one document too late.
        assert seen["prior_at_l1_call_time"] == 0.05
    finally:
        clear_cache()
