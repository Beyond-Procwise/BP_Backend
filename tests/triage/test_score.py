from decimal import Decimal as D

from src.services.triage.model import Outcome, Result, Severity
from src.services.triage.score import score_result, threshold_gbp
from tests.triage.helpers import make_cfg

CFG = make_cfg()


def res(rule="unit_price", cls="money", outcome=Outcome.CONFLICT, exposure="450",
        basis="9000", fx="1", conf=1.0):
    return Result(deal_id="D", rule_id=rule, field_class=cls, outcome=outcome,
                  claim_doc="INV-1", field_name="x", exposure=D(exposure),
                  basis_total=D(basis) if basis else None, fx_to_gbp=D(fx) if fx else None,
                  confidence=conf, currency="GBP")


def test_threshold_floor_and_ceiling():
    assert threshold_gbp(res(basis="9000"), CFG) == D("45")
    assert threshold_gbp(res(basis="1000"), CFG) == D("25")
    assert threshold_gbp(res(basis="10000000"), CFG) == D("5000")


def test_worked_examples_from_the_spec():
    assert score_result(res(exposure="450"), CFG).severity == Severity.S1
    s2 = score_result(res(exposure="45"), CFG)
    assert s2.severity == Severity.S2 and s2.score == 40.0
    s3 = score_result(res(exposure="10"), CFG)
    assert s3.severity == Severity.S3 and round(s3.score, 1) == 13.9


def test_matches_and_notes():
    assert score_result(res(outcome=Outcome.MATCH), CFG).severity == Severity.S0
    assert score_result(res(outcome=Outcome.WITHIN_TOL), CFG).severity == Severity.S0
    assert score_result(res(outcome=Outcome.EXPLAINED), CFG).severity == Severity.S3


def test_unverifiable_is_capped_at_s2():
    assert score_result(res(outcome=Outcome.UNVERIFIABLE, exposure="100000"), CFG).severity == Severity.S2


def test_unverifiable_money_is_at_least_s2():
    assert score_result(res(outcome=Outcome.UNVERIFIABLE, exposure="0"), CFG).severity == Severity.S2


def test_missing_fx_caps_at_s2():
    r = score_result(res(exposure="100000", fx=None), CFG)
    assert r.severity == Severity.S2 and r.score_inputs["fx_missing"] is True


def test_always_s1_rules_ignore_the_amount():
    for rule in ("currency", "cumulative_total", "duplicate"):
        assert score_result(res(rule=rule, exposure="0.01"), CFG).severity == Severity.S1


def test_override_beats_fx_cap():
    assert score_result(res(rule="currency", exposure="100000", fx=None), CFG).severity == Severity.S1


def test_min_s2_rules():
    assert score_result(res(rule="payment_terms", cls="terms", exposure="0"), CFG).severity == Severity.S2
    assert score_result(res(rule="invoice_date", cls="date", exposure="0"), CFG).severity == Severity.S2


def test_max_s3_rules():
    assert score_result(res(rule="description", cls="description", exposure="1000000"), CFG).severity == Severity.S3
    assert score_result(res(rule="no_po", cls="reference", outcome=Outcome.ABSENT_AUTHORITATIVE,
                            exposure="1000000"), CFG).severity == Severity.S3


def test_low_confidence_lowers_the_score():
    assert score_result(res(exposure="450", conf=0.5), CFG).score == 40.0
