"""The Action Centre row values the writer builds (no database)."""
from src.services.triage import writer
from src.services.triage.model import Severity
from tests.triage.helpers import deal, inv, line, pipeline, po


def test_line_arithmetic_expected_value_says_expected_not_none():
    ds = deal(po(), inv(lines=[line(1, qty="1", price="13531.49", amount="13631.49")]))
    (f,) = [f for f in pipeline(ds).findings if f.rule_id == "line_arithmetic"]
    v = writer._finding_values("RUN-1", f)
    assert v[10] == "expected: 13531.49"
    assert v[9] == "INV-1: 13631.49"


def test_missing_values_are_a_dash_not_none():
    ds = deal(po(), inv("INV-9", po_id=None))
    (f,) = [f for f in pipeline(ds).findings if f.rule_id == "no_po"]
    f.lead.severity = Severity.S2        # no_po is capped at S3; lift it to reach the writer
    v = writer._finding_values("RUN-1", f)
    assert v[9] == "INV-9: -" and v[10] == "expected: -"


def test_every_mirrored_rule_has_an_action_centre_issue_type():
    from src.services.triage.model import CATEGORY
    assert set(writer.MIRROR_ISSUE_TYPE) == set(CATEGORY) - {"duplicate"}
    assert len(set(writer.MIRROR_ISSUE_TYPE.values())) == len(writer.MIRROR_ISSUE_TYPE)


# --- the Action Centre mirror row carries money (GBP) --------------------------

from decimal import Decimal as D  # noqa: E402

from src.services.triage.model import Finding, Outcome, Result  # noqa: E402


def _res(rule="quantity", claim="1", auth="1", claim_amount=None, auth_amount=None,
         fx="1", line_ref="1", currency="GBP"):
    return Result(deal_id="DEAL-1", rule_id=rule, field_class="quantity",
                  outcome=Outcome.CONFLICT, claim_doc="INV-1", field_name=rule,
                  claim_line=line_ref, claim_value=claim, auth_value=auth,
                  claim_amount=None if claim_amount is None else D(claim_amount),
                  auth_amount=None if auth_amount is None else D(auth_amount),
                  currency=currency, fx_to_gbp=None if fx is None else D(fx))


def test_mirror_values_sum_the_money_over_a_group_of_causes():
    f = Finding("DEAL-1", "quantity",
                [_res(claim="6", auth="3.00", claim_amount="18490.88", auth_amount="9245.44"),
                 _res(claim="12", auth="10", claim_amount="144", auth_amount="120",
                      line_ref="2")], "PO-1|quantity")
    assert writer._mirror_values(f) == ("18634.88", "9365.44")


def test_mirror_values_convert_to_gbp():
    f = Finding("DEAL-1", "unit_price",
                [_res("unit_price", "13.50", "12.00", "4050", "3600", fx="0.5",
                      currency="EUR")], "k")
    assert writer._mirror_values(f) == ("2025.00", "1800.00")


def test_mirror_values_without_an_fx_rate_fall_back_to_the_text():
    f = Finding("DEAL-1", "unit_price",
                [_res("unit_price", "13.50", "12.00", "4050", "3600", fx=None)], "k")
    assert writer._mirror_values(f) == ("13.50", "12.00")


def test_mirror_values_fall_back_when_any_cause_lacks_an_amount():
    f = Finding("DEAL-1", "quantity",
                [_res(claim="6", auth="3.00", claim_amount="18", auth_amount="9"),
                 _res(claim="12", auth="10", line_ref="2")], "PO-1|quantity")
    assert writer._mirror_values(f) == ("6", "3.00")


def test_mirror_values_for_a_currency_finding_stay_text():
    ds = deal(po(), inv(currency="EUR"))
    (f,) = [f for f in pipeline(ds).findings if f.rule_id == "currency"]
    assert writer._mirror_values(f) == ("EUR", "GBP")


def test_mirror_values_for_the_price_error_deal():
    ds = deal(po(lines=[line(1, qty="300", price="12.00")]),
              inv(lines=[line(1, qty="300", price="13.50")]))
    (f,) = [f for f in pipeline(ds).findings if f.rule_id == "unit_price"]
    assert writer._mirror_values(f) == ("4050.00", "3600.00")
