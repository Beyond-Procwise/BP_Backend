"""Who must sign a deal, derived from its category and value -- the SpendIQ Pipeline's Approve
stage rules (engine.js APPROVAL_ROUTES / APPROVAL_VALUE_RULES / APPROVAL_NONVALUE_RULES /
dealApprovalRoute), moved to the backend so a board paper can trace them (ruled 2026-09-25).
Every expectation below is what the screen shows for the same deal."""
from decimal import Decimal

from src.services import approval_route as ar


def _rows(route):
    return [(r.who, r.placement) for r in route.rows]


def test_a_category_with_its_own_matrix_uses_it():
    r = ar.route("Logistics", Decimal("10000"), "GBP")
    assert r.matrix == "Logistics" and r.matched is True
    assert _rows(r) == [("AI validation", "required"), ("Category buyer", "required"),
                        ("Finance gate", "required")]
    assert r.rows[1].why == "In the route at any value."


def test_an_unknown_category_falls_back_to_the_default_matrix_and_says_so():
    r = ar.route("Marine fuel", Decimal("10"), "GBP")
    assert r.matrix == "Default" and r.matched is False
    assert _rows(r) == [("AI validation", "required"), ("Approver", "required")]


def test_a_value_rule_places_the_approver_by_value():
    over = ar.route("SaaS / IT", Decimal("480000"), "GBP")
    assert ("CFO sign-off", "required") in _rows(over)
    assert over.rows[-1].why == "Required > £250k — this deal is GBP 480,000."
    under = ar.route("SaaS / IT", Decimal("250000"), "GBP")          # not strictly greater
    assert ("CFO sign-off", "not-required") in _rows(under)
    small = ar.route("Office & facilities", Decimal("49999"), "GBP")
    assert ("Category buyer", "required") in _rows(small)
    assert ("Category buyer", "not-required") in _rows(
        ar.route("Office & facilities", Decimal("50000"), "GBP"))


def test_a_value_in_another_currency_cannot_be_tested_and_says_why():
    r = ar.route("SaaS / IT", Decimal("480000"), "USD")
    cfo = r.rows[-1]
    assert cfo.placement == "untestable"
    assert "set in GBP and this deal is billed in USD" in cfo.why


def test_no_value_cannot_be_tested():
    cfo = ar.route("SaaS / IT", None, None).rows[-1]
    assert cfo.placement == "untestable" and "carries no value" in cfo.why


def test_a_rule_that_is_not_about_value_is_never_guessed():
    r = ar.route("Platform / Enterprise", Decimal("1"), "GBP")
    pricing = [x for x in r.rows if x.who == "Pricing approval"][0]
    assert pricing.placement == "untestable" and "discount threshold" in pricing.why


def test_the_rules_match_the_screen_s_word_for_word():
    """A second copy of a rule set is how two screens come to disagree; this pins the copy."""
    assert ar.APPROVAL_ROUTES["SaaS / IT"][-1] == ("CFO sign-off", "Required > £250k")
    assert ar.APPROVAL_VALUE_RULES == {
        "SaaS / IT|CFO sign-off": ("gt", Decimal(250000)),
        "Office & facilities|Category buyer": ("lt", Decimal(50000))}
    assert set(ar.APPROVAL_ROUTES) == {"SaaS / IT", "Operations", "Logistics",
                                       "Office & facilities", "Platform / Enterprise",
                                       "Prof. services", "Default"}
