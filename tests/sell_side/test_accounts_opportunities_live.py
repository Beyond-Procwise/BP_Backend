from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, opportunities as opp
from src.services.sell_side._db import NotFound, StateConflict
from tests.sell_side.conftest import live, seed_item

pytestmark = live


def _acct(conn, suffix="A"):
    return accounts.create_account(conn, account_id=f"LIVETEST-{suffix}",
                                   account_name="Livetest Ltd", default_currency="gbp")


def test_an_account_round_trips_with_contacts_and_scope(live_db):
    conn, _ = live_db
    a = _acct(conn)
    assert a["default_currency"] == "GBP"
    accounts.add_contact(conn, a["account_id"], contact_name="Ann", is_primary=True)
    accounts.add_contact(conn, a["account_id"], contact_name="Bob", is_primary=True)
    accounts.set_history_scope(conn, a["account_id"], source_kind="our_invoices",
                               completeness="complete")
    got = accounts.get_account(conn, a["account_id"])
    assert [c["contact_name"] for c in got["contacts"] if c["is_primary"]] == ["Bob"]
    assert got["history_scope"][0]["source_kind"] == "our_invoices"


def test_a_duplicate_account_id_conflicts(live_db):
    conn, _ = live_db
    _acct(conn)
    with pytest.raises(StateConflict):
        _acct(conn)


def test_an_unknown_scope_kind_is_refused(live_db):
    conn, _ = live_db
    a = _acct(conn)
    with pytest.raises(ValueError):
        accounts.set_history_scope(conn, a["account_id"], source_kind="gossip",
                                   completeness="complete")


def test_an_opportunity_prices_off_the_tier_and_leaves_probability_null(live_db):
    conn, dist = live_db
    a = _acct(conn)
    item = seed_item(conn, dist, "LIVETEST-O1", cost="10.0000", tiers=[(D("10"), D("9.0000"))])
    o = opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="upsell",
                               catalog_item_id=item, expected_quantity=D("10"),
                               expected_unit_price=D("12.50"))
    assert o["currency"] == "GBP"
    assert (o["expected_revenue"], o["expected_cost"], o["expected_margin"]) == \
        (D("125.00"), D("90.00"), D("35.00"))
    assert o["win_probability"] is None and o["win_probability_basis"] is None
    assert (o["phase_id"], o["subprocess_id"]) == ("sales.opportunity", "sales.opportunity.qualified")


def test_no_price_means_no_revenue_and_no_margin(live_db):
    conn, dist = live_db
    a = _acct(conn)
    item = seed_item(conn, dist, "LIVETEST-O2")
    o = opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="refill",
                               catalog_item_id=item, expected_quantity=D("3"))
    assert (o["expected_revenue"], o["expected_margin"]) == (None, None)
    assert o["expected_cost"] == D("30.00")


def test_a_currency_other_than_the_catalogs_is_refused_not_converted(live_db):
    conn, dist = live_db
    a = _acct(conn)
    item = seed_item(conn, dist, "LIVETEST-O3", currency="GBP")
    with pytest.raises(ValueError, match="FX"):
        opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="upsell",
                               catalog_item_id=item, currency="EUR")


def test_a_justification_attaches_and_a_blank_claim_is_refused(live_db):
    conn, dist = live_db
    a = _acct(conn)
    o = opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="upgrade",
                               currency="GBP")
    opp.add_justification(conn, o["sales_opportunity_id"], kind="end_of_life",
                          claim="The 2960X reached end of sale on 2026-06-30.",
                          customer_safe=True)
    with pytest.raises(ValueError):
        opp.add_justification(conn, o["sales_opportunity_id"], kind="benchmark", claim="  ")
    got = opp.get_opportunity(conn, o["sales_opportunity_id"])
    assert [j["kind"] for j in got["justifications"]] == ["end_of_life"]


def test_a_stage_off_the_ladder_is_refused(live_db):
    conn, _ = live_db
    a = _acct(conn)
    o = opp.create_opportunity(conn, account_id=a["account_id"], opportunity_type="upsell",
                               currency="GBP")
    opp.set_stage(conn, o["sales_opportunity_id"], phase_id="sales.margin",
                  subprocess_id="sales.margin.discount-checked")
    with pytest.raises(ValueError):
        opp.set_stage(conn, o["sales_opportunity_id"], phase_id="sales.margin",
                      subprocess_id="sales.approval.pricing-approval")


def test_a_missing_opportunity_is_not_found(live_db):
    conn, _ = live_db
    with pytest.raises(NotFound):
        opp.get_opportunity(conn, -1)


def test_win_probability_cannot_be_passed_in():
    import inspect
    assert "win_probability" not in inspect.signature(opp.create_opportunity).parameters
