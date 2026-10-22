"""Agreed customer prices and distributor special bids (reseller review items 1 and 2)."""
import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, agreed_pricing, costing, quotes, special_bids
from src.services.sell_side._db import StateConflict, dict_cursor
from tests.sell_side.conftest import live, seed_item

pytestmark = live
TODAY = dt.date.today()
DAY = dt.timedelta(days=1)


def _setup(conn, dist, sku="LIVETEST-P1", **kw):
    accounts.create_account(conn, account_id="LIVETEST-P", account_name="Livetest Ltd")
    return seed_item(conn, dist, sku, **kw)


def _price(conn, dist, sku="LIVETEST-P1", **kw):
    kw.setdefault("price_kind", "fixed")
    kw.setdefault("unit_price", D("12.0000"))
    kw.setdefault("valid_from", TODAY - DAY)
    return agreed_pricing.register(
        conn, account_id=kw.pop("account_id", "LIVETEST-P"), distributor_id=dist,
        distributor_sku=sku, currency=kw.pop("currency", "GBP"), created_by="LIVETEST-u", **kw)


def _bid(conn, dist, sku="LIVETEST-P1", **kw):
    kw.setdefault("cost_price", D("7.0000"))
    kw.setdefault("valid_from", TODAY - DAY)
    kw.setdefault("valid_to", TODAY + 30 * DAY)
    return special_bids.register(
        conn, account_id=kw.pop("account_id", "LIVETEST-P"), distributor_id=dist,
        distributor_sku=sku, bid_reference=kw.pop("bid_reference", "LIVETEST-BID-1"),
        currency=kw.pop("currency", "GBP"), created_by="LIVETEST-u", **kw)


def _age(conn, table, id_col, row_id):
    """Make a registered row end yesterday (registration itself refuses a past expiry)."""
    with conn.cursor() as cur:
        cur.execute(f"UPDATE proc.{table} SET valid_from = %s, valid_to = %s WHERE {id_col} = %s",
                    (TODAY - 9 * DAY, TODAY - DAY, row_id))
    conn.commit()


def _draft(conn, item, qty="4", price=None, **kw):
    line = {"catalog_item_id": item, "quantity": D(qty)}
    if price is not None:
        line["unit_price"] = D(price)
    return quotes.create_draft(
        conn, account_id="LIVETEST-P", currency="GBP", valid_until=TODAY + 30 * DAY,
        lines=[line], created_by="LIVETEST-author", **kw)


def _internal_line(conn, q):
    return quotes.get_quote_internal(conn, q["sales_quote_id"])["lines"][0]


# --- agreed customer prices ------------------------------------------------------------

def test_a_line_with_no_price_takes_the_agreed_fixed_price(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000", list_price="15.0000")
    agreed = _price(conn, dist, unit_price=D("12.5000"))
    q = _draft(conn, item)
    assert q["lines"][0]["unit_price"] == D("12.5000")
    il = _internal_line(conn, q)
    assert (il["price_source"], il["account_price_id"]) == ("agreed_price", agreed["account_price_id"])


def test_a_discount_agreement_is_taken_off_the_list_price(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, list_price="20.0000")
    _price(conn, dist, price_kind="discount_pct", unit_price=None, discount_pct=D("10"))
    assert _draft(conn, item).get("lines")[0]["unit_price"] == D("18.0000")


def test_a_discount_agreement_on_an_item_with_no_list_price_gives_no_price(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, list_price=None)
    _price(conn, dist, price_kind="discount_pct", unit_price=None, discount_pct=D("10"))
    with pytest.raises(ValueError, match="no unit_price"):
        _draft(conn, item)


def test_a_price_entered_by_hand_still_wins_and_says_so(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    _price(conn, dist, unit_price=D("12.0000"))
    q = _draft(conn, item, price="14.00")
    assert q["lines"][0]["unit_price"] == D("14.0000")
    assert _internal_line(conn, q)["price_source"] == "entered"


def test_the_highest_quantity_break_at_or_below_the_quantity_applies(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    _price(conn, dist, unit_price=D("13.0000"), min_quantity=D("0"))
    _price(conn, dist, unit_price=D("11.0000"), min_quantity=D("10"))
    assert _draft(conn, item, qty="4")["lines"][0]["unit_price"] == D("13.0000")
    assert _draft(conn, item, qty="10")["lines"][0]["unit_price"] == D("11.0000")


@pytest.mark.parametrize("override", [
    {"_aged": True},                                # ended
    {"valid_from": TODAY + DAY},                    # not started
    {"min_quantity": D("100")},                     # needs more units
    {"currency": "EUR"},                            # another currency, never converted
    {"account_id": "LIVETEST-OTHER"},               # another customer's
])
def test_an_agreement_that_does_not_apply_gives_no_price(live_db, override):
    conn, dist = live_db
    item = _setup(conn, dist)
    accounts.create_account(conn, account_id="LIVETEST-OTHER", account_name="Other Ltd")
    aged = override.pop("_aged", False)
    row = _price(conn, dist, **override)
    if aged:
        _age(conn, "bp_account_price", "account_price_id", row["account_price_id"])
    with pytest.raises(ValueError, match="no unit_price"):
        _draft(conn, item)


def test_a_withdrawn_agreement_gives_no_price(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    agreed = _price(conn, dist)
    agreed_pricing.withdraw(conn, agreed["account_price_id"], actor="LIVETEST-u")
    with pytest.raises(ValueError, match="no unit_price"):
        _draft(conn, item)


def test_an_agreement_survives_a_catalog_reprice(live_db):
    """Keyed by SKU, not by the catalog row a reprice closes."""
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    _price(conn, dist, unit_price=D("12.0000"))
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET valid_to = now() WHERE catalog_item_id = %s", (item,))
        cur.execute(
            "INSERT INTO proc.bp_catalog_item (source_id, distributor_id, distributor_sku, "
            "item_description, currency, list_price, cost_price) SELECT source_id, distributor_id, "
            "distributor_sku, item_description, currency, list_price, 11.0000 "
            "FROM proc.bp_catalog_item WHERE catalog_item_id = %s RETURNING catalog_item_id", (item,))
        new_item = cur.fetchone()[0]
    conn.commit()
    assert _draft(conn, new_item)["lines"][0]["unit_price"] == D("12.0000")


def test_a_bad_agreement_is_refused_at_registration(live_db):
    conn, dist = live_db
    _setup(conn, dist)
    with pytest.raises(ValueError):
        _price(conn, dist, price_kind="fixed", unit_price=None)
    with pytest.raises(ValueError):
        _price(conn, dist, price_kind="discount_pct", unit_price=None, discount_pct=D("100"))
    with pytest.raises(ValueError):
        _price(conn, dist, valid_to=TODAY - 5 * DAY)


# --- distributor special bids ----------------------------------------------------------

def test_a_special_bid_replaces_the_cost_for_its_customer_and_is_recorded(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000", tiers=[(D("2"), D("9.0000"))])
    bid = _bid(conn, dist, cost_price=D("7.0000"))
    q = _draft(conn, item, price="14.00")
    il = _internal_line(conn, q)
    assert il["unit_cost"] == D("7.0000")
    assert (il["cost_source"], il["special_bid_id"]) == ("special_bid", bid["special_bid_id"])
    assert il["line_margin"] == D("28.00")


@pytest.mark.parametrize("override", [
    {"_aged": True},                                            # expired
    {"min_quantity": D("10")},                                  # below its band
    {"max_quantity": D("3")},                                   # above what was approved
    {"currency": "EUR"},
    {"account_id": "LIVETEST-OTHER"},                           # registered for someone else
])
def test_a_bid_that_does_not_apply_leaves_the_catalog_cost(live_db, override):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    accounts.create_account(conn, account_id="LIVETEST-OTHER", account_name="Other Ltd")
    aged = override.pop("_aged", False)
    row = _bid(conn, dist, **override)
    if aged:
        _age(conn, "bp_catalog_special_bid", "special_bid_id", row["special_bid_id"])
    il = _internal_line(conn, _draft(conn, item, price="14.00"))
    assert (il["unit_cost"], il["cost_source"], il["special_bid_id"]) == (D("10.0000"), "catalog", None)


def test_a_withdrawn_bid_leaves_the_catalog_cost(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    bid = _bid(conn, dist)
    special_bids.withdraw(conn, bid["special_bid_id"], actor="LIVETEST-u")
    assert _internal_line(conn, _draft(conn, item, price="14.00"))["cost_source"] == "catalog"


def test_the_cheapest_applicable_bid_wins(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    _bid(conn, dist, cost_price=D("8.0000"), bid_reference="LIVETEST-BID-A")
    cheap = _bid(conn, dist, cost_price=D("6.5000"), bid_reference="LIVETEST-BID-B")
    il = _internal_line(conn, _draft(conn, item, price="14.00"))
    assert (il["unit_cost"], il["special_bid_id"]) == (D("6.5000"), cheap["special_bid_id"])


def test_costing_without_an_account_ignores_bids(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    _bid(conn, dist)
    assert costing.cost_at(dict_cursor(conn), item, D("4")).unit_cost == D("10.0000")


def test_a_bid_must_expire_and_have_a_reference(live_db):
    conn, dist = live_db
    _setup(conn, dist)
    with pytest.raises(ValueError):
        _bid(conn, dist, valid_to=None)
    with pytest.raises(ValueError):
        _bid(conn, dist, bid_reference="  ")


def _to_review(conn, item):
    q = _draft(conn, item, price="14.00")
    quotes.submit(conn, q["sales_quote_id"], actor="LIVETEST-author")
    return q["sales_quote_id"]


def test_a_quote_resting_on_an_expired_bid_cannot_be_approved(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    bid = _bid(conn, dist)
    qid = _to_review(conn, item)
    _age(conn, "bp_catalog_special_bid", "special_bid_id", bid["special_bid_id"])
    with pytest.raises(StateConflict, match="special bid"):
        quotes.approve(conn, qid, approver="LIVETEST-approver")


def test_a_quote_resting_on_a_live_bid_is_approved(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    _bid(conn, dist)
    qid = _to_review(conn, item)
    assert quotes.approve(conn, qid, approver="LIVETEST-approver")["status"] == "approved"


def test_the_customer_view_carries_neither_the_bid_cost_nor_its_reference(live_db):
    from src.services.sell_side import quote_render

    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    _bid(conn, dist, cost_price=D("7.3300"), bid_reference="LIVETEST-SECRET-REF")
    qid = _to_review(conn, item)
    quotes.approve(conn, qid, approver="LIVETEST-approver")
    text = repr(quote_render.customer_view(
        quotes.get_quote(conn, qid), quotes.get_quote_internal(conn, qid)))
    assert "7.33" not in text and "LIVETEST-SECRET-REF" not in text
