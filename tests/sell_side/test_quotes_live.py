import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, money, opportunities as opp, outcomes, quotes
from src.services.sell_side._db import StateConflict
from tests.sell_side.conftest import live, seed_item

pytestmark = live
TODAY = dt.date.today()


def _setup(conn, dist, **item_kw):
    accounts.create_account(conn, account_id="LIVETEST-Q", account_name="Livetest Ltd")
    return seed_item(conn, dist, item_kw.pop("sku", "LIVETEST-Q1"), **item_kw)


def _draft(conn, item, **kw):
    kw.setdefault("created_by", "sub-author")
    return quotes.create_draft(
        conn, account_id="LIVETEST-Q", currency="GBP",
        valid_until=TODAY + dt.timedelta(days=30),
        lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00")}], **kw)


def test_a_draft_snapshots_cost_and_totals(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000", list_price="15.0000")
    q = _draft(conn, item)
    (line,) = q["lines"]
    assert q["quote_ref"].startswith(f"SQ-{TODAY:%Y%m%d}-")
    assert (q["status"], q["phase_id"], q["subprocess_id"]) == \
        ("draft", "sales.opportunity", "sales.opportunity.quote-drafted")
    assert (line["unit_cost"], line["list_price_at_quote"], line["line_total"],
            line["line_margin"]) == (D("10.0000"), D("15.0000"), D("56.00"), D("16.00"))
    assert (q["total_ex_tax"], q["total_cost"], q["total_margin"]) == \
        (D("56.00"), D("40.00"), D("16.00"))


def test_a_repricing_after_drafting_does_not_move_the_quote(live_db):
    """Acceptance criterion 5."""
    conn, dist = live_db
    item = _setup(conn, dist, cost="10.0000")
    q = _draft(conn, item)
    with conn.cursor() as cur:  # the catalog reprices: close this version, open a dearer one
        cur.execute("UPDATE proc.bp_catalog_item SET valid_to = now() WHERE catalog_item_id = %s", (item,))
        cur.execute(
            "INSERT INTO proc.bp_catalog_item (source_id, distributor_id, distributor_sku, "
            "item_description, currency, cost_price) SELECT source_id, distributor_id, "
            "distributor_sku, item_description, currency, 12.0000 FROM proc.bp_catalog_item "
            "WHERE catalog_item_id = %s", (item,))
    conn.commit()
    (line,) = quotes.get_quote(conn, q["sales_quote_id"])["lines"]
    assert (line["unit_cost"], line["line_margin"]) == (D("10.0000"), D("16.00"))


def test_a_closed_catalog_version_cannot_be_quoted(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    with conn.cursor() as cur:
        cur.execute("UPDATE proc.bp_catalog_item SET valid_to = now() WHERE catalog_item_id = %s", (item,))
    conn.commit()
    with pytest.raises(ValueError, match="closed"):
        _draft(conn, item)


def test_a_foreign_currency_line_is_refused(live_db):
    conn, dist = live_db
    item = _setup(conn, dist, currency="EUR")
    with pytest.raises(ValueError, match="FX"):
        _draft(conn, item)


def test_an_already_expired_validity_is_refused(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    with pytest.raises(ValueError):
        quotes.create_draft(conn, account_id="LIVETEST-Q", currency="GBP",
                            valid_until=TODAY - dt.timedelta(days=1), created_by="x",
                            lines=[{"catalog_item_id": item, "quantity": D("1"),
                                    "unit_price": D("1")}])


def test_the_happy_path_and_its_ladder(live_db):
    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    qid = q["sales_quote_id"]
    assert quotes.submit(conn, qid, actor="sub-author")["status"] == "in_review"
    a = quotes.approve(conn, qid, approver="sub-approver")
    assert (a["status"], a["approved_by"], a["subprocess_id"]) == \
        ("approved", "sub-approver", "sales.approval.pricing-approval")
    i = quotes.issue(conn, qid, actor="sub-approver")
    assert i["status"] == "issued" and i["issued_at"] is not None


def test_nobody_approves_their_own_quote(live_db):
    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    quotes.submit(conn, q["sales_quote_id"], actor="sub-author")
    with pytest.raises(StateConflict, match="own"):
        quotes.approve(conn, q["sales_quote_id"], approver="sub-author")


def test_an_anonymous_approval_is_refused(live_db):
    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    quotes.submit(conn, q["sales_quote_id"], actor="sub-author")
    with pytest.raises(StateConflict):
        quotes.approve(conn, q["sales_quote_id"], approver=None)


def test_a_draft_cannot_be_issued(live_db):
    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    with pytest.raises(StateConflict):
        quotes.issue(conn, q["sales_quote_id"], actor="sub-approver")


def test_superseding_retires_the_old_quote(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    old = _draft(conn, item)
    new = _draft(conn, item, supersedes_id=old["sales_quote_id"])
    assert new["supersedes_id"] == old["sales_quote_id"]
    assert quotes.get_quote(conn, old["sales_quote_id"])["status"] == "superseded"
    with pytest.raises(StateConflict):
        _draft(conn, item, supersedes_id=old["sales_quote_id"])


def test_a_real_approved_quote_renders_with_no_internal_field(live_db):
    from src.services.sell_side import quote_render as qr

    conn, dist = live_db
    q = _draft(conn, _setup(conn, dist))
    quotes.submit(conn, q["sales_quote_id"], actor="sub-author")
    quotes.approve(conn, q["sales_quote_id"], approver="sub-approver")
    view = qr.customer_view(quotes.get_quote(conn, q["sales_quote_id"]))
    flat = repr(view)
    assert not any(f"'{f}'" in flat for f in qr.INTERNAL_FIELDS)
    assert "10.0000" not in qr.render_html(view)


def test_a_zero_quantity_is_refused(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    with pytest.raises(ValueError, match="quantity"):
        quotes.create_draft(
            conn, account_id="LIVETEST-Q", currency="GBP",
            valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
            lines=[{"catalog_item_id": item, "quantity": D("0"), "unit_price": D("14.00")}])


def test_a_negative_price_is_refused(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    with pytest.raises(ValueError, match="unit_price"):
        quotes.create_draft(
            conn, account_id="LIVETEST-Q", currency="GBP",
            valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
            lines=[{"catalog_item_id": item, "quantity": D("1"), "unit_price": D("-1")}])


def test_a_quote_cannot_supersede_another_accounts(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    accounts.create_account(conn, account_id="LIVETEST-Q2", account_name="Livetest Two Ltd")
    other = quotes.create_draft(
        conn, account_id="LIVETEST-Q2", currency="GBP",
        valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
        lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00")}])
    with pytest.raises(ValueError, match="same account"):
        _draft(conn, item, supersedes_id=other["sales_quote_id"])
    assert quotes.get_quote(conn, other["sales_quote_id"])["status"] == "draft"


# --- final review fixes (2026-09-11) ----------------------------------------

def test_a_line_cannot_claim_another_accounts_opportunity(live_db):
    """Finding 1: without the account check a customer_safe claim written for
    customer B could render on customer A's quote via /customer and .html."""
    conn, dist = live_db
    item = _setup(conn, dist)
    accounts.create_account(conn, account_id="LIVETEST-Q2", account_name="Livetest Two Ltd")
    other_item = seed_item(conn, dist, "LIVETEST-Q2-SKU")
    other_opp = opp.create_opportunity(conn, account_id="LIVETEST-Q2", opportunity_type="upsell",
                                       catalog_item_id=other_item)
    with pytest.raises(ValueError, match="line 1"):
        quotes.create_draft(
            conn, account_id="LIVETEST-Q", currency="GBP",
            valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
            lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00"),
                    "sales_opportunity_id": other_opp["sales_opportunity_id"]}])


def test_a_line_cannot_claim_another_accounts_justification(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    accounts.create_account(conn, account_id="LIVETEST-Q2", account_name="Livetest Two Ltd")
    other_item = seed_item(conn, dist, "LIVETEST-Q2-SKU")
    other_opp = opp.create_opportunity(conn, account_id="LIVETEST-Q2", opportunity_type="upsell",
                                       catalog_item_id=other_item)
    other_just = opp.add_justification(conn, other_opp["sales_opportunity_id"],
                                       kind="price_gap", claim="a claim about customer B")
    with pytest.raises(ValueError, match="line 1"):
        quotes.create_draft(
            conn, account_id="LIVETEST-Q", currency="GBP",
            valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
            lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00"),
                    "justification_id": other_just["justification_id"]}])


def test_a_lines_justification_must_belong_to_the_lines_own_opportunity(live_db):
    """Same account, but the justification is attached to a *different*
    opportunity than the one the line names -- also refused."""
    conn, dist = live_db
    item = _setup(conn, dist)
    opp1 = opp.create_opportunity(conn, account_id="LIVETEST-Q", opportunity_type="upsell",
                                  catalog_item_id=item)
    item2 = seed_item(conn, dist, "LIVETEST-Q-SKU2")
    opp2 = opp.create_opportunity(conn, account_id="LIVETEST-Q", opportunity_type="upsell",
                                  catalog_item_id=item2)
    just2 = opp.add_justification(conn, opp2["sales_opportunity_id"],
                                  kind="price_gap", claim="belongs to opp2, not opp1")
    with pytest.raises(ValueError, match="line 1"):
        quotes.create_draft(
            conn, account_id="LIVETEST-Q", currency="GBP",
            valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
            lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00"),
                    "sales_opportunity_id": opp1["sales_opportunity_id"],
                    "justification_id": just2["justification_id"]}])


def test_a_matching_opportunity_and_justification_on_the_same_account_are_accepted(live_db):
    """The positive case the three guards above must not break."""
    conn, dist = live_db
    item = _setup(conn, dist)
    o = opp.create_opportunity(conn, account_id="LIVETEST-Q", opportunity_type="upsell",
                               catalog_item_id=item)
    j = opp.add_justification(conn, o["sales_opportunity_id"], kind="price_gap", claim="ok")
    q = quotes.create_draft(
        conn, account_id="LIVETEST-Q", currency="GBP",
        valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
        lines=[{"catalog_item_id": item, "quantity": D("4"), "unit_price": D("14.00"),
                "sales_opportunity_id": o["sales_opportunity_id"],
                "justification_id": j["justification_id"]}])
    (line,) = q["lines"]
    assert (line["sales_opportunity_id"], line["justification_id"]) == \
        (o["sales_opportunity_id"], j["justification_id"])


def test_a_4dp_unit_price_is_accepted_and_multiplies_out_exactly(live_db):
    """Finding 2: a price at the column's own precision must store and total
    exactly -- qty 1000 x 1.2346 = 1234.60, not a rounded-down other number."""
    conn, dist = live_db
    item = _setup(conn, dist)
    q = quotes.create_draft(
        conn, account_id="LIVETEST-Q", currency="GBP",
        valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
        lines=[{"catalog_item_id": item, "quantity": D("1000"), "unit_price": D("1.2346")}])
    (line,) = q["lines"]
    assert line["unit_price"] == D("1.2346")
    assert line["line_total"] == money.q2(D("1000") * line["unit_price"])
    assert line["line_total"] == D("1234.60")


def test_a_5dp_unit_price_is_refused_live(live_db):
    conn, dist = live_db
    item = _setup(conn, dist)
    with pytest.raises(ValueError, match="more than 4 decimal"):
        quotes.create_draft(
            conn, account_id="LIVETEST-Q", currency="GBP",
            valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
            lines=[{"catalog_item_id": item, "quantity": D("1000"),
                    "unit_price": D("1.23456")}])


def test_a_quote_with_an_outcome_cannot_be_superseded(live_db):
    """Finding 4: superseding an issued quote that already has a recorded
    outcome would silently erase its won/lost state."""
    conn, dist = live_db
    item = _setup(conn, dist)
    q = _draft(conn, item)
    quotes.submit(conn, q["sales_quote_id"], actor="sub-author")
    quotes.approve(conn, q["sales_quote_id"], approver="sub-approver")
    quotes.issue(conn, q["sales_quote_id"], actor="sub-approver")
    outcomes.record_outcome(conn, sales_quote_id=q["sales_quote_id"], outcome="won",
                            outcome_date=TODAY, recorded_by="sub-approver")
    with pytest.raises(StateConflict):
        _draft(conn, item, supersedes_id=q["sales_quote_id"])
    assert quotes.get_quote(conn, q["sales_quote_id"])["status"] == "issued"


def test_a_discount_percentage_that_would_overflow_the_column_is_refused(live_db):
    """Finding 5: numeric(7,4) tops out at 999.9999; list 1.00 vs price 2000
    computes discount_pct -1999, which must be refused before it ever reaches
    the INSERT and becomes a 500."""
    conn, dist = live_db
    item = _setup(conn, dist, cost="1.0000", list_price="1.0000")
    with pytest.raises(ValueError, match="line 1"):
        quotes.create_draft(
            conn, account_id="LIVETEST-Q", currency="GBP",
            valid_until=TODAY + dt.timedelta(days=30), created_by="sub-author",
            lines=[{"catalog_item_id": item, "quantity": D("1"), "unit_price": D("2000")}])
