import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, quotes
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
