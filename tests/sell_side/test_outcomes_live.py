import datetime as dt
from decimal import Decimal as D

import pytest

from src.services.sell_side import accounts, calibration, opportunities as opp, outcomes, quotes
from src.services.sell_side._db import StateConflict
from tests.sell_side.conftest import live, seed_item

pytestmark = live
TODAY = dt.date.today()


def _issued_quote(conn, dist, n=1):
    accounts.create_account(conn, account_id=f"LIVETEST-W{n}", account_name="Livetest")
    item = seed_item(conn, dist, f"LIVETEST-W{n}")
    o = opp.create_opportunity(conn, account_id=f"LIVETEST-W{n}", opportunity_type="upsell",
                               catalog_item_id=item)
    q = quotes.create_draft(conn, account_id=f"LIVETEST-W{n}", currency="GBP",
                            valid_until=TODAY + dt.timedelta(days=5), created_by="sub-a",
                            lines=[{"catalog_item_id": item, "quantity": D("2"),
                                    "unit_price": D("20"),
                                    "sales_opportunity_id": o["sales_opportunity_id"]}])
    quotes.submit(conn, q["sales_quote_id"], actor="sub-a")
    quotes.approve(conn, q["sales_quote_id"], approver="sub-b")
    quotes.issue(conn, q["sales_quote_id"], actor="sub-b")
    return q["sales_quote_id"], o["sales_opportunity_id"]


def test_a_win_is_recorded_once_and_closes_the_opportunity(live_db):
    conn, dist = live_db
    qid, oid = _issued_quote(conn, dist)
    row = outcomes.record_outcome(conn, sales_quote_id=qid, outcome="won",
                                  outcome_date=TODAY, recorded_by="sub-b")
    assert (row["won_value"], row["won_margin"]) == (D("40.00"), D("20.00"))
    assert opp.get_opportunity(conn, oid)["outcome"] == "won"
    with pytest.raises(StateConflict):
        outcomes.record_outcome(conn, sales_quote_id=qid, outcome="lost",
                                outcome_date=TODAY, recorded_by="sub-b")


def test_an_expired_quote_leaves_the_opportunity_open(live_db):
    conn, dist = live_db
    qid, oid = _issued_quote(conn, dist)
    outcomes.record_outcome(conn, sales_quote_id=qid, outcome="expired",
                            outcome_date=TODAY, recorded_by="sub-b")
    assert quotes.get_quote(conn, qid)["status"] == "expired"
    assert opp.get_opportunity(conn, oid)["outcome"] == "open"


def test_a_lost_reason_on_a_win_is_refused(live_db):
    conn, dist = live_db
    qid, _ = _issued_quote(conn, dist)
    with pytest.raises(ValueError):
        outcomes.record_outcome(conn, sales_quote_id=qid, outcome="won", lost_reason="price",
                                outcome_date=TODAY, recorded_by="sub-b")


def _skip_if_the_wider_corpus_would_be_mutated(conn):
    """Finding 7: calibrate() updates every open opportunity of a type,
    database-wide -- not just LIVETEST rows -- and clean() does not revert
    them. Running this against a corpus that already has non-LIVETEST open
    opportunities would leave a permanent side effect, so refuse instead."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM proc.bp_sales_opportunity "
                    "WHERE account_id NOT LIKE 'LIVETEST-%' AND outcome = 'open'")
        n = cur.fetchone()[0]
    if n:
        pytest.skip(f"{n} non-LIVETEST open opportunities exist; calibrate() "
                    "would mutate them database-wide and clean() cannot undo it")


def test_below_the_threshold_win_probability_stays_null(live_db):
    """Acceptance criterion 7."""
    conn, dist = live_db
    _skip_if_the_wider_corpus_would_be_mutated(conn)
    qid, _ = _issued_quote(conn, dist, n=1)
    outcomes.record_outcome(conn, sales_quote_id=qid, outcome="won",
                            outcome_date=TODAY, recorded_by="sub-b")
    _, open_oid = _issued_quote(conn, dist, n=2)
    calibration.calibrate(conn)  # threshold 30; one closed upsell in the test data
    got = opp.get_opportunity(conn, open_oid)
    assert (got["win_probability"], got["win_probability_basis"]) == (None, None)


def test_at_the_threshold_open_opportunities_are_calibrated(live_db, monkeypatch):
    conn, dist = live_db
    _skip_if_the_wider_corpus_would_be_mutated(conn)
    monkeypatch.setattr(calibration, "_MIN_CLOSED", lambda: 1)
    qid, _ = _issued_quote(conn, dist, n=1)
    outcomes.record_outcome(conn, sales_quote_id=qid, outcome="won",
                            outcome_date=TODAY, recorded_by="sub-b")
    _, open_oid = _issued_quote(conn, dist, n=2)
    calibration.calibrate(conn)
    got = opp.get_opportunity(conn, open_oid)
    assert got["win_probability_basis"] == "calibrated"
    assert got["win_probability"] is not None
