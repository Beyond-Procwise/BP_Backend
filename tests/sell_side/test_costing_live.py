from decimal import Decimal as D

import pytest

from src.services.sell_side import costing
from src.services.sell_side._db import NotFound, dict_cursor
from tests.sell_side.conftest import live, seed_item

pytestmark = live


def test_the_highest_break_at_or_below_the_quantity_applies(live_db):
    conn, dist = live_db
    item = seed_item(conn, dist, "LIVETEST-T1", cost="10.0000",
                     tiers=[(D("10"), D("9.0000")), (D("50"), D("8.0000"))])
    cur = dict_cursor(conn)
    assert costing.cost_at(cur, item, D("5")).unit_cost == D("10.0000")
    at_10 = costing.cost_at(cur, item, D("10"))
    assert (at_10.unit_cost, at_10.cost_tier_applied) == (D("9.0000"), D("10"))
    assert costing.cost_at(cur, item, D("75")).unit_cost == D("8.0000")


def test_no_cost_anywhere_is_none_not_zero(live_db):
    conn, dist = live_db
    item = seed_item(conn, dist, "LIVETEST-T2", cost=None)
    assert costing.cost_at(dict_cursor(conn), item, D("1")).unit_cost is None


def test_a_missing_item_is_not_found(live_db):
    conn, _ = live_db
    with pytest.raises(NotFound):
        costing.cost_at(dict_cursor(conn), -1, D("1"))
