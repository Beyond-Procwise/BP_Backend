"""Loader against the real database. Run with PROCWISE_TEST_LIVE_DB=1."""
import os

import pytest

from src.services.db import get_conn
from src.services.triage.loader import list_deal_ids, load_deal_sets

pytestmark = pytest.mark.skipif(os.environ.get("PROCWISE_TEST_LIVE_DB") != "1",
                                reason="needs PROCWISE_TEST_LIVE_DB=1")

KNOWN = "DEALV2-005049"   # Kestrel: one INR PO, three invoices


class CountingCursor:
    def __init__(self, cur):
        self.cur, self.n = cur, 0

    def execute(self, *args):
        self.n += 1
        return self.cur.execute(*args)

    def fetchall(self):
        return self.cur.fetchall()

    def fetchone(self):
        return self.cur.fetchone()


def test_known_deal_loads_with_lines_and_fx():
    with get_conn() as conn:
        sets = load_deal_sets(conn.cursor(), [KNOWN])
    ds = sets[KNOWN]
    assert len(ds.pos) >= 1 and ds.pos[0].lines
    assert len(ds.invoices) >= 3
    assert all(i.lines for i in ds.invoices)
    assert ds.invoices[0].currency == "INR"
    assert ds.invoices[0].fx_to_gbp is not None


def test_unknown_deal_is_absent():
    with get_conn() as conn:
        assert load_deal_sets(conn.cursor(), ["NO-SUCH-DEAL"]) == {}


def test_query_count_does_not_grow_with_batch_size():
    with get_conn() as conn:
        ids = list_deal_ids(conn.cursor())[:50]
        cur = CountingCursor(conn.cursor())
        sets = load_deal_sets(cur, ids)
    assert len(sets) == 50
    # 7 table queries + at most 4 FX lookups per distinct currency (5 in the corpus)
    assert cur.n <= 7 + 4 * 5
