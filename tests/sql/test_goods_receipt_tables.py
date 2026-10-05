"""The six tables, and the one column that must NOT exist on any of them."""
import os, pytest
from src.services.db import get_conn

pytestmark = pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="live db")

TIERS = ("raw", "stg", "trgt")


@pytest.mark.parametrize("tier", TIERS)
def test_header_and_line_tables_exist(tier):
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT table_name FROM information_schema.tables
                        WHERE table_schema='proc' AND table_name = ANY(%s)""",
                    ([f"bp_goods_receipt_{tier}", f"bp_goods_receipt_line_items_{tier}"],))
        assert len({r[0] for r in cur.fetchall()}) == 2


@pytest.mark.parametrize("tier", TIERS)
def test_no_price_column_anywhere_on_a_goods_receipt(tier):
    """A goods receipt has no prices. A column is an invitation; there is none."""
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT table_name, column_name FROM information_schema.columns
                        WHERE table_schema='proc' AND table_name LIKE %s
                          AND (column_name ~* '(price|amount|total|value|cost|currency|tax)')""",
                    (f"bp_goods_receipt%{tier}",))
        assert cur.fetchall() == []


def test_trgt_is_keyed_by_deal_id():
    with get_conn() as c, c.cursor() as cur:
        cur.execute("""SELECT 1 FROM information_schema.columns
                        WHERE table_schema='proc' AND table_name='bp_goods_receipt_trgt'
                          AND column_name='deal_id'""")
        assert cur.fetchone() is not None
