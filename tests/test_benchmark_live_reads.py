"""The pool query's reads, exercised against a fake cursor.

A fake rather than a database: these tests pin which COLUMNS the SQL selects
and how NULLs are treated, which is exactly where the currency defect lived.
"""
from services.benchmark_live import _to_points, load_benchmark_pool


class FakeCursor:
    """Records the SQL it was given and replays a canned result."""

    def __init__(self, rows, description):
        self._rows = rows
        self.description = description
        self.sql = ""

    def execute(self, sql, params=None):
        self.sql = sql

    def fetchall(self):
        return self._rows


def test_pool_query_joins_the_purchase_order_header_for_currency():
    cur = FakeCursor([], [("point_id",), ("item_description",),
                          ("unit_of_measure",), ("currency",), ("unit_price",),
                          ("quantity",), ("doc_id",)])
    load_benchmark_pool(cur)
    sql = " ".join(cur.sql.split()).lower()
    assert "bp_purchase_order_trgt" in sql, "PO header is never joined"
    assert "coalesce(p.currency, h.currency)" in sql


def test_header_currency_survives_into_the_point():
    rows = [{"point_id": "po:1", "item_description": "Widget",
             "unit_of_measure": "each", "currency": "USD",
             "unit_price": 10.0, "quantity": 2, "doc_id": "PO1"}]
    assert _to_points(rows)[0].currency == "USD"


def test_pool_query_excludes_the_deal_under_analysis():
    cur = FakeCursor([], [("point_id",), ("item_description",),
                          ("unit_of_measure",), ("currency",), ("unit_price",),
                          ("quantity",), ("doc_id",)])
    load_benchmark_pool(cur, exclude_deal_id="DEAL-1")
    sql = " ".join(cur.sql.split()).lower()
    assert "p.deal_id is distinct from" in sql
    assert "i.deal_id is distinct from" in sql


def test_excluded_own_document_count_is_reported():
    """Read the two pool sizes and report the difference, so a line that gates
    because its own documents were removed can be explained."""
    from services.benchmark_live import _pool_delta
    assert _pool_delta(120, 100) == 20
