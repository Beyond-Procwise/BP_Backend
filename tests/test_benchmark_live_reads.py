"""The pool query's reads, exercised against a fake cursor.

A fake rather than a database: these tests pin which COLUMNS the SQL selects
and how NULLs are treated, which is exactly where the currency defect lived.
"""
from services.benchmark.models import BenchmarkSettings
from services.benchmark_live import _to_points, benchmark_deal, load_benchmark_pool


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


class DispatchingFakeCursor:
    """Answers benchmark_deal's queries by inspecting SQL text (and, for the
    pool query, the bound params) rather than by call order.

    benchmark_deal issues three queries today (quote lines, unscoped pool,
    scoped pool) and a fourth is coming in the next task (suspect_points).
    A fixed-position fake ("1st call is quote lines, 2nd is unscoped, ...")
    would silently start answering the wrong query the moment a call is
    inserted or reordered. Dispatching on content is resilient to that: an
    unrecognised query just gets an empty result instead of derailing every
    other query's answer.

    The pool query's SQL text is IDENTICAL for the scoped and unscoped call
    (only the `%(deal)s` parameter differs), so content alone cannot tell
    them apart — this fake also reads the bound `deal` param to pick the
    right canned rows, and records it so the test can assert the exact
    exclude_deal_id the scoped call actually used.
    """

    QUOTE_COLS = ["quote_line_id", "quote_id", "item_description", "quantity",
                  "unit_price", "unit_of_measure", "currency", "country", "region"]
    POOL_COLS = ["point_id", "item_description", "unit_of_measure", "currency",
                 "unit_price", "quantity", "doc_id"]

    def __init__(self, quote_rows, full_pool_rows, scoped_pool_rows):
        self._quote_rows = quote_rows
        self._full_pool_rows = full_pool_rows
        self._scoped_pool_rows = scoped_pool_rows
        self.scoped_deal_seen = "__not_called__"
        self.description = []
        self._pending: list = []

    def execute(self, sql, params=None):
        low = " ".join(sql.split()).lower()
        if "bp_quote_line_items_trgt" in low:
            self.description = [(c,) for c in self.QUOTE_COLS]
            self._pending = self._quote_rows
        elif "bp_po_line_items_trgt" in low:
            self.description = [(c,) for c in self.POOL_COLS]
            deal = (params or {}).get("deal")
            if deal is None:
                self._pending = self._full_pool_rows
            else:
                self.scoped_deal_seen = deal
                self._pending = self._scoped_pool_rows
        else:
            # Not a query this fake knows about yet (e.g. the next task's
            # suspect_points query) — answer empty rather than crash.
            self.description = []
            self._pending = []

    def fetchall(self):
        return self._pending


def test_benchmark_deal_feeds_the_scoped_pool_to_the_engine():
    """This is the one wiring point no other test exercises: benchmark_deal
    must pass exclude_deal_id=deal_id to the scoped load, and feed THAT pool
    -- not the unscoped one -- to the engine. Swap full_pool/scoped_pool (or
    drop the exclude_deal_id argument) in benchmark_deal and this is the
    test that fails; every other benchmark test still passes."""
    deal_id = "DEAL-1"
    quote_rows = [
        ("QL1", "Q1", "Widget", 10, 100.0, "each", "GBP", "UK", "London"),
    ]
    # Same item/uom/currency as the quote line, so it WOULD match and enter
    # matched_point_ids if the wrong (unscoped) pool reached the engine.
    own_row = ("po:OWN", "Widget", "each", "GBP", 100.0, 5, "PO-OWN")
    other_rows = [
        ("po:1", "Widget", "each", "GBP", 90.0, 5, "PO1"),
        ("po:2", "Widget", "each", "GBP", 95.0, 5, "PO2"),
        ("inv:1", "Widget", "each", "GBP", 92.0, 5, "INV1"),
    ]
    full_pool_rows = [own_row] + other_rows
    scoped_pool_rows = other_rows  # what the DB would return once it excludes deal_id

    cur = DispatchingFakeCursor(quote_rows, full_pool_rows, scoped_pool_rows)
    result = benchmark_deal(cur, deal_id, BenchmarkSettings())

    assert result["own_documents_excluded"] == 1
    assert cur.scoped_deal_seen == deal_id

    matched_ids = result["lines"][0]["matched_point_ids"]
    assert "po:OWN" not in matched_ids
    assert set(matched_ids) == {"po:1", "po:2", "inv:1"}
