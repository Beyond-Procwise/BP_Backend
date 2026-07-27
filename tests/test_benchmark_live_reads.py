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

    def fetchone(self):
        return self._rows[0] if self._rows else None


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


def test_flagged_documents_query_reads_open_findings_only():
    cur = FakeCursor(
        [("purchase_order", "PO1"), ("invoice", "INV2")],
        [("doc_type",), ("doc_pk_candidate",)],
    )
    from services.benchmark_live import load_flagged_documents
    assert load_flagged_documents(cur) == {"po:PO1", "inv:INV2"}
    sql = " ".join(cur.sql.split()).lower()
    assert "bp_extraction_discrepancy" in sql
    assert "status = 'open'" in sql


def test_flagged_documents_are_type_qualified_so_a_shared_raw_id_cannot_collide():
    """Live bp_sqldb has a PO and an invoice both numbered '123456/22' --
    po_id and invoice_id are not disjoint namespaces. An open finding on the
    PO must not make the invoice sharing that raw id look flagged too."""
    cur = FakeCursor(
        [("purchase_order", "123456/22")],
        [("doc_type",), ("doc_pk_candidate",)],
    )
    from services.benchmark_live import load_flagged_documents
    flagged = load_flagged_documents(cur)
    assert flagged == {"po:123456/22"}
    assert "inv:123456/22" not in flagged


def test_suspect_points_counts_matched_points_from_flagged_documents():
    from services.benchmark_live import _count_suspect
    doc_by_point = {"po:1": "po:PO1", "po:2": "po:PO2", "inv:3": "inv:INV3"}
    assert _count_suspect(["po:1", "inv:3"], doc_by_point, {"po:PO1"}) == 1
    assert _count_suspect(["po:2"], doc_by_point, {"po:PO1"}) == 0


def test_suspect_points_does_not_conflate_a_po_and_invoice_sharing_an_id():
    """The same PO/invoice id collision, exercised through _count_suspect:
    a matched PO point must not be counted as suspect from an open finding
    against the invoice of the same raw id, and vice versa."""
    from services.benchmark_live import _count_suspect
    doc_by_point = {"po:1": "po:123456/22", "inv:2": "inv:123456/22"}
    flagged = {"inv:123456/22"}  # only the invoice has an open finding
    assert _count_suspect(["po:1"], doc_by_point, flagged) == 0
    assert _count_suspect(["inv:2"], doc_by_point, flagged) == 1


class DispatchingFakeCursor:
    """Answers benchmark_deal's queries by inspecting SQL text (and, for the
    pool query, the bound params) rather than by call order.

    benchmark_deal issues four queries (quote lines, scoped pool, unscoped
    pool COUNT, open-findings documents). A fixed-position fake ("1st call is
    quote lines, 2nd is the pool, ...") would silently start answering the
    wrong query the moment a call is inserted or reordered. Dispatching on
    content is resilient to that: an unrecognised query just gets an empty
    result instead of derailing every other query's answer.

    The scoped pool query and the unscoped COUNT query both touch
    bp_po_line_items_trgt, so the "count(*)" text (only present in the count
    query — see count_benchmark_pool) is checked first to tell them apart;
    this fake also reads the bound `deal` param on the row-returning pool
    query and records it so the test can assert the exact exclude_deal_id
    the scoped call actually used.
    """

    QUOTE_COLS = ["quote_line_id", "quote_id", "item_description", "quantity",
                  "unit_price", "unit_of_measure", "currency", "country", "region"]
    POOL_COLS = ["point_id", "item_description", "unit_of_measure", "currency",
                 "unit_price", "quantity", "doc_id"]

    def __init__(self, quote_rows, full_pool_count, scoped_pool_rows, flagged_rows=()):
        self._quote_rows = quote_rows
        self._full_pool_count = full_pool_count
        self._scoped_pool_rows = scoped_pool_rows
        self._flagged_rows = flagged_rows
        self.scoped_deal_seen = "__not_called__"
        self.description = []
        self._pending: list = []
        self._scalar = None

    def execute(self, sql, params=None):
        low = " ".join(sql.split()).lower()
        if "bp_quote_line_items_trgt" in low:
            self.description = [(c,) for c in self.QUOTE_COLS]
            self._pending = self._quote_rows
        elif "count(*)" in low:
            self._scalar = self._full_pool_count
        elif "bp_po_line_items_trgt" in low:
            self.description = [(c,) for c in self.POOL_COLS]
            self.scoped_deal_seen = (params or {}).get("deal")
            self._pending = self._scoped_pool_rows
        elif "bp_extraction_discrepancy" in low:
            self.description = [("doc_type",), ("doc_pk_candidate",)]
            self._pending = self._flagged_rows
        else:
            # Not a query this fake knows about yet — answer empty rather
            # than crash, so an unrelated future query can't derail this one.
            self.description = []
            self._pending = []

    def fetchall(self):
        return self._pending

    def fetchone(self):
        return (self._scalar,)


def test_benchmark_deal_feeds_the_scoped_pool_to_the_engine():
    """This is the one wiring point no other test exercises: benchmark_deal
    must pass exclude_deal_id=deal_id to the scoped load, and feed THAT pool
    -- not the unscoped count -- to the engine. Swap full_count/scoped_pool
    (or drop the exclude_deal_id argument) in benchmark_deal and this is the
    test that fails; every other benchmark test still passes."""
    deal_id = "DEAL-1"
    quote_rows = [
        ("QL1", "Q1", "Widget", 10, 100.0, "each", "GBP", "UK", "London"),
    ]
    # Same item/uom/currency as the quote line, so it WOULD match and enter
    # matched_point_ids if the wrong (unscoped) pool reached the engine.
    other_rows = [
        ("po:1", "Widget", "each", "GBP", 90.0, 5, "PO1"),
        ("po:2", "Widget", "each", "GBP", 95.0, 5, "PO2"),
        ("inv:1", "Widget", "each", "GBP", 92.0, 5, "INV1"),
    ]
    full_pool_count = len(other_rows) + 1  # + the deal's own excluded document
    scoped_pool_rows = other_rows  # what the DB would return once it excludes deal_id

    cur = DispatchingFakeCursor(quote_rows, full_pool_count, scoped_pool_rows)
    result = benchmark_deal(cur, deal_id, BenchmarkSettings())

    assert result["own_points_excluded"] == 1
    assert cur.scoped_deal_seen == deal_id

    matched_ids = result["lines"][0]["matched_point_ids"]
    assert "po:OWN" not in matched_ids
    assert set(matched_ids) == {"po:1", "po:2", "inv:1"}


def test_benchmark_deal_reports_suspect_points_per_line():
    """A line's matched pool ids that trace back to a flagged document (PO1
    carries an open discrepancy finding) are counted and disclosed alongside
    the benchmark, without being dropped from the comparison itself."""
    deal_id = "DEAL-1"
    quote_rows = [
        ("QL1", "Q1", "Widget", 10, 100.0, "each", "GBP", "UK", "London"),
    ]
    scoped_pool_rows = [
        ("po:1", "Widget", "each", "GBP", 90.0, 5, "PO1"),
        ("po:2", "Widget", "each", "GBP", 95.0, 5, "PO2"),
        ("inv:1", "Widget", "each", "GBP", 92.0, 5, "INV1"),
    ]
    cur = DispatchingFakeCursor(
        quote_rows, len(scoped_pool_rows), scoped_pool_rows,
        flagged_rows=[("purchase_order", "PO1")],
    )
    result = benchmark_deal(cur, deal_id, BenchmarkSettings())

    matched_ids = result["lines"][0]["matched_point_ids"]
    assert set(matched_ids) == {"po:1", "po:2", "inv:1"}
    # Only po:1 maps back to PO1, the flagged document -> exactly one suspect point.
    assert result["lines"][0]["suspect_points"] == 1
    assert any("data-quality finding" in d for d in result["disclosures"])


def test_count_pool_query_shares_the_load_query_s_predicates():
    """count_benchmark_pool must count the same rows load_benchmark_pool
    would return -- same NULL-price/description filters, same deal exclusion
    -- so 'own_points_excluded' stays derived from two views of one pool
    rather than two different queries that can silently drift apart."""
    from services.benchmark_live import count_benchmark_pool

    cur = FakeCursor([(0,)], [("count",)])
    count_benchmark_pool(cur, exclude_deal_id="DEAL-1")
    sql = " ".join(cur.sql.split()).lower()
    assert "count(*)" in sql
    assert "p.unit_price is not null and p.item_description is not null" in sql
    assert "i.unit_price is not null and i.item_description is not null" in sql
    assert "p.deal_id is distinct from" in sql
    assert "i.deal_id is distinct from" in sql


def test_count_pool_query_returns_a_scalar():
    from services.benchmark_live import count_benchmark_pool

    cur = FakeCursor([(42,)], [("count",)])
    assert count_benchmark_pool(cur) == 42


def test_quote_line_with_no_quantity_is_skipped_not_priced_at_zero():
    """The identical defect already fixed on the history side (a NULL
    quantity silently becoming 0.0): a quote line with no recorded quantity
    has no meaningful total or gap, so defaulting it to zero used to make
    quoted_total/benchmark_total collapse to just the adders and emit
    total_cost_gap == 0.0 -- a confident 'no gap' fabricated from absent
    data. It must be skipped, and the skip counted and disclosed."""
    deal_id = "DEAL-1"
    quote_rows = [
        ("QL1", "Q1", "Widget", 10, 100.0, "each", "GBP", "UK", "London"),
        ("QL2", "Q1", "Consulting", None, 200.0, "hour", "GBP", "UK", "London"),
    ]
    scoped_pool_rows = [
        ("po:1", "Widget", "each", "GBP", 90.0, 5, "PO1"),
        ("po:2", "Widget", "each", "GBP", 95.0, 5, "PO2"),
        ("inv:1", "Widget", "each", "GBP", 92.0, 5, "INV1"),
    ]
    cur = DispatchingFakeCursor(quote_rows, len(scoped_pool_rows), scoped_pool_rows)
    result = benchmark_deal(cur, deal_id, BenchmarkSettings())

    # Only QL1 (has a quantity) is benchmarked; QL2 (no quantity) is skipped
    # entirely rather than entering the results with a fabricated total_cost_gap.
    assert result["line_count"] == 1
    assert result["lines"][0]["quantity"] == 10.0
    assert result["lines"][0]["source_item_description"] == "Widget"
    assert not any(
        line["source_item_description"] == "Consulting" for line in result["lines"])
    assert result["skipped_no_quantity_count"] == 1
    assert any("skipped" in d for d in result["disclosures"])
