from services.price_outlier.detector import (
    Finding, _resolved_currency, build_peer_index, describe, find_outliers,
    peers_for,
)
from services.price_outlier.rule import OutlierSettings, assess


def _row(item, uom, ccy, price, doc="PO1", kind="po"):
    return {"point_id": f"{kind}:{price}", "item_description": item,
            "unit_of_measure": uom, "currency": ccy, "unit_price": price,
            "quantity": 1, "doc_id": doc}


def test_peer_index_groups_on_the_same_key_the_engine_matches_on():
    rows = [_row("  Widget  ", "Each", "gbp", 100.0, doc="PO1"),
            _row("widget", "each", "GBP", 102.0, doc="PO2"),
            _row("Widget", "box", "GBP", 500.0, doc="PO3")]
    index = build_peer_index(rows)
    assert index[("widget", "each", "GBP")] == [("po:PO1", 100.0), ("po:PO2", 102.0)]
    assert index[("widget", "box", "GBP")] == [("po:PO3", 500.0)]


def test_peers_exclude_the_line_s_own_document():
    rows = [_row("Widget", "each", "GBP", 100.0, doc="PO1"),
            _row("Widget", "each", "GBP", 900.0, doc="PO2")]
    index = build_peer_index(rows)
    assert peers_for(index, ("widget", "each", "GBP"), own_doc="po:PO1") == [900.0]


def test_peers_for_an_unknown_key_is_empty_not_an_error():
    assert peers_for({}, ("nothing", "each", "GBP"), own_doc=None) == []


def test_peers_from_a_different_document_type_sharing_the_same_raw_id_survive():
    """Live bp_sqldb has a purchase order and an invoice both numbered
    '123456/22' -- po_id and invoice_id are not disjoint namespaces. Excluding
    by raw doc id would wrongly drop the invoice's price from the PO's peer
    set (and vice versa), even though they are two different documents and
    the invoice's price is legitimate comparison evidence for the PO line."""
    rows = [_row("Widget", "each", "GBP", 100.0, doc="123456/22", kind="po"),
            _row("Widget", "each", "GBP", 900.0, doc="123456/22", kind="inv")]
    index = build_peer_index(rows)
    peers = peers_for(index, ("widget", "each", "GBP"), own_doc="po:123456/22")
    assert peers == [900.0], "the invoice's price must remain a valid peer"


def test_note_names_the_comparison_in_plain_english():
    verdict = assess(11.69, [1.17] * 18, OutlierSettings())
    note = describe(3, "A4 Ruled Notebook", 11.69, verdict, uom="each", currency="GBP")
    assert "line 3" in note
    assert "A4 Ruled Notebook" in note
    assert "11.69" in note
    assert "1.17" in note
    assert "18 comparable" in note
    assert "check the unit price and quantity" in note
    assert "ABOVE" in note


def test_note_reads_as_english_when_the_price_is_below_the_usual():
    """The direction word must be spelled out both ways -- a bare '÷' (or a
    '/', which the downstream output-safety gate rewrites as a URL-like
    route) is not a sentence a reviewer can act on."""
    verdict = assess(0.02, [2.0] * 7, OutlierSettings())
    note = describe(3, "Coffee Filters", 0.02, verdict, uom="each", currency="GBP")
    assert "BELOW" in note
    assert "ABOVE" not in note
    assert "/" not in note
    assert "÷" not in note
    assert "100.0" in note


def test_note_includes_the_real_unit_and_currency_not_a_hardcoded_each():
    """A three-currency corpus with an hourly rate must not read as
    '2.00 each' -- that silently drops both the real unit and the currency
    the price is actually being compared in. This string is the entire
    human-facing product of the feature."""
    verdict = assess(150.0, [50.0] * 6, OutlierSettings())
    note = describe(7, "Consulting", 150.0, verdict, uom="hour", currency="EUR")
    assert "EUR" in note
    assert "hour" in note
    assert "each" not in note


def test_line_currency_wins_over_header_currency():
    """Matches load_benchmark_pool/load_quote_lines' COALESCE(line, header):
    when a line carries its own currency, it must win over the document
    header's, even when the two disagree."""
    line = {"doc_pk": "PO1", "currency": "USD"}
    assert _resolved_currency(line, {"PO1": "GBP"}) == "USD"


def test_missing_line_currency_falls_back_to_the_header():
    """Invoice lines (and any quote/PO line left blank) have no currency of
    their own; the header is the only source then."""
    line = {"doc_pk": "PO1", "currency": None}
    assert _resolved_currency(line, {"PO1": "GBP"}) == "GBP"


class _FakeCursor:
    """Answers find_outliers' queries by inspecting SQL text for the table it
    names, mirroring DispatchingFakeCursor in test_benchmark_live_reads.py.

    find_outliers issues 7 queries against one cursor (1 pool query, then a
    header query + a line-items query per of the 3 source tables). Dispatching
    on content rather than call order means an unrelated or reordered query
    can't silently derail the answer to a different one.
    """

    _POOL_COLS = ["point_id", "item_description", "unit_of_measure", "currency",
                  "unit_price", "quantity", "doc_id"]
    _LINE_COLS = ["doc_pk", "line_number", "item_description",
                  "unit_of_measure", "unit_price", "currency"]

    def __init__(self, pool_rows, po_lines=(), invoice_lines=(),
                 po_currency=(), invoice_currency=()):
        self._pool_rows = pool_rows
        self._po_lines = po_lines
        self._invoice_lines = invoice_lines
        self._po_currency = po_currency
        self._invoice_currency = invoice_currency
        self.po_lines_sql = ""
        self.invoice_lines_sql = ""
        self.description = []
        self._pending: list = []

    def execute(self, sql, params=None):
        norm = " ".join(sql.split()).lower()
        if "union all" in norm:
            self.description = [(c,) for c in self._POOL_COLS]
            self._pending = self._pool_rows
        elif "bp_purchase_order_trgt" in norm:
            self.description = [("po_id",), ("currency",)]
            self._pending = self._po_currency
        elif "bp_invoice_trgt" in norm and "line_items" not in norm:
            self.description = [("invoice_id",), ("currency",)]
            self._pending = self._invoice_currency
        elif "bp_quote_trgt" in norm:
            self.description = [("quote_id",), ("currency",)]
            self._pending = []
        elif "bp_po_line_items_trgt" in norm:
            self.po_lines_sql = norm
            self.description = [(c,) for c in self._LINE_COLS]
            self._pending = self._po_lines
        elif "bp_invoice_line_items_trgt" in norm:
            self.invoice_lines_sql = norm
            self.description = [(c,) for c in self._LINE_COLS]
            self._pending = self._invoice_lines
        elif "bp_quote_line_items_trgt" in norm:
            self.description = [(c,) for c in self._LINE_COLS]
            self._pending = []
        else:
            self.description = []
            self._pending = []

    def fetchall(self):
        return self._pending


def _pool_tuple(item, uom, ccy, price, doc, kind):
    """A real cursor's fetchall() yields tuples in SELECT order, not dicts --
    unlike the `_row` helper above, which build_peer_index consumes directly."""
    return (f"{kind}:{doc}:{price}", item, uom, ccy, price, 1, doc)


def test_find_outliers_end_to_end_over_a_fake_cursor():
    """Exercises find_outliers itself, not just its helpers: source iteration
    over two of the three tables (so the invoice line_no vs PO line_number
    column difference is exercised), the header-currency substitution for a
    document whose header currency is NULL, and Finding construction."""
    pool_rows = (
        [_pool_tuple("Widget", "each", "GBP", 10.0, doc=f"POX{i}", kind="po")
         for i in range(1, 6)]
        + [_pool_tuple("Gadget", "each", "GBP", 50.0, doc="INVX1", kind="inv")]
    )
    cur = _FakeCursor(
        pool_rows=pool_rows,
        # Tuples in SELECT order (doc_pk, line_number, item_description,
        # unit_of_measure, unit_price, currency), as a real cursor.fetchall()
        # returns. Neither line here carries its own currency (None), so both
        # fall back to the header.
        po_lines=[("PO1", 1, "Widget", "each", 1000.0, None)],
        invoice_lines=[("INV1", 1, "Gadget", "each", 55.0, None)],
        # NULL header currency on PO1 -- proves the default ("GBP") the
        # header substitution falls back to, not a coincidental match.
        po_currency=[("PO1", None)],
        invoice_currency=[("INV1", "GBP")],
    )

    findings = find_outliers(cur, OutlierSettings())

    assert len(findings) == 1, "the gadget line has only 1 peer (< min_peers) and must not flag"
    finding = findings[0]
    assert finding.doc_type == "purchase_order"
    assert finding.doc_pk == "PO1"
    assert finding.line_number == 1
    assert finding.verdict.severity == "critical"
    assert finding.verdict.peer_count == 5
    assert "ABOVE" in finding.note
    assert "GBP" in finding.note
    assert "each" in finding.note

    # The two source tables name their line-number column differently; both
    # queries must actually have been built and executed.
    assert "line_no as line_number" in cur.invoice_lines_sql
    assert "line_no as line_number" not in cur.po_lines_sql


def test_find_outliers_prefers_the_line_s_own_currency_over_its_header():
    """End-to-end version of test_line_currency_wins_over_header_currency:
    a PO line carrying its own currency (USD) must be compared against USD
    peers, even though its document header says GBP -- if the header won,
    this line would find zero peers in its (wrong) currency and merely gate
    on too few peers instead of flagging."""
    pool_rows = [
        _pool_tuple("Widget", "each", "USD", 10.0, doc=f"POX{i}", kind="po")
        for i in range(1, 6)
    ]
    cur = _FakeCursor(
        pool_rows=pool_rows,
        po_lines=[("PO1", 1, "Widget", "each", 1000.0, "USD")],
        po_currency=[("PO1", "GBP")],  # header disagrees with the line
    )

    findings = find_outliers(cur, OutlierSettings())

    assert len(findings) == 1
    assert findings[0].verdict.peer_count == 5
    assert "USD" in findings[0].note
