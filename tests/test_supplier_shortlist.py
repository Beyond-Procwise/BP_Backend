"""A shortlist must come from what suppliers have actually sold, not from how
well they delivered something else (capability audit 2026-10-08: a laptop
requirement shortlisted 'Harbourline Logistics' on delivery and risk alone)."""
import pytest

from src.services import supplier_shortlist as sl


@pytest.mark.parametrize("title, term", [
    ("Business Laptops (14-inch, 16GB RAM, 512GB SSD, 3-year warranty)", "laptop"),
    ("200 Standard Desktop Computers", "computer"),
    ("Safety Gloves", "glove"),
    ("Office Chair", "chair"),
    ("Cloud Backup Service", "service"),
])
def test_the_product_term_is_the_head_noun_of_the_title(title, term):
    assert sl.product_term({"title": title}) == term


@pytest.mark.parametrize("req", [None, {}, {"title": ""}, {"title": "  "}, {"title": "(only brackets)"},
                                 {"title": "IT"}, "not a dict"])
def test_no_usable_title_means_no_term(req):
    assert sl.product_term(req) is None


class _Cur:
    def __init__(self, rows): self.rows, self.sql, self.args = rows, None, None
    def execute(self, sql, args=None): self.sql, self.args = sql, args
    def fetchall(self): return self.rows
    def __enter__(self): return self
    def __exit__(self, *a): return False


class _Conn:
    def __init__(self, rows): self.cur = _Cur(rows)
    def cursor(self): return self.cur


def test_suppliers_selling_searches_all_three_line_tables_with_a_bound_term():
    conn = _Conn([("S1",), ("S2",), ("S1",), (None,)])
    assert sl.suppliers_selling(conn, "laptop") == {"S1", "S2"}
    assert "bp_po_line_items_trgt" in conn.cur.sql
    assert "bp_invoice_line_items_trgt" in conn.cur.sql
    assert "bp_quote_line_items_trgt" in conn.cur.sql
    # the term is a bound parameter, never spliced into the SQL
    assert "laptop" not in conn.cur.sql
    assert all(a == "%laptop%" for a in conn.cur.args)


def test_a_like_wildcard_in_the_term_is_escaped():
    conn = _Conn([])
    sl.suppliers_selling(conn, "50%_off")
    assert all("\\%" in a and "\\_" in a for a in conn.cur.args)
