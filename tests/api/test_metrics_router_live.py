"""The extraction-metrics endpoints must read tables that actually exist.

`GET /metrics/extraction` and `GET /metrics/extraction/recent` returned 500 for
months against a live database, because their SQL named `proc.bp_invoice`,
`proc.bp_quote` and `proc.bp_purchase_order` — table names that have never
existed in this schema. The pipeline writes `_raw` -> `_stg` -> `_trgt`, and
`_trgt` is the final, deal-keyed destination. Nothing caught it: there was no
test for this router at all, and the output-safety layer turned the
`UndefinedTable` error into a generic message, so the failure looked like a
transient server error rather than a query naming a table that isn't there.

Two tests, doing different jobs:

  * `test_every_table_the_metrics_sql_names_exists` reads the SQL out of the
    module and checks each `proc.<table>` against the live catalog. This is the
    one that would have caught the original bug, and it catches the whole class
    — a future edit naming a table that does not exist fails here, by name,
    rather than surfacing as a 500 in production.
  * the endpoint tests call the route functions against the live database and
    assert the shape they return. A query can reference real tables and still
    be wrong; these prove the endpoints actually answer.

Live-only. Run with:
    set -a && . ./.env && set +a
    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/api/test_metrics_router_live.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


def _live_tables() -> set[str]:
    """Every table and view in the `proc` schema of the configured database."""
    from services.db import get_conn

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'proc'"
            )
            return {r[0] for r in cur.fetchall()}


def _tables_named_in_metrics_sql() -> set[str]:
    """Every `proc.<table>` the metrics router's SQL refers to.

    Only string literals that look like SQL are scanned. Scanning the whole
    file would sweep up prose from the module docstring -- `proc.bp_*` there
    parses as a table called `bp_`, which no schema will ever contain, and the
    test would fail for a reason that has nothing to do with the query.
    """
    import ast

    source = (_ROOT / "src" / "api" / "routers" / "metrics.py").read_text()
    named: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
            if "FROM" in text.upper() and "proc." in text:
                named.update(re.findall(r"\bproc\.([a-z_][a-z0-9_]*)", text))
    return named


def test_every_table_the_metrics_sql_names_exists():
    named = _tables_named_in_metrics_sql()
    assert named, "parsed no table names out of metrics.py — the regex is wrong"

    missing = sorted(named - _live_tables())
    assert not missing, (
        f"metrics.py queries {len(missing)} table(s) that do not exist: {missing}. "
        "The pipeline writes _raw/_stg/_trgt; _trgt is the final destination."
    )


def test_extraction_quality_summary_answers():
    from api.routers.metrics import extraction_quality_summary

    body = extraction_quality_summary()

    assert set(body) == {"totals", "vendors"}
    assert set(body["totals"]) >= {
        "processed", "with_lines", "zero_lines_with_total",
        "line_capture_rate", "zero_lines_rate",
    }
    assert isinstance(body["vendors"], list)
    # This corpus has invoices, quotes and POs in _trgt, so the endpoint must
    # find some. A zero here means it is reading the wrong tables again.
    assert body["totals"]["processed"] > 0, "no documents aggregated from _trgt"
    for vendor in body["vendors"]:
        assert set(vendor) >= {"supplier_id", "doc_type", "processed", "last_seen_at"}
        assert vendor["doc_type"] in {"Invoice", "Quote", "Purchase_Order"}
        assert 0.0 <= vendor["zero_lines_rate"] <= 1.0
        assert 0.0 <= vendor["line_capture_rate"] <= 1.0


def test_extraction_quality_summary_answers_in_reasonable_time():
    """An operator dashboard that takes minutes is still broken.

    The first version of this query counted line items with a correlated
    subquery per document. None of `bp_invoice_line_items_trgt`,
    `bp_quote_line_items_trgt` or `bp_po_line_items_trgt` carries a single
    index, so each of the ~38.5k parent rows sequentially scanned a 55k–116k
    row table: EXPLAIN put the plan at cost 161,402,845 and it had not
    returned after two minutes.

    The budget below is deliberately generous. It is not a benchmark — it is a
    guard against reintroducing a plan that is quadratic in the corpus, which
    is invisible on an empty test database and only bites on real data.
    """
    import time

    from api.routers.metrics import extraction_quality_summary

    started = time.monotonic()
    body = extraction_quality_summary()
    elapsed = time.monotonic() - started

    assert body["totals"]["processed"] > 0
    assert elapsed < 30.0, (
        f"summary took {elapsed:.1f}s. Check EXPLAIN for a per-row SubPlan "
        "over the *_line_items_trgt tables — they have no indexes."
    )


def test_recent_extractions_answers_and_honours_limit():
    from api.routers.metrics import recent_extractions

    body = recent_extractions(limit=5)

    assert set(body) == {"records"}
    assert 0 < len(body["records"]) <= 5
    for record in body["records"]:
        assert set(record) >= {
            "doc_type", "pk", "supplier_id", "total", "lines", "needs_review", "seen_at",
        }
        assert record["doc_type"] in {"Invoice", "Quote", "Purchase_Order"}
        assert record["needs_review"] is (record["total"] > 0 and record["lines"] == 0)
