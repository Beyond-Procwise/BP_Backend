"""Every physical pipeline must be wired at every map that routes on it.

A pipeline is named in nine places -- `validate._PIPELINES`, four maps in
`persistence`, five in `promotion` -- and a document only reaches `_stg` if ALL
of them know it. A missing entry is a `KeyError` or a silent skip in the middle
of a live ingestion, which is exactly how the goods receipt's wiring was found
to be half-done.

So this does not restate a list. It takes `_PIPELINES` as the authority, asks
each map whether it knows every pipeline, and then asks the DATABASE whether
every table those maps name actually exists. Live-only for the second half.
"""
from __future__ import annotations

import os

import pytest

from src.services.concepts.routing import pipeline_for_category
from src.services.concepts.validate import _PIPELINES
from src.services.extraction import persistence, promotion

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")

_PERSISTENCE_MAPS = ("_RAW_TABLES", "_LINE_RAW_TABLES", "_LINE_RAW_INDEX_COL", "_DOC_PK_FIELD")
_PROMOTION_MAPS = ("_RAW_TO_STG", "_LINE_RAW_TO_STG", "_LINE_STG_PK", "_LINE_STG_INDEX", "_STG_PK")


@pytest.mark.parametrize("map_name", _PERSISTENCE_MAPS)
def test_persistence_knows_every_pipeline(map_name):
    mapping = getattr(persistence, map_name)
    missing = sorted(_PIPELINES - set(mapping))
    assert not missing, f"persistence.{map_name} does not know: {missing}"


@pytest.mark.parametrize("map_name", _PROMOTION_MAPS)
def test_promotion_knows_every_pipeline(map_name):
    mapping = getattr(promotion, map_name)
    missing = sorted(_PIPELINES - set(mapping))
    assert not missing, f"promotion.{map_name} does not know: {missing}"


def test_a_goods_receipt_category_routes_at_the_goods_receipt_pipeline():
    """The uploader types a word; the vocabulary turns it into a pipeline. All
    eleven spellings land on the same one -- that is the whole point of seeding
    them."""
    for spelling in ("goods receipt", "GRN", "Delivery Note", "packing slip",
                     "proof of delivery", "despatch note"):
        pipeline, concept = pipeline_for_category(spelling)
        assert (pipeline, concept) == ("goods_receipt", "doctype.goods_receipt"), spelling


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_every_table_the_maps_name_exists():
    from src.services.db import get_conn

    named = set()
    for m in (persistence._RAW_TABLES, persistence._LINE_RAW_TABLES):
        named.update(m.values())
    for m in (promotion._RAW_TO_STG, promotion._LINE_RAW_TO_STG):
        for pair in m.values():
            named.update(pair)

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT table_schema || '.' || table_name
                 FROM information_schema.tables WHERE table_schema = 'proc'""")
        present = {r[0] for r in cur.fetchall()}
    missing = sorted(named - present)
    assert not missing, f"tables named by the routing maps that do not exist: {missing}"


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
@pytest.mark.parametrize("pipeline", sorted(_PIPELINES))
def test_every_stg_key_column_is_unique(pipeline):
    """promote() upserts with ON CONFLICT on this column. Without a unique
    index the statement raises mid-promotion, and re-extracting a document that
    already exists is the ordinary case, not an edge one."""
    from src.services.db import get_conn

    _raw, stg = promotion._RAW_TO_STG[pipeline]
    schema, table = stg.split(".", 1)
    col = promotion._STG_PK[pipeline]
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT 1 FROM pg_index i
                 JOIN pg_class t ON t.oid = i.indrelid
                 JOIN pg_namespace n ON n.oid = t.relnamespace
                WHERE n.nspname = %s AND t.relname = %s AND i.indisunique
                  AND (SELECT array_agg(a.attname ORDER BY a.attnum)
                         FROM pg_attribute a
                        WHERE a.attrelid = t.oid
                          AND a.attnum = ANY(i.indkey)) = ARRAY[%s]::name[]""",
            (schema, table, col),
        )
        assert cur.fetchone() is not None, (
            f"{stg} has no unique index on {col}, so promote()'s ON CONFLICT fails"
        )


@pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")
def test_a_goods_receipt_travels_raw_to_stg_through_the_ordinary_writer():
    """The claim the design rests on -- 'the tables mirror the invoice family
    precisely so the existing writer applies' -- stated as a test rather than a
    comment. Nothing goods-receipt-specific is called: persistence.write_raw,
    persistence.write_line_items_raw and promotion.promote are the same
    functions every other document type goes through.
    """
    from uuid import uuid4

    from src.services.db import get_conn
    from src.services.extraction.persistence import write_line_items_raw, write_raw

    grn = f"GRN-TEST-{uuid4().hex[:8].upper()}"
    raw_id = None
    try:
        raw_id = write_raw(
            doc_type="goods_receipt",
            file_path=f"s3://test/{grn}.pdf",
            process_monitor_id=None,
            trace_id=uuid4(),
            pipeline_version="test",
            columns={"grn_id": grn, "po_id": "4500018832",
                     "supplier_name": "Northwind Trading Ltd"},
            parser_snapshot={},
            promotion_status="pending",
        )
        assert write_line_items_raw(
            doc_type="goods_receipt", raw_id=raw_id,
            # No line_no here: write_line_items_raw stamps the index column
            # itself from the list position, and passing it too is a duplicate
            # column in the INSERT.
            line_items=[{"item_description": "Widget A",
                         "quantity_received": 10, "unit_of_measure": "each",
                         "po_line_ref": "10"}],
        ) == 1

        result = promotion.promote(raw_id, "goods_receipt")
        assert result.get("ok"), result

        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("SELECT po_id, supplier_name FROM proc.bp_goods_receipt_stg "
                        "WHERE grn_id = %s", (grn,))
            assert cur.fetchone() == ("4500018832", "Northwind Trading Ltd")
            cur.execute("SELECT goods_receipt_line_id, quantity_received, unit_of_measure "
                        "FROM proc.bp_goods_receipt_line_items_stg WHERE grn_id = %s", (grn,))
            assert cur.fetchall() == [(f"{grn}-L1", 10, "each")]
    finally:
        with get_conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_stg WHERE grn_id = %s", (grn,))
            cur.execute("DELETE FROM proc.bp_goods_receipt_stg WHERE grn_id = %s", (grn,))
            if raw_id is not None:
                cur.execute("DELETE FROM proc.bp_goods_receipt_line_items_raw WHERE raw_id = %s", (raw_id,))
                cur.execute("DELETE FROM proc.bp_goods_receipt_raw WHERE raw_id = %s", (raw_id,))
