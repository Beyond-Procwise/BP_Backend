"""The knowledge graph's source tables must exist, with the primary keys it names.

ENTITY_TABLE_MAP named six tables that had been dropped when the extraction
pipeline moved to _stg -> _trgt: proc.bp_invoice, bp_invoice_line_items,
bp_purchase_order, bp_po_line_items, bp_quote, bp_quote_line_items. Every
document loader read a table that did not exist, caught the error, logged it at
DEBUG, and returned 0. The scheduled job then logged "KG sync completed" with
every document count at zero.

The graph stopped gaining documents on 2026-07-31 and nobody noticed for eleven
days, while the Ask surface and AgentNick's describe_platform tool kept reading
it. Two of the non-document entries were wrong in the same silent way:
bp_category has no category_id column at all, and bp_policy's key is policy_id
rather than id — so 19 live policy rows never loaded.

A map of table names is exactly the kind of thing that rots when tables are
renamed, and nothing in the type system can catch it. This is what catches it.

The schema checks need a live database and skip without one. The structural
checks below do not, and run everywhere.
"""
from __future__ import annotations

import os

import pytest

from src.services.procurement_kg_builder import ENTITY_TABLE_MAP


# --------------------------------------------------------------------------
# Structural — no database needed
# --------------------------------------------------------------------------

def test_the_map_is_not_empty():
    """Guards the guard: an empty map would make every check below vacuous."""
    assert len(ENTITY_TABLE_MAP) >= 8


def test_no_document_entry_points_at_a_pre_renovation_table():
    """The dropped names, pinned by name so a revert is loud.

    Each of these was a real entry in this map and each returned 0 for eleven
    days. A future edit that reinstates one — by copy-paste from an old branch,
    or by 'simplifying' the _trgt suffix away — fails here.
    """
    dropped = {
        "proc.bp_invoice", "proc.bp_invoice_line_items",
        "proc.bp_purchase_order", "proc.bp_po_line_items",
        "proc.bp_quote", "proc.bp_quote_line_items",
        "proc.bp_approvals",
    }
    offenders = {
        entity: table
        for entity, (table, _pk, _label) in ENTITY_TABLE_MAP.items()
        if table in dropped
    }
    assert not offenders, (
        f"these entries name tables that do not exist: {offenders}. "
        f"The document tiers are proc.bp_*_trgt."
    )


@pytest.mark.parametrize("entity", [
    "Invoice", "InvoiceLine", "PurchaseOrder", "POLine", "Quote", "QuoteLine",
])
def test_every_document_entity_reads_the_trgt_tier(entity):
    """_stg is the staging tier and _trgt is what the product reports from.
    A graph built from _stg would disagree with every screen."""
    table = ENTITY_TABLE_MAP[entity][0]
    assert table.endswith("_trgt"), (
        f"{entity} reads {table}; document nodes must come from the _trgt tier"
    )


# --------------------------------------------------------------------------
# Against the live schema
# --------------------------------------------------------------------------

def _real_connection():
    """A genuine psycopg2 connection, NOT services.db.get_conn.

    get_conn substitutes a fake connection when PYTEST_CURRENT_TEST is set
    (src/services/db.py:1120). That is right for the suites that exercise logic
    without a database — and fatal here, because this test's entire purpose is
    to compare a hardcoded map against the REAL schema. Asking a stub whether a
    table exists proves nothing at all.
    """
    if not os.getenv("DB_HOST"):
        pytest.skip("no database configured (DB_HOST unset)")
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        return psycopg2.connect(
            host=os.environ["DB_HOST"], dbname=os.environ["DB_NAME"],
            user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"],
            port=os.environ.get("DB_PORT", "5432"), connect_timeout=5,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"database unreachable: {type(exc).__name__}")


@pytest.fixture(scope="module")
def schema_columns():
    """{table_name: {column, ...}} for the proc schema, read once."""
    conn = _real_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'proc'"
        )
        out: dict[str, set[str]] = {}
        for table, column in cur.fetchall():
            out.setdefault(table, set()).add(column)
    finally:
        conn.close()

    # Guard the guard: an empty read would make every assertion below pass.
    assert len(out) > 50, (
        f"only {len(out)} proc tables visible — the connection is not seeing "
        f"the real schema, so these checks would be vacuous"
    )
    return out


@pytest.mark.parametrize("entity", sorted(ENTITY_TABLE_MAP))
def test_each_source_table_exists_with_the_primary_key_named(
    entity, schema_columns
):
    table, pk, _label = ENTITY_TABLE_MAP[entity]
    schema, _, name = table.partition(".")
    assert schema == "proc", f"{entity} reads {table}, outside the proc schema"
    columns = schema_columns.get(name, set())

    assert columns, (
        f"{entity} reads {table}, which does not exist. Every node of this type "
        f"would silently fail to load."
    )
    assert pk in columns, (
        f"{entity} reads {table} keyed on {pk!r}, which is not a column there. "
        f"Rows load with a null key and are skipped, silently. "
        f"Columns: {sorted(columns)[:12]}"
    )
