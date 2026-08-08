"""Canonical master data must stay reachable, and stay read-only.

uicanvas is authoritative for reference data. BP_Backend reaches it through
postgres_fdw rather than a copy, so there is exactly one place the canonical
rows live and no question of which side is right when two copies disagree.

These tests exist because the failure mode is silent: if the foreign server or
the views disappear, nothing raises at import time — queries simply start
returning "table does not exist" deep inside whatever was using them, or worse,
a local empty table with the same-ish name gets used instead.

Live-only. Run with:
    set -a && . ./.env && set +a
    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/services/test_canonical_masters.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")


@pytest.fixture()
def cur():
    from src.services.db import get_conn
    with get_conn() as conn:
        yield conn.cursor()


# (view, the minimum row count that proves it is really the canonical source)
CANONICAL_VIEWS = [
    ("proc.bp_category_master", 200),
    ("proc.bp_product_master", 150),
    ("proc.bp_category_product_map", 20),
    ("proc.bp_supplier_master", 1000),
    ("proc.bp_contract_master", 3000),
]


@pytest.mark.parametrize("view,minimum", CANONICAL_VIEWS)
def test_canonical_view_is_reachable_and_populated(cur, view, minimum):
    cur.execute(f"SELECT count(*) FROM {view}")
    count = cur.fetchone()[0]
    assert count >= minimum, (
        f"{view} returned {count} rows, expected at least {minimum} — the FDW "
        "link to uicanvas is broken or pointing somewhere unexpected"
    )


def test_the_taxonomy_carries_its_unspsc_codes(cur):
    """UNSPSC is the coded standard the B3 analysis concluded did not exist —
    because it only ever queried the databases where the canonical data isn't.
    If these stop arriving, that conclusion silently becomes true again."""
    cur.execute("""
        SELECT count(*), count(unspsc_code), count(DISTINCT category_level_5)
          FROM proc.bp_category_master
    """)
    rows, coded, leaves = cur.fetchone()
    assert coded == rows, f"{rows - coded} categories lost their UNSPSC code"
    assert leaves > 200, "the 5-level hierarchy has collapsed"


def test_the_product_master_still_carries_units_of_measure(cur):
    """This is the only real unit vocabulary in the system. src/services/facts/
    uom.py is a hand-typed map and this is what it should be reconciled against."""
    cur.execute("""
        SELECT count(*) FROM proc.bp_product_master
         WHERE btrim(coalesce(unit_of_measure, '')) <> ''
    """)
    assert cur.fetchone()[0] >= 50


@pytest.mark.parametrize("statement", [
    "UPDATE proc.bp_product_master SET unit_of_measure = 'x'",
    "DELETE FROM proc.bp_contract_master",
    "INSERT INTO proc.bp_category_master (unspsc_code) VALUES ('x')",
])
def test_the_canonical_source_cannot_be_written_through(cur, statement):
    """Enforced by the wrapper (server option updatable 'false'), not by
    convention. A write reaching uicanvas from here would corrupt the
    authoritative copy for every system that reads it."""
    with pytest.raises(Exception) as exc:
        cur.execute(statement)
    assert "not allow" in str(exc.value).lower(), (
        f"expected the wrapper to refuse the write, got: {exc.value}"
    )


def test_the_degraded_local_category_table_was_not_replaced(cur):
    """proc.bp_category is a different, flat (item_description, category) table.
    Overwriting it would have broken whatever still reads it, so the canonical
    taxonomy is exposed under a new name instead. This pins that separation."""
    cur.execute("""
        SELECT string_agg(column_name, ',' ORDER BY ordinal_position)
          FROM information_schema.columns
         WHERE table_schema = 'proc' AND table_name = 'bp_category'
    """)
    assert cur.fetchone()[0] == "item_description,category"
