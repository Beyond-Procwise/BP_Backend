"""The phase's acceptance bar.

"A query over Finding can reconstruct, for any opportunity, the unit price,
contract version, term, and source document of every number contributing to it
— without parsing free text."

Two things are asserted here. First, textually: the query must not reach into
JSONB or use a regex, because a reconstruction that parses free text is exactly
what this phase exists to remove. Second, behaviourally: the reconstruction
must let a caller tell a unit rate from a total WITHOUT inspecting the number,
which is the question the model was built to answer.

Per F2, provenance is real only on bp_sqldb; bp_testdb is a seeded corpus with
almost none. Run with PROCWISE_TEST_LIVE_DB=1 and PGDATABASE=bp_sqldb for the
live half — the structural assertions run everywhere.
"""
from __future__ import annotations

import os
import re
import sys
from decimal import Decimal
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# The acceptance query. One statement, four tables, no free text anywhere.
RECONSTRUCT_SQL = """
SELECT o.opportunity_ref_id,
       ff.role                AS fact_role,
       f.fact_id,
       f.measure_role,
       f.basis_uom,
       f.arithmetic_state,
       f.unit_price,
       f.quantity,
       f.uom,
       f.uom_normalised,
       f.extended_value,
       f.currency,
       f.fx_rate,
       f.fx_rate_date,
       f.contract_id,
       f.document_version,
       f.term_start,
       f.term_end,
       f.term_months,
       p.document_id,
       p.page,
       p.locator,
       p.verbatim_snippet
  FROM proc.bp_opportunity     o
  JOIN proc.bp_finding_fact    ff ON ff.opportunity_ref_id = o.opportunity_ref_id
  JOIN proc.bp_commercial_fact f  ON f.fact_id = ff.fact_id
  JOIN proc.bp_fact_provenance p  ON p.fact_id = f.fact_id
 WHERE o.opportunity_ref_id = %s
 ORDER BY ff.role, f.fact_id, p.field_path
"""

_BANNED = {
    "->>": "JSONB text extraction",
    "->": "JSONB navigation",
    "#>": "JSONB path extraction",
    "jsonb_extract": "JSONB extraction",
    "jsonb_path": "JSONB path query",
    "calculation_details": "the deprecated free-text payload",
    "regexp_": "regular expression parsing",
    "~*": "regular expression matching",
    " like ": "pattern matching on a value",
    "substring(": "string surgery",
    "split_part": "string surgery",
}


def test_the_query_parses_no_free_text():
    """Assert textually, the way Phase 1a's coverage harness asserts its SQL
    reads the provenance table. A reconstruction that has to parse a string is
    not a reconstruction, it is a second extractor."""
    sql = RECONSTRUCT_SQL.lower()
    for token, why in _BANNED.items():
        assert token not in sql, f"acceptance query uses {token!r} ({why})"


def test_the_query_reads_all_four_tables():
    sql = RECONSTRUCT_SQL.lower()
    for table in ("proc.bp_opportunity", "proc.bp_finding_fact",
                  "proc.bp_commercial_fact", "proc.bp_fact_provenance"):
        assert table in sql, f"acceptance query must join {table}"


def test_the_query_returns_the_columns_the_brief_names():
    sql = RECONSTRUCT_SQL.lower()
    for column in ("unit_price", "quantity", "uom", "currency",
                   "contract_id", "document_version", "term_start", "term_end",
                   "measure_role", "basis_uom", "arithmetic_state",
                   "document_id", "page", "locator", "verbatim_snippet"):
        assert column in sql, f"acceptance query must return {column}"


def test_it_is_a_single_statement():
    assert RECONSTRUCT_SQL.strip().rstrip(";").count(";") == 0


# ---------------------------------------------------------------------------
# Live reconstruction. Skipped unless pointed at a real database.
# ---------------------------------------------------------------------------

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live_only = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

REF = "acceptance-reconstruct-ref"


def _seed(cur):
    """Two facts on one finding, same number, different roles.

    This is the comparability case: 4586.65 as a unit rate and 4586.65 as a
    line total are the SAME number and utterly different facts. If the
    reconstruction cannot separate them, a later comparison will put one
    supplier's unit rate against another's total and return a confident wrong
    answer.
    """
    _clean(cur)
    cur.execute(
        "INSERT INTO proc.bp_opportunity (opportunity_id, opportunity_ref_id, detector_type) "
        "VALUES (%s, %s, %s) ON CONFLICT (opportunity_id) DO NOTHING",
        (REF, REF, "Acceptance Harness"),
    )

    rows = [
        ("acc-unit-each", "unit_rate", "each", "consistent", Decimal("4586.65")),
        ("acc-extended", "extended_line", None, "consistent", Decimal("4586.65")),
        ("acc-unit-month", "unit_rate", "month", "consistent", Decimal("4586.65")),
    ]
    for fact_id, role, basis, arith, price in rows:
        cur.execute(
            "INSERT INTO proc.bp_commercial_fact "
            "(fact_id, fact_type, measure_role, basis_uom, arithmetic_state, "
            " unit_price, quantity, extended_value, currency, contract_id, "
            " document_version, term_months) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (fact_id, "line_unit_price", role, basis, arith, price,
             Decimal("2"), Decimal("9173.30"), "GBP", "CTR-1", "v3", 12),
        )
        cur.execute(
            "INSERT INTO proc.bp_fact_provenance "
            "(fact_id, document_id, field_path, locator, page, verbatim_snippet) "
            "VALUES (%s,%s,%s,%s,%s,%s)",
            (fact_id, "INV-ACC-1", "line_items[0].unit_price",
             "bbox:10,20,30,40", 1, "4,586.65"),
        )
        cur.execute(
            "INSERT INTO proc.bp_finding_fact (opportunity_ref_id, fact_id, role) "
            "VALUES (%s,%s,%s)",
            (REF, fact_id, "contributing"),
        )


def _clean(cur):
    cur.execute("DELETE FROM proc.bp_finding_fact WHERE opportunity_ref_id = %s", (REF,))
    cur.execute("DELETE FROM proc.bp_commercial_fact WHERE fact_id LIKE 'acc-%%'")
    cur.execute("DELETE FROM proc.bp_opportunity WHERE opportunity_ref_id = %s", (REF,))


@pytest.fixture()
def live_cur():
    from src.services.db import get_conn
    with get_conn() as conn:
        conn.autocommit = False
        cur = conn.cursor()
        try:
            _seed(cur)
            yield cur
        finally:
            _clean(cur)
            conn.commit()


@live_only
def test_the_reconstruction_returns_every_contributing_number(live_cur):
    live_cur.execute(RECONSTRUCT_SQL, (REF,))
    cols = [d[0] for d in live_cur.description]
    rows = [dict(zip(cols, r)) for r in live_cur.fetchall()]

    assert len(rows) == 3, "every contributing fact must come back"
    for row in rows:
        # provenance, reachable without parsing anything
        assert row["document_id"]
        assert row["locator"]
        assert row["verbatim_snippet"]
        # the commercial reconstruction the brief asks for
        assert row["unit_price"] is not None
        assert row["currency"] == "GBP"
        assert row["contract_id"] == "CTR-1"
        assert row["document_version"] == "v3"
        assert row["term_months"] == 12


@live_only
def test_a_unit_rate_and_a_total_are_distinguishable_without_reading_the_number(live_cur):
    """The comparability criterion. Same value, different roles — and the
    reconstruction separates them on measure_role alone."""
    live_cur.execute(RECONSTRUCT_SQL, (REF,))
    cols = [d[0] for d in live_cur.description]
    rows = [dict(zip(cols, r)) for r in live_cur.fetchall()]

    by_role = {}
    for row in rows:
        by_role.setdefault(row["measure_role"], []).append(row)

    assert "unit_rate" in by_role and "extended_line" in by_role
    values = {row["unit_price"] for row in rows}
    assert len(values) == 1, "the fixture's whole point is one identical value"
    assert by_role["unit_rate"][0]["measure_role"] != by_role["extended_line"][0]["measure_role"]


@live_only
def test_two_unit_rates_on_different_bases_are_not_treated_as_comparable(live_cur):
    """£4,586.65 per each and £4,586.65 per month are not the same price. The
    basis is what says so."""
    live_cur.execute(RECONSTRUCT_SQL, (REF,))
    cols = [d[0] for d in live_cur.description]
    rows = [dict(zip(cols, r)) for r in live_cur.fetchall()]

    rates = [r for r in rows if r["measure_role"] == "unit_rate"]
    assert len(rates) == 2
    bases = {r["basis_uom"] for r in rates}
    assert bases == {"each", "month"}, "the basis must survive the reconstruction"
    assert len({r["unit_price"] for r in rates}) == 1, (
        "identical numbers, different bases: comparing them on value alone "
        "would be the exact error measure_role and basis_uom exist to prevent"
    )


@live_only
def test_the_arithmetic_state_survives_so_reliability_is_visible(live_cur):
    live_cur.execute(RECONSTRUCT_SQL, (REF,))
    cols = [d[0] for d in live_cur.description]
    rows = [dict(zip(cols, r)) for r in live_cur.fetchall()]
    assert all(r["arithmetic_state"] for r in rows)
