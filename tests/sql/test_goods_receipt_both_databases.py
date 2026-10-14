"""The goods-receipt schema exists in BOTH databases, identically.

bp_sqldb has run behind before. The governance tables were eight migrations
behind as recently as 2026-09-15, and a missing partial unique index there cost
65 days of discrepancy findings -- writes were REJECTED and nothing said so.
Deployment to both is a prerequisite of this feature, not a follow-up, so it is
a test rather than a checklist item.

bp_sqldb is not in .env (which names bp_testdb), so it is reached by overriding
the database name on the same cluster.

Live-only. Run with:
    set -a && . ./.env && set +a
    PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest \
        tests/sql/test_goods_receipt_both_databases.py
"""
from __future__ import annotations

import os
import re

import pytest

_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
pytestmark = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

DATABASES = ("bp_testdb", "bp_sqldb")

_PRICED = re.compile(r"(price|amount|total|value|cost|currency|tax)", re.I)

_TABLES = tuple(
    f"bp_goods_receipt{suffix}_{tier}"
    for suffix in ("", "_line_items")
    for tier in ("raw", "stg", "trgt")
)


def _connect(dbname: str):
    import psycopg2

    try:
        return psycopg2.connect(
            host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT", 5432),
            dbname=dbname, user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"), connect_timeout=15)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot reach {dbname}: {exc}")


def _rows(dbname: str, sql: str, args=()):
    conn = _connect(dbname)
    try:
        with conn, conn.cursor() as cur:
            cur.execute(sql, args)
            return cur.fetchall()
    finally:
        conn.close()


@pytest.mark.parametrize("dbname", DATABASES)
def test_all_six_tables_exist(dbname):
    present = {r[0] for r in _rows(dbname,
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='proc' AND table_name = ANY(%s)", (list(_TABLES),))}
    assert present == set(_TABLES), f"{dbname} is missing {sorted(set(_TABLES) - present)}"


@pytest.mark.parametrize("dbname", DATABASES)
def test_no_priced_column_on_any_of_them(dbname):
    bad = _rows(dbname,
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema='proc' AND table_name = ANY(%s)", (list(_TABLES),))
    offenders = sorted({(t, c) for t, c in bad if _PRICED.search(c)})
    assert offenders == [], f"{dbname}: {offenders}"


@pytest.mark.parametrize("dbname", DATABASES)
def test_the_trgt_tier_is_keyed_by_deal_id(dbname):
    for table in ("bp_goods_receipt_trgt", "bp_goods_receipt_line_items_trgt"):
        cols = {r[0] for r in _rows(dbname,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='proc' AND table_name=%s", (table,))}
        assert "deal_id" in cols, f"{dbname}.{table} has no deal_id"


def test_the_column_sets_are_identical_between_the_two():
    """Not just present in both -- the SAME. A column added to one database and
    not the other is how a reader works on staging and raises UndefinedColumn in
    production."""
    def shape(dbname):
        return {
            (t, c, d) for t, c, d in _rows(dbname,
                "SELECT table_name, column_name, data_type "
                "FROM information_schema.columns "
                "WHERE table_schema='proc' AND table_name = ANY(%s)", (list(_TABLES),))
        }
    a, b = shape("bp_testdb"), shape("bp_sqldb")
    assert a == b, (
        "only in bp_testdb: " + repr(sorted(a - b)) +
        "\nonly in bp_sqldb: " + repr(sorted(b - a)))


@pytest.mark.parametrize("dbname", DATABASES)
def test_the_vocabulary_row_is_there_with_its_eleven_aliases(dbname):
    rows = _rows(dbname,
        "SELECT pipeline_doc_type, aliases, status FROM proc.bp_document_type "
        "WHERE concept_code='doctype.goods_receipt'")
    assert rows, f"{dbname} has no doctype.goods_receipt row"
    pipeline, aliases, status = rows[0]
    assert pipeline == "goods_receipt"
    assert status == "active"
    assert aliases == ["goods receipt", "goods received note", "grn", "delivery note",
                       "despatch note", "dispatch note", "advice note", "packing list",
                       "packing slip", "proof of delivery", "pod"]


@pytest.mark.parametrize("dbname", DATABASES)
def test_the_check_constraint_accepts_the_fifth_pipeline(dbname):
    [(definition,)] = _rows(dbname,
        "SELECT pg_get_constraintdef(oid) FROM pg_constraint "
        "WHERE conname='ck_bp_document_type_pipeline'")
    assert "goods_receipt" in definition, definition


@pytest.mark.parametrize("dbname", DATABASES)
def test_every_unit_declares_a_receipt_basis(dbname):
    unclassified = _rows(dbname,
        "SELECT uom_code FROM proc.bp_uom_canonical WHERE receipt_basis IS NULL")
    assert unclassified == [], f"{dbname}: {unclassified}"


@pytest.mark.parametrize("dbname", DATABASES)
def test_the_tolerance_policy_is_active_and_states_all_three_rules(dbname):
    rows = _rows(dbname,
        "SELECT policy_type, policy_status, policy_details->'rules' FROM proc.bp_policy "
        "WHERE policy_details->>'policy_identifier'='receipt_tolerances'")
    assert len(rows) == 1, f"{dbname} has {len(rows)} receipt_tolerances rows"
    policy_type, status, rules = rows[0]
    assert policy_type == "limit"      # or the drift test never sees it
    assert status == 1
    assert set(rules) == {"over_delivery_pct", "billed_over_received_qty",
                          "uom_conversion_required"}


@pytest.mark.parametrize("dbname", DATABASES)
def test_the_deal_overview_offers_both_columns_and_not_the_old_one(dbname):
    cols = {r[0] for r in _rows(dbname,
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='proc' AND table_name='bp_deal_overview'")}
    assert "value_reconciled" in cols
    assert "three_way_matched" in cols
    assert "three_way_match" not in cols


@pytest.mark.parametrize("dbname", DATABASES)
def test_the_kpi_view_survived_the_cascade(dbname):
    """DROP VIEW ... CASCADE takes bp_deal_kpis with it. The migration rebuilds
    it in the same transaction; if that were ever dropped from the migration,
    the KPI row would simply be gone and the gateway would 500."""
    cols = {r[0] for r in _rows(dbname,
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema='proc' AND table_name='bp_deal_kpis'")}
    assert "three_way_match_pct" in cols
    assert "three_way_matched_pct" in cols
