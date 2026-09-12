"""Live-database fixture for the sell side.

Every row these tests write carries the LIVETEST sentinel (account ids, SKUs,
feed names, mapping profiles), and `clean` removes exactly those rows, children
first -- before the test, so a crashed earlier run cannot poison this one, and
after it. The services commit on their own, so a rolled-back transaction would
prove nothing; cleanup is by sentinel instead.

Run: set -a && . ./.env && set +a; PROCWISE_TEST_LIVE_DB=1 ./venv/bin/python -m pytest tests/sell_side
"""
import os

import pytest

SENTINEL = "LIVETEST"
_LIVE = os.environ.get("PROCWISE_TEST_LIVE_DB", "").strip().lower() in ("1", "true", "yes", "on")
live = pytest.mark.skipif(not _LIVE, reason="needs PROCWISE_TEST_LIVE_DB=1")

_QUOTES = "SELECT sales_quote_id FROM proc.bp_sales_quote WHERE account_id LIKE 'LIVETEST-%%'"


def clean(cur):
    cur.execute(f"DELETE FROM proc.bp_sales_quote_outcome WHERE sales_quote_id IN ({_QUOTES})")
    cur.execute(f"DELETE FROM proc.bp_sales_quote_line WHERE sales_quote_id IN ({_QUOTES})")
    cur.execute("UPDATE proc.bp_sales_quote SET supersedes_id = NULL WHERE account_id LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_sales_quote WHERE account_id LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_sales_opportunity WHERE account_id LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_account WHERE account_id LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_catalog_item_match WHERE distributor_sku LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_catalog_item_relation WHERE from_sku LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_catalog_item WHERE distributor_sku LIKE 'LIVETEST-%%'")
    cur.execute("DELETE FROM proc.bp_catalog_source WHERE feed_name LIKE 'LIVETEST%%'")
    cur.execute("DELETE FROM proc.bp_catalog_mapping WHERE mapping_profile LIKE 'livetest%%'")


@pytest.fixture
def live_db():
    import psycopg2

    conn = psycopg2.connect(
        host=os.environ["DB_HOST"], port=os.getenv("DB_PORT", 5432),
        dbname=os.environ["DB_NAME"], user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"], connect_timeout=10)
    try:
        with conn.cursor() as cur:
            clean(cur)
            cur.execute("SELECT supplier_id FROM proc.bp_supplier ORDER BY supplier_id LIMIT 1")
            distributor_id = cur.fetchone()[0]
        conn.commit()
        yield conn, distributor_id
    finally:
        conn.rollback()
        with conn.cursor() as cur:
            clean(cur)
        conn.commit()
        conn.close()


def seed_item(conn, distributor_id, sku, *, cost="10.0000", list_price="15.0000",
              currency="GBP", mpn=None, description="LIVETEST widget", tiers=()):
    """One current catalog version (+ optional tiers). Returns catalog_item_id."""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO proc.bp_catalog_source (distributor_id, feed_name, content_sha256, "
            "mapping_profile, price_effective, status) VALUES (%s, %s, %s, 'livetest_v1', "
            "CURRENT_DATE, 'imported') ON CONFLICT (distributor_id, content_sha256) "
            "DO UPDATE SET status = 'imported' RETURNING source_id",
            (distributor_id, f"{SENTINEL} seed", f"{SENTINEL}-seed-{sku}"))
        source_id = cur.fetchone()[0]
        cur.execute(
            "INSERT INTO proc.bp_catalog_item (source_id, distributor_id, distributor_sku, mpn, "
            "item_description, currency, list_price, cost_price) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING catalog_item_id",
            (source_id, distributor_id, sku, mpn, description, currency, list_price, cost))
        item_id = cur.fetchone()[0]
        for min_q, tier_cost in tiers:
            cur.execute(
                "INSERT INTO proc.bp_catalog_cost_tier VALUES (%s, %s, %s, %s)",
                (item_id, min_q, tier_cost, currency))
    conn.commit()
    return item_id
