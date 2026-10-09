"""A re-read reaches _trgt only when it is no worse than what _trgt holds (linking_engine).

Both promotion paths copied a document into _trgt only while it was absent, so a re-read fixed
_stg and _trgt kept the old figures: Aureus AUR-2025-0619 V1 showed its 3-year subtotals as Year 1
costs for ten weeks. But on 2026-10-09, 19 of the 22 re-read documents had LOST their total and
every priced line; a blind refresh would have wiped them. So the rule is a verdict, not a copy.
"""
import os

import pytest

from src.services import linking_engine as le


def test_a_reread_at_least_as_complete_replaces_the_record():
    assert le.reread_verdict(1326000, 3485184, 4, 4) is None          # Aureus V1: right figures, same lines
    assert le.reread_verdict(1100, 1000, 5, 4) is None                # more lines priced
    assert le.reread_verdict(None, None, 0, 0) is None                # nothing to lose


def test_a_reread_that_lost_figures_never_overwrites_the_record():
    assert le.reread_verdict(None, 1200000, 0, 4) == "the re-read has no total"   # Meridia V3's re-read
    assert le.reread_verdict(1200000, 1200000, 2, 4) == "the re-read prices 2 lines, the record 4"


def test_the_scheduled_promotion_runs_the_refresh():
    import inspect
    assert "refresh_rereads(cur)" in inspect.getsource(le._promote)


def test_deal_and_identity_columns_survive_a_refresh():
    # Deal columns are never copied stg->trgt; a refresh also keeps when the record was first made
    # and restores each line's deal stamps, which deleting and re-inserting the lines would drop.
    assert {"created_date", "created_by"} <= le._REREAD_KEEP
    assert set(le._LINE_STAMPS) == {"deal_id", "deal_name", "document_id"}
    assert {"deal_id", "deal_name", "document_id", "deal_date"} <= le._DEAL_COLS


@pytest.mark.skipif(os.getenv("PROCWISE_TEST_LIVE_DB") != "1", reason="needs the live database")
def test_live_aureus_v1_is_corrected_and_lossy_rereads_are_kept(monkeypatch):
    """Against bp_testdb, in a transaction that is rolled back: put Aureus V1 back the way its
    July read left it, run the refresh, and check what it does to it and to Meridia V3."""
    import psycopg2
    monkeypatch.setattr(le, "record_action", lambda **k: None)
    conn = psycopg2.connect(host=os.environ["DB_HOST"], dbname=os.environ["DB_NAME"],
                            user=os.environ["DB_USER"], password=os.environ["DB_PASSWORD"])
    conn.autocommit = False
    cur = conn.cursor()
    try:
        cur.execute("update proc.bp_quote_line_items_trgt set line_total = case line_number when 1 then 3485184 "
                    "when 2 then 304410 when 3 then 136674 else line_total end where quote_id = 'AUR-2025-0619'")
        cur.execute("update proc.bp_quote_trgt set total_amount = 3485184, last_modified_date = '2026-07-30' "
                    "where quote_id = 'AUR-2025-0619'")
        out = le.refresh_rereads(cur)
        assert out["quote"]["refreshed"] >= 1
        cur.execute("select line_total, deal_id from proc.bp_quote_line_items_trgt "
                    "where quote_id = 'AUR-2025-0619' order by line_number")
        assert [r[0] for r in cur.fetchall()] == [1122000, 98000, 44000, 62000]
        cur.execute("select total_amount from proc.bp_quote_trgt where quote_id = 'AUR-2025-0619'")
        assert cur.fetchone()[0] == 1326000
        cur.execute("select total_amount from proc.bp_quote_trgt where quote_id = 'MCP-Q-7740 (V3)'")
        assert cur.fetchone()[0] == 1200000                       # its lossy re-read was not applied
        cur.execute("select count(*) from proc.bp_extraction_discrepancy where issue_type = 'reread_not_applied' "
                    "and doc_pk_candidate = 'MCP-Q-7740 (V3)' and status = 'open'")
        assert cur.fetchone()[0] == 1
        assert le.refresh_rereads(cur)["quote"]["refreshed"] == 0  # idempotent
    finally:
        conn.rollback()
        conn.close()
