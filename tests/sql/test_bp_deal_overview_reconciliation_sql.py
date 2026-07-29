# tests/sql/test_bp_deal_overview_reconciliation_sql.py
"""three_way_match previously only checked that a quote, a PO, and an invoice
document all EXIST for a deal — never that their amounts agree. A deal
invoiced at 3x its PO value (DEALV2-005049, Kestrel Supplies) showed
three_way_match=true with price_variance_pct=200, and no opportunity was ever
raised because nothing read that variance. This migration makes
three_way_match require amount reconciliation, not just document presence."""
from pathlib import Path

SQL = Path(
    "deploy/sql/2026-07-29_bp_deal_overview_value_reconciliation.sql"
).read_text().lower()


def test_replaces_the_existing_view():
    assert "create or replace view proc.bp_deal_overview" in SQL


def test_three_way_match_still_requires_all_three_document_types():
    assert "quote_count>0" in SQL.replace(" ", "")
    assert "po_count>0" in SQL.replace(" ", "")
    assert "invoice_count>0" in SQL.replace(" ", "")


def test_three_way_match_now_checks_amount_reconciliation_not_just_presence():
    # the fixed variance tolerance must gate three_way_match directly —
    # merely keeping price_variance_pct as a separate column (the old bug)
    # doesn't fix the badge.
    assert "<= 0.10" in SQL.replace(" ", "") or "<=0.10" in SQL.replace(" ", "")


def test_price_variance_pct_is_still_exposed():
    assert "price_variance_pct" in SQL
