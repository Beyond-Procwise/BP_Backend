"""The analysis-event DDL says what the spec says it says.

Asserted against the migration text rather than a live database so the test
runs anywhere. Applying it is verified in Task 13 against bp_sqldb.
"""
from pathlib import Path

SQL = (Path(__file__).resolve().parents[2]
       / "deploy" / "sql" / "2026-08-01_bp_analysis.sql").read_text()


def test_three_tables_created_idempotently():
    for table in ("bp_analysis", "bp_analysis_document", "bp_analysis_deal"):
        assert f"CREATE TABLE IF NOT EXISTS proc.{table}" in SQL


def test_session_id_is_unique_so_start_is_idempotent():
    assert "session_id      TEXT UNIQUE" in SQL


def test_version_is_unique_per_deal_not_globally():
    assert "UNIQUE (deal_id, version)" in SQL


def test_status_and_mode_are_constrained():
    assert "CHECK (status IN ('running', 'complete', 'failed'))" in SQL
    assert "CHECK (mode IN ('new', 'amend', 'bulk'))" in SQL


def test_child_rows_cascade_so_an_analysis_deletes_cleanly():
    assert SQL.count("ON DELETE CASCADE") == 2
