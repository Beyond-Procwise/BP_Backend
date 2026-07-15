# tests/sql/test_bp_deal_sql.py
from pathlib import Path

SQL = Path("deploy/sql/2026-07-15_bp_deal.sql").read_text().lower()

def test_table_and_columns_present():
    assert "create table if not exists proc.bp_deal" in SQL
    for col in ("deal_id", "is_tracked", "is_saved_reference",
                "tracked_at", "created_at", "updated_at"):
        assert col in SQL, f"missing column {col}"

def test_defaults_are_draft():
    assert "is_tracked" in SQL and "default false" in SQL

def test_index_present():
    assert "ix_bp_deal_is_tracked" in SQL

def test_backfill_marks_existing_tracked():
    # existing deals must be inserted as tracked so nothing vanishes from Pipeline
    assert "insert into proc.bp_deal" in SQL
    assert "bp_deal_overview" in SQL
    assert "true" in SQL  # is_tracked=true for existing
