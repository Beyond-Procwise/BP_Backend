# tests/sql/test_bp_deal_proposal_flags_sql.py
from pathlib import Path

SQL = Path("deploy/sql/2026-07-23_bp_deal_proposal_flags.sql").read_text()


def test_adds_flags_column_idempotently():
    assert "ALTER TABLE proc.bp_deal_proposal" in SQL
    assert "ADD COLUMN IF NOT EXISTS flags JSONB" in SQL


def test_ddl_is_transactional():
    assert SQL.strip().startswith("BEGIN") and "COMMIT;" in SQL
