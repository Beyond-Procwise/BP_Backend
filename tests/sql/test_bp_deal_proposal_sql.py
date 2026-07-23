# tests/sql/test_bp_deal_proposal_sql.py
from pathlib import Path

SQL = Path("deploy/sql/2026-07-23_bp_deal_proposal.sql").read_text()

def test_creates_both_proposal_tables_with_bp_prefix():
    assert "CREATE TABLE IF NOT EXISTS proc.bp_deal_proposal" in SQL
    assert "CREATE TABLE IF NOT EXISTS proc.bp_deal_proposal_member" in SQL

def test_proposal_has_required_columns_and_status_default():
    for col in ("proposal_id", "batch_deal_id", "proposed_name", "confidence",
                "status", "created_at", "confirmed_at", "confirmed_by", "deal_id"):
        assert col in SQL, col
    assert "DEFAULT 'proposed'" in SQL

def test_member_carries_evidence_and_review_and_fk():
    for col in ("doc_type", "doc_pk", "base_reference", "role",
                "match_score", "match_evidence", "review_required", "review_reasons"):
        assert col in SQL, col
    assert "REFERENCES proc.bp_deal_proposal" in SQL

def test_indexes_follow_convention_and_ddl_is_transactional():
    assert "ix_bp_deal_proposal_batch" in SQL
    assert "ix_bp_deal_proposal_member_proposal" in SQL
    assert SQL.strip().startswith("BEGIN") and "COMMIT;" in SQL
