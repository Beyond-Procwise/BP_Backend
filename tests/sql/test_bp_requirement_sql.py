from pathlib import Path

SQL = Path("deploy/sql/2026-06-17_bp_requirement.sql").read_text()


def test_table_and_key_columns_present():
    lowered = SQL.lower()
    assert "create table if not exists proc.bp_requirement" in lowered
    for col in (
        "requirement_id", "session_id", "status", "created_by",
        "title", "category", "description", "quantity", "unit",
        "target_budget", "currency", "needed_by_date", "delivery_location",
        "priority", "specifications", "constraints",
        "completeness_score", "missing_fields", "seed_context",
        "created_at", "updated_at",
    ):
        assert col in lowered, f"missing column {col}"


def test_status_check_and_indexes_and_idempotent():
    lowered = SQL.lower()
    assert "begin;" in lowered and "commit;" in lowered
    assert "ix_bp_requirement_status" in lowered
    assert "ix_bp_requirement_category" in lowered
    assert "ix_bp_requirement_created_by" in lowered
    for state in ("draft", "gathering", "complete", "handed_off", "abandoned"):
        assert state in lowered, f"missing status {state} in CHECK"
