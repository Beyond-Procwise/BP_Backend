# tests/sql/test_bp_style_profile_immutable_sql.py
#
# Invariant 5 is enforced by a trigger rather than by the repository, because the
# repository is not the only thing that will ever hold a connection to this table.
# The behavioural proof lives in tests/services/test_style_repository.py, which runs
# against a real Postgres; these assert the migration pair's shape.
from pathlib import Path

SQL = Path("deploy/sql/2026-07-27_bp_style_profile_immutable.sql").read_text()
ROLLBACK = Path("deploy/sql/2026-07-27_bp_style_profile_immutable_rollback.sql").read_text()


def test_migration_is_transactional():
    assert SQL.strip().startswith("BEGIN") and "COMMIT;" in SQL
    assert ROLLBACK.strip().startswith("BEGIN") and "COMMIT;" in ROLLBACK


def test_is_idempotent():
    """Re-running must be a no-op, not an error."""
    assert "CREATE OR REPLACE FUNCTION" in SQL
    assert "DROP TRIGGER IF EXISTS" in SQL


def test_fires_before_update_on_every_row():
    assert "BEFORE UPDATE ON proc.bp_style_profile" in SQL
    assert "FOR EACH ROW" in SQL


def test_freezes_profile_json_once_approved():
    assert "NEW.profile_json IS DISTINCT FROM OLD.profile_json" in SQL
    assert "immutable once approved" in SQL


def test_covers_superseded_as_well_as_approved():
    """A superseded profile was approved once, and the drafts citing it must keep pointing
    at the rules that actually produced them. Editing history is worse than deleting it,
    because it still looks trustworthy."""
    assert "OLD.state NOT IN ('APPROVED', 'SUPERSEDED')" in SQL


def test_freezes_row_identity_too():
    for col in ("NEW.user_ref", "NEW.intent", "NEW.version"):
        assert f"{col} IS DISTINCT FROM" in SQL, col


def test_approval_is_a_one_way_door():
    assert "cannot return to" in SQL


def test_leaves_state_and_is_active_mutable():
    """Standing a profile down to SUPERSEDED is the normal lifecycle; freezing those
    columns outright would make approval itself impossible."""
    assert "NEW.is_active IS DISTINCT FROM OLD.is_active" not in SQL


def test_rollback_removes_both_trigger_and_function():
    assert "DROP TRIGGER IF EXISTS tr_bp_style_profile_freeze_approved" in ROLLBACK
    assert "DROP FUNCTION IF EXISTS proc.bp_style_profile_freeze_approved" in ROLLBACK


def test_rollback_does_not_touch_the_rows():
    assert "DELETE" not in ROLLBACK.upper()
    assert "DROP TABLE" not in ROLLBACK.upper()
