"""The critique DDL says what the spec says it says.

Asserted against the migration text rather than a live database so the test
runs anywhere. Applying it against bp_testdb is verified in Task 9.
"""
from pathlib import Path

SQL = (Path(__file__).resolve().parents[2]
       / "deploy" / "sql" / "2026-09-09_opportunity_critique.sql").read_text()
ROLLBACK = (Path(__file__).resolve().parents[2]
            / "deploy" / "sql" / "2026-09-09_opportunity_critique_rollback.sql").read_text()


def test_both_tables_created_idempotently():
    for table in ("bp_opportunity_critique", "bp_opportunity_gap"):
        assert f"CREATE TABLE IF NOT EXISTS proc.{table}" in SQL


def test_keyed_on_ref_id_not_the_per_run_counter():
    # opportunity_id is a per-run counter that changes for the same finding
    # between mining runs; keying on it orphans every verdict. See
    # src/services/opportunity_store.py:24.
    assert "opportunity_ref_id TEXT        NOT NULL" in SQL
    assert "opportunity_id" not in SQL


def test_verdict_is_constrained_to_the_five_values():
    assert (
        "CHECK (verdict IN ('VALID', 'VALID_REFRAMED', 'INVALID', "
        "'UNASSESSED', 'DUPLICATE'))"
    ) in SQL


def test_confidence_uses_the_existing_ladder():
    assert "CHECK (confidence IN ('ASSERTED', 'CORROBORATED', 'UNASSESSED'))" in SQL


def test_gap_type_is_constrained_to_the_seven_types():
    for gap_type in ("MISSING_EVIDENCE", "STALE_EVIDENCE", "UNVERIFIED_ASSERTION",
                     "NORMALISATION_NEEDED", "NO_LEVER", "NO_THRESHOLD",
                     "DETECTOR_LOGIC"):
        assert gap_type in SQL


def test_effort_is_constrained():
    assert "CHECK (effort IN ('LOW', 'MEDIUM', 'HIGH'))" in SQL


def test_versions_that_produced_the_verdict_are_recorded():
    # Without these, tuning a threshold leaves stale verdicts on the page
    # presenting themselves as current. Spec section 9.1.
    for col in ("prompt_version", "policy_versions", "formula_versions"):
        assert col in SQL


def test_every_test_result_is_stored_including_passes():
    assert "tests              JSONB" in SQL


def test_shadow_columns_present():
    assert "would_have_suppressed" in SQL
    assert "shadowed" in SQL


def test_gaps_cascade_so_a_critique_deletes_cleanly():
    assert "ON DELETE CASCADE" in SQL


def test_rollback_drops_both_tables():
    assert "DROP TABLE IF EXISTS proc.bp_opportunity_gap" in ROLLBACK
    assert "DROP TABLE IF EXISTS proc.bp_opportunity_critique" in ROLLBACK
