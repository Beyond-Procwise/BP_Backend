"""The migration is checked as text, not by running it, so the suite stays
database-free. What matters is that the DDL and the Python enums cannot drift
apart: a CHECK constraint listing a state the code never produces (or missing
one it does) is a fail-open bug that would only surface in production."""
import pathlib
import re

import pytest

from src.services.mail_intake.states import ArtifactState, BundleState, QuarantineReason

SQL = pathlib.Path("scripts/migrations/2026-08-07-mail-intake.sql").read_text()

EXPECTED_TABLES = (
    "proc.bp_ingest_bundle",
    "proc.bp_ingest_artifact",
    "proc.bp_ingest_sender_auth",
    "proc.bp_ingest_arrival",
    "proc.bp_artifact_derivation",
)


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_table_is_created_idempotently(table):
    assert f"CREATE TABLE IF NOT EXISTS {table}" in SQL


def test_every_bundle_state_appears_in_its_check_constraint():
    clause = re.search(r"state\s+TEXT NOT NULL.*?CHECK \(state IN \((.*?)\)\)", SQL, re.S)
    assert clause, "bp_ingest_bundle.state must carry a CHECK constraint"
    listed = set(re.findall(r"'([a-z_]+)'", clause.group(1)))
    assert {s.value for s in BundleState} <= listed


def test_every_quarantine_reason_appears_in_the_ddl():
    for reason in QuarantineReason:
        assert f"'{reason.value}'" in SQL, f"{reason.value} missing from CHECK constraint"


def test_every_artifact_state_appears_in_the_ddl():
    for state in ArtifactState:
        assert f"'{state.value}'" in SQL, f"{state.value} missing from CHECK constraint"


def test_content_hash_is_unique_within_the_dedup_scope():
    # Spec §5.3: dedup scope is the binding owner, not the binding.
    assert "ix_bp_ingest_artifact_owner_sha" in SQL
    assert "UNIQUE" in SQL


def test_derivation_table_has_no_update_path():
    # Spec: reprocessing appends. A plain append-only table has no ON CONFLICT
    # DO UPDATE anywhere near it.
    section = SQL.split("bp_artifact_derivation")[1]
    assert "DO UPDATE" not in section


def test_binding_gains_intake_columns():
    for column in ("folder_scope", "poll_interval_seconds", "last_sync_at"):
        assert f"ADD COLUMN IF NOT EXISTS {column}" in SQL


def test_raw_mime_key_is_not_nullable():
    # Stage 1 persists raw MIME before parsing; a bundle without it is a bundle
    # whose provenance cannot be reconstructed.
    assert re.search(r"raw_mime_key\s+TEXT NOT NULL", SQL)
