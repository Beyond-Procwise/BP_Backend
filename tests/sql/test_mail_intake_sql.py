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


def _table_block(name: str) -> str:
    """The CREATE TABLE body for `name`.

    Assertions run against one table's body rather than the whole file, so a
    constraint on a different table can never satisfy them by accident.
    """
    match = re.search(
        r"CREATE TABLE IF NOT EXISTS proc\." + name + r"\s*\((.*?)\n\);", SQL, re.S
    )
    assert match, f"{name} must be created"
    return match.group(1)


def _check_values(block: str, column: str) -> set:
    """The exact value set of `column`'s CHECK (... IN (...)) within one table."""
    match = re.search(
        r"\n\s*" + column + r"\b.*?CHECK\s*\(.*?IN\s*\((.*?)\)\)", block, re.S
    )
    assert match, f"{column} must carry a CHECK constraint"
    return set(re.findall(r"'([a-z_]+)'", match.group(1)))


@pytest.mark.parametrize("table", EXPECTED_TABLES)
def test_table_is_created_idempotently(table):
    assert f"CREATE TABLE IF NOT EXISTS {table}" in SQL


def test_bundle_state_check_matches_the_enum_exactly():
    # Equality, not subset. A CHECK listing a state the code cannot produce is
    # as much a defect as one missing a state the code does produce: the first
    # silently permits an unreachable value, the second rejects a legitimate one
    # at runtime. A subset assertion only ever catches the second.
    assert _check_values(_table_block("bp_ingest_bundle"), "state") == {
        s.value for s in BundleState
    }


def test_artifact_state_check_matches_the_enum_exactly():
    assert _check_values(_table_block("bp_ingest_artifact"), "state") == {
        s.value for s in ArtifactState
    }


@pytest.mark.parametrize("table", ["bp_ingest_bundle", "bp_ingest_artifact"])
def test_quarantine_reason_check_matches_the_enum_exactly(table):
    assert _check_values(_table_block(table), "quarantine_reason") == {
        r.value for r in QuarantineReason
    }


def test_content_hash_is_unique_within_the_dedup_scope():
    # Spec §5.3: dedup scope is the binding OWNER, not the binding. Asserted as
    # one match so a UNIQUE keyword elsewhere in the file cannot satisfy it.
    match = re.search(
        r"CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_ingest_artifact_owner_sha\s+"
        r"ON proc\.bp_ingest_artifact\s*\(([^)]*)\)",
        SQL,
    )
    assert match, "the dedup index must exist and must be UNIQUE"
    assert [c.strip() for c in match.group(1).split(",")] == ["owner_ref", "sha256"]


@pytest.mark.parametrize("table", ["bp_ingest_bundle", "bp_ingest_artifact"])
def test_binding_id_is_bigint_and_references_the_binding_table(table):
    # proc.bp_mailbox_binding.binding_id is BIGINT GENERATED ALWAYS AS IDENTITY
    # and is that table's primary key. TEXT here would need an explicit cast on
    # every join, and without the FK a bundle could name a binding that does
    # not exist — which is exactly the audit-spine break the design forbids.
    block = _table_block(table)
    match = re.search(r"\n\s*binding_id\s+(\w+)[^,]*?REFERENCES\s+(\S+)\s*\(binding_id\)", block)
    assert match, f"{table}.binding_id must be a typed FK to the binding table"
    assert match.group(1) == "BIGINT"
    assert match.group(2) == "proc.bp_mailbox_binding"


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


def test_binding_provider_constraint_includes_relay():
    # The mail-intake binding uses provider='relay'. The original constraint
    # (graph/gmail/imap) did not permit this, so the migration widens it.
    assert re.search(
        r"ALTER TABLE proc\.bp_mailbox_binding\s+ADD CONSTRAINT bp_mailbox_binding_provider_check\s+"
        r"CHECK\s*\(\s*provider\s*=\s*ANY\s*\(\s*ARRAY\['graph',\s*'gmail',\s*'imap',\s*'relay'\]\s*\)",
        SQL,
        re.S
    ), "provider constraint must include 'relay'"


def test_binding_role_constraint_includes_intake():
    # The mail-intake binding uses role='intake'. The original constraint
    # (exemplar_source/draft_target/both) did not permit this, so the migration widens it.
    assert re.search(
        r"ALTER TABLE proc\.bp_mailbox_binding\s+ADD CONSTRAINT bp_mailbox_binding_role_check\s+"
        r"CHECK\s*\(\s*role\s*=\s*ANY\s*\(\s*ARRAY\['exemplar_source',\s*'draft_target',\s*'both',\s*'intake'\]\s*\)",
        SQL,
        re.S
    ), "role constraint must include 'intake'"


def test_the_dedup_index_covers_exactly_the_original_states():
    """The unique index and find_existing_artifact must name the same states, or
    a second arrival raises at insert (the 2026-10-14 live failure) or two
    originals of one file both reach extraction."""
    from src.services.mail_intake.repo import ORIGINAL_STATES

    fix = pathlib.Path("scripts/migrations/2026-10-14-mail-intake-dedup-index.sql").read_text()
    match = re.search(r"CREATE UNIQUE INDEX ix_bp_ingest_artifact_owner_sha.*?state IN \((.*?)\);", fix, re.S)
    assert match, "the replacement index must be created"
    assert set(re.findall(r"'([a-z_]+)'", match.group(1))) == set(ORIGINAL_STATES)
