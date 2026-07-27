# tests/sql/test_bp_style_engine_sql.py
#
# Phase 0 of the email style subsystem. These assert the shape of the migration pair,
# in the same style as the other tests in this directory. Several also stand in for
# design invariants that must survive later edits to the DDL.
from pathlib import Path

SQL = Path("deploy/sql/2026-07-27_bp_style_engine.sql").read_text()
ROLLBACK = Path("deploy/sql/2026-07-27_bp_style_engine_rollback.sql").read_text()


def _ddl_only(sql: str) -> str:
    """``sql`` with ``--`` comment lines removed.

    The absence tests below ('no tenant_id', 'no vector column') must read the DDL, not
    the prose. The header comments deliberately name those things to explain why they are
    absent, and that explanation is worth more than the convenience of a naive substring
    search over the whole file.
    """

    return "\n".join(
        line for line in sql.splitlines() if not line.lstrip().startswith("--")
    )


DDL = _ddl_only(SQL)


def test_creates_every_table_with_bp_prefix():
    for table in (
        "proc.bp_style_intent",
        "proc.bp_style_ingest_staging",
        "proc.bp_style_profile",
        "proc.bp_style_exemplar",
        "proc.bp_mailbox_binding",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in SQL, table


def test_ddl_is_transactional_and_idempotent():
    assert SQL.strip().startswith("BEGIN") and "COMMIT;" in SQL
    assert ROLLBACK.strip().startswith("BEGIN") and "COMMIT;" in ROLLBACK
    # Re-running must be a no-op, not an error.
    assert "ON CONFLICT (code) DO NOTHING" in SQL
    assert "ON CONFLICT (config_key) DO NOTHING" in SQL


def test_indexes_follow_the_ix_bp_convention():
    for index in (
        "ix_bp_style_ingest_staging_purge",
        "ix_bp_style_ingest_staging_batch",
        "ix_bp_style_profile_active",
        "ix_bp_style_profile_lookup",
        "ix_bp_style_exemplar_lookup",
        "ix_bp_mailbox_binding_user",
    ):
        assert index in SQL, index


# --- amendments agreed in Phase -1 -------------------------------------------------

def test_no_tenant_id_anywhere():
    """This platform has no tenant dimension; a constant column would imply isolation
    that does not exist. See docs/style-engine/existing-email-inventory.md section 2."""
    assert "tenant_id" not in DDL.lower()


def test_identity_is_user_ref_not_persona_id():
    """'Persona' already means 'report voice' in proc.bp_prompt. Two meanings would be
    misread, so the new tables key on the Cognito sub as user_ref."""
    assert "persona_id" not in DDL.lower()
    assert "user_ref" in DDL


def test_embeddings_use_the_platform_model_dimension():
    """1024 = BAAI/bge-large-en-v1.5, already the platform standard (config/settings.py).
    A second embedding model would make profiles compiled under one and retrieved under
    another return confident nonsense, silently."""
    assert "CREATE EXTENSION IF NOT EXISTS vector" in DDL
    assert DDL.count("vector(1024)") == 2  # exemplar + staging
    assert "embedding_model" in DDL


def test_exemplar_embeddings_are_indexed_for_cosine_search():
    assert "USING hnsw (embedding vector_cosine_ops)" in DDL
    assert "ix_bp_style_exemplar_embedding" in DDL


def test_vectors_live_on_the_row_they_describe_not_in_a_second_store():
    """Lifecycle, not preference. These vectors are derived from correspondence, so they
    must die exactly when their row dies. As a column that is automatic; in Qdrant it is a
    second delete against a second service that can fail silently, leaving a lossy copy of
    a customer's email after we told them it was purged."""
    # The staging vector must sit on the staging row, so the purge DELETE takes it too.
    staging = DDL[DDL.index("bp_style_ingest_staging"):DDL.index("bp_style_profile")]
    assert "embedding" in staging
    assert "purge_after" in staging
    assert "qdrant" not in DDL.lower()


def test_keys_are_identity_columns_not_gen_random_uuid():
    """pgcrypto is not installed on this cluster and every existing bp_ table keys this way."""
    assert "gen_random_uuid" not in DDL
    assert "BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY" in DDL


# --- invariants enforced structurally ----------------------------------------------

def test_only_one_active_profile_per_scope():
    """Invariant 5/6 rest on this: a transaction that fails to stand the previous active
    version down cannot commit, which is what makes approval atomic rather than hopeful."""
    assert "CREATE UNIQUE INDEX IF NOT EXISTS ix_bp_style_profile_active" in SQL
    assert "ON proc.bp_style_profile (user_ref, intent) WHERE is_active" in SQL


def test_active_profile_must_be_approved_and_have_an_approver():
    """Invariant 6 — nothing auto-activates, and an approval names who did it."""
    assert "ck_bp_style_profile_active_is_approved" in SQL
    assert "CHECK (NOT is_active OR state = 'APPROVED')" in SQL
    assert "ck_bp_style_profile_approved_has_approver" in SQL


def test_profile_states_are_constrained():
    assert "CHECK (state IN ('UNCOMPILED','DRAFT','APPROVED','SUPERSEDED'))" in SQL


def test_credential_ref_must_be_a_secrets_manager_arn():
    """Invariant 8 — a token or password cannot be stored here even by mistake."""
    assert "ck_bp_mailbox_binding_credential_is_arn" in SQL
    assert "CHECK (credential_ref LIKE 'arn:aws:secretsmanager:%')" in SQL


def test_binding_cannot_activate_without_scope_proof():
    """A binding is only live once a read against a control mailbox was DENIED."""
    assert "ck_bp_mailbox_binding_active_needs_scope_proof" in SQL
    assert "CHECK (NOT is_active OR scope_verified_at IS NOT NULL)" in SQL


def test_staging_carries_a_purge_deadline():
    """Invariant 9 — staging is a queue, not a store."""
    assert "purge_after" in SQL
    assert "INTERVAL '24 hours'" in SQL


# --- vocabulary --------------------------------------------------------------------

def test_seeds_the_merged_intent_vocabulary():
    """One vocabulary, not two: the specified codes plus the ones EmailDraftingAgent
    already emits in production."""
    specified = (
        "rfq_invite", "clarification_request", "negotiation_counter",
        "award_notification", "supplier_rejection", "escalation",
        "contract_variation", "exit_notification", "internal_update",
    )
    already_live = ("follow_up", "reminder", "thank_you")
    for code in specified + already_live:
        assert f"'{code}'" in SQL, code


def test_seeds_the_all_scope_sentinel():
    """The user-level profile is the primary artifact. It needs a real code so it can
    carry a foreign key and take part in the one-active-per-scope unique index — a NULL
    intent could not, since Postgres treats NULLs as distinct."""
    assert "'_all'" in SQL


def test_legacy_codes_are_mapped_for_the_existing_normaliser():
    assert "legacy_code" in SQL
    for legacy in ("'rfq'", "'clarification'", "'negotiation'", "'award'", "'update'"):
        assert legacy in SQL, legacy


# --- one drafting path (invariant 10) ----------------------------------------------

def test_extends_the_existing_draft_table_rather_than_creating_a_second_one():
    """Two draft tables means two provenance stories and only one of them is correct."""
    assert "CREATE TABLE IF NOT EXISTS proc.bp_style_draft" not in SQL
    assert "ALTER TABLE proc.draft_rfq_emails" in SQL


def test_draft_provenance_is_complete():
    for col in (
        "style_user_ref", "style_intent", "style_mode",
        "style_profile_id", "style_profile_version", "style_fallback_level",
        "style_exemplar_ids", "style_exemplar_set_hash", "style_mailbox_binding_id",
        "style_message_ids", "style_retrieved_at", "style_model_id",
        "style_prompt_version", "external_draft_ref",
    ):
        assert col in SQL, col


def test_fallback_level_is_range_constrained():
    """Invariant 4 — a level outside 0-3 would mean the ladder was bypassed."""
    assert "CHECK (style_fallback_level BETWEEN 0 AND 3)" in SQL


def test_provenance_columns_are_nullable():
    """Hand-typed drafts from POST /workflows/email/prepare have no style provenance and
    must not be made to look as though they do."""
    for line in SQL.splitlines():
        if "ADD COLUMN IF NOT EXISTS style_" in line:
            assert "NOT NULL" not in line, line


# --- configuration -----------------------------------------------------------------

def test_config_seeds_into_the_existing_admin_store():
    assert "INSERT INTO proc.bp_admin_config" in SQL
    assert "'style_engine'" in SQL
    assert '"deployment_mode": "A"' in SQL
    assert '"min_exemplars": 3' in SQL


# --- rollback ----------------------------------------------------------------------

def test_rollback_leaves_the_vector_extension_installed():
    """Dropping it would cascade into any other pgvector column added since, and an unused
    extension costs nothing."""
    assert "DROP EXTENSION" not in _ddl_only(ROLLBACK)


def test_rollback_drops_everything_the_migration_created():
    for table in (
        "proc.bp_mailbox_binding",
        "proc.bp_style_exemplar",
        "proc.bp_style_profile",
        "proc.bp_style_ingest_staging",
        "proc.bp_style_intent",
    ):
        assert f"DROP TABLE IF EXISTS {table}" in ROLLBACK, table
    assert "DELETE FROM proc.bp_admin_config WHERE config_key = 'style_engine'" in ROLLBACK


def test_rollback_drops_dependent_columns_before_their_tables():
    """A DROP TABLE would fail while draft_rfq_emails still references it."""
    alter_at = ROLLBACK.index("ALTER TABLE proc.draft_rfq_emails")
    drop_at = ROLLBACK.index("DROP TABLE IF EXISTS proc.bp_style_profile")
    assert alter_at < drop_at


def test_rollback_leaves_the_pre_existing_draft_table_standing():
    """draft_rfq_emails predates this work and holds live rows."""
    assert "DROP TABLE IF EXISTS proc.draft_rfq_emails" not in ROLLBACK
    assert "TRUNCATE" not in ROLLBACK.upper()


def test_rollback_warns_that_approvals_are_not_recoverable():
    assert "DESTRUCTIVE" in ROLLBACK
