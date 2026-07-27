# tests/sql/test_bp_style_feedback_sql.py
#
# Phase 7. The schema's job here is mostly to make a mistake impossible: there must be no
# column that could hold an email body, and no way for a suggestion to act on itself.
from pathlib import Path

SQL = Path("deploy/sql/2026-07-27_bp_style_feedback.sql").read_text()
ROLLBACK = Path("deploy/sql/2026-07-27_bp_style_feedback_rollback.sql").read_text()


def _ddl_only(sql: str) -> str:
    return "\n".join(l for l in sql.splitlines() if not l.lstrip().startswith("--"))


DDL = _ddl_only(SQL)


def test_creates_both_tables_with_the_bp_prefix():
    assert "CREATE TABLE IF NOT EXISTS proc.bp_style_divergence" in DDL
    assert "CREATE TABLE IF NOT EXISTS proc.bp_style_recompile_suggestion" in DDL


def test_ddl_is_transactional_and_idempotent():
    assert SQL.strip().startswith("BEGIN") and "COMMIT;" in SQL
    assert ROLLBACK.strip().startswith("BEGIN") and "COMMIT;" in ROLLBACK
    assert DDL.count("IF NOT EXISTS") >= 2


def test_no_column_could_hold_an_email_body():
    """The claim that makes this loop acceptable. A future 'sent_body TEXT' should fail
    here before it ships."""
    for forbidden in ("sent_body", "sent_text", "body ", "content", "message_body"):
        assert forbidden not in DDL.lower(), forbidden


def test_the_score_is_bounded():
    assert "CHECK (score >= 0 AND score <= 1)" in DDL


def test_a_draft_is_observed_at_most_once():
    """A second row would double-count that draft in every average built from this table."""
    assert "UNIQUE (draft_id)" in DDL


def test_only_one_suggestion_is_open_per_scope():
    """Otherwise every sweep raises another and the user is nagged by a queue."""
    assert "ix_bp_style_suggestion_open" in DDL
    assert "(user_ref, intent) WHERE status = 'pending'" in DDL


def test_a_suggestion_can_only_be_pending_accepted_or_dismissed():
    assert "CHECK (status IN ('pending', 'accepted', 'dismissed'))" in DDL
    assert "'applied'" not in DDL, "there is no state in which the system acts by itself"


def test_an_actioned_suggestion_names_who_actioned_it():
    assert "ck_bp_style_suggestion_actioned_has_actor" in DDL


def test_divergence_rows_die_with_their_draft():
    assert "REFERENCES proc.draft_rfq_emails(id) ON DELETE CASCADE" in DDL


def test_indexes_follow_the_convention():
    for index in ("ix_bp_style_divergence_scope", "ix_bp_style_divergence_profile",
                  "ix_bp_style_suggestion_open", "ix_bp_style_suggestion_status"):
        assert index in DDL, index


def test_rollback_drops_both_tables():
    assert "DROP TABLE IF EXISTS proc.bp_style_recompile_suggestion" in ROLLBACK
    assert "DROP TABLE IF EXISTS proc.bp_style_divergence" in ROLLBACK


def test_rollback_leaves_the_draft_table_standing():
    assert "DROP TABLE IF EXISTS proc.draft_rfq_emails" not in ROLLBACK
