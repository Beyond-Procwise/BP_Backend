"""Every DB write for mail intake. Tests drive the shared FakeCursor from
conftest, so the suite never needs a database.

Two invariants get their own tests because both are silent when broken: a state
transition must be validated before it is written, and nothing that reaches a
log may carry body text, credentials or attachment content.
"""
import pytest

from src.services.mail_intake import repo
from src.services.mail_intake.states import (
    ArtifactState,
    BundleState,
    IllegalTransition,
    QuarantineReason,
)

from .conftest import FakeCursor


def test_insert_bundle_writes_received_state_and_returns_an_id():
    cur = FakeCursor([("11111111-1111-1111-1111-111111111111",)])
    bundle_id = repo.insert_bundle(
        cur,
        binding_id=1,
        owner_ref="user-7",
        provider_message_id="prov-1",
        rfc_message_id="<a@b.test>",
        thread_id=None,
        sender_address="quotes@copperleaf.test",
        sender_display_name="Sales",
        sender_domain="copperleaf.test",
        recipients=["intake@acme.procureiq.io"],
        subject="Quote",
        received_at=None,
        body_text="body",
        body_html=None,
        raw_mime_key="mail-intake/raw/user-7/x.eml",
    )
    assert bundle_id
    sql, params = cur.calls[0]
    assert "INSERT INTO proc.bp_ingest_bundle" in sql
    assert BundleState.RECEIVED.value in params


def test_insert_bundle_returns_none_when_the_message_was_already_ingested():
    """The idempotency contract. ON CONFLICT DO NOTHING returns no row, and the
    caller must be able to tell that apart from a successful insert — otherwise
    a re-delivered event proceeds against a bundle_id that has no row behind
    it, every later UPDATE matches nothing, and the artifact inserts break
    their foreign key."""
    cur = FakeCursor()  # no row returned => conflict
    assert (
        repo.insert_bundle(
            cur,
            binding_id=1,
            owner_ref="user-7",
            provider_message_id="prov-1",
            rfc_message_id="<a@b.test>",
            thread_id=None,
            sender_address="quotes@copperleaf.test",
            sender_display_name="Sales",
            sender_domain="copperleaf.test",
            recipients=[],
            subject="Quote",
            received_at=None,
            body_text=None,
            body_html=None,
            raw_mime_key="mail-intake/raw/user-7/x.eml",
        )
        is None
    )


def test_transition_validates_before_writing():
    cur = FakeCursor()
    with pytest.raises(IllegalTransition):
        repo.transition_artifact(
            cur, "ART-1", ArtifactState.TYPED, ArtifactState.STORED
        )
    assert cur.calls == []


def test_legal_transition_writes_state_and_stamps_updated_at():
    cur = FakeCursor()
    repo.transition_artifact(
        cur, "ART-1", ArtifactState.GATED, ArtifactState.STORED
    )
    sql, params = cur.calls[0]
    assert "UPDATE proc.bp_ingest_artifact" in sql
    assert "updated_at = now()" in sql
    assert ArtifactState.STORED.value in params


def test_quarantine_transition_records_its_reason():
    cur = FakeCursor()
    repo.transition_artifact(
        cur,
        "ART-1",
        ArtifactState.TYPED,
        ArtifactState.QUARANTINED,
        reason=QuarantineReason.MACRO,
        detail={"container": "ooxml"},
    )
    sql, params = cur.calls[0]
    assert QuarantineReason.MACRO.value in params
    assert "quarantine_reason" in sql


def test_claim_ready_artifacts_selects_only_ready_rows():
    # Row sets are queued as a list, always — see FakeCursor.fetchall.
    cur = FakeCursor([[{"artifact_id": "ART-1"}]])
    rows = repo.claim_ready_artifacts(cur, limit=10)
    sql, params = cur.calls[0]
    # Assert the FILTER, not merely that the value was bound. Checking only the
    # parameter would keep passing if a future edit dropped `WHERE state = %s`
    # from the SQL and left the value as a stray binding — which is precisely
    # the regression this test exists to catch.
    assert "WHERE state = %s" in sql
    assert ArtifactState.READY_FOR_EXTRACTION.value in params
    assert rows == [{"artifact_id": "ART-1"}]


def test_derivation_is_append_only():
    cur = FakeCursor()
    repo.append_derivation(cur, "ART-1", "v3", "run-9")
    sql, _ = cur.calls[0]
    assert "INSERT INTO proc.bp_artifact_derivation" in sql
    assert "UPDATE" not in sql
    assert "ON CONFLICT" not in sql


def test_find_existing_artifact_scopes_to_the_owner():
    cur = FakeCursor([("ART-7",)])
    assert repo.find_existing_artifact(cur, "user-7", "abc", None) == "ART-7"
    sql, params = cur.calls[0]
    # The owner predicate must be IN THE SQL. Asserting only the bound value
    # would pass even if the scoping clause were removed — and losing it means
    # one customer's content hash matching another's, which is the worst
    # failure this function can have.
    assert "WHERE owner_ref = %s" in sql
    assert "user-7" in params


def test_a_quarantined_copy_is_never_the_original_a_clean_arrival_duplicates():
    """Dedup runs after the safety gate, so the arrival being checked is clean.
    Matching it to a QUARANTINED artifact marked it a duplicate of something
    that was never read: a clean re-send of a once-quarantined file (a false
    positive since fixed, or one a person released) was lost with no trace.
    The predicate must be in the SQL, as with owner scoping above."""
    cur = FakeCursor([])
    repo.find_existing_artifact(cur, "user-7", "abc", None)
    sql, params = cur.calls[0]
    # Only an original that was handed to extraction counts — the same set the
    # unique index ix_bp_ingest_artifact_owner_sha covers (2026-10-14 migration).
    assert "AND state = ANY(%s)" in sql
    assert ["ready_for_extraction", "extraction_complete"] in params


def test_mark_duplicate_validates_the_state_it_is_given():
    """The transition check must consult the artifact's actual state.

    An earlier version passed two hardcoded literals to assert_transition, so
    it could never raise whatever state the row was really in — it looked
    exactly like the sibling functions that do validate, while validating
    nothing.
    """
    cur = FakeCursor()
    with pytest.raises(IllegalTransition):
        repo.mark_duplicate(cur, "ART-1", "ART-2", current=ArtifactState.TYPED)
    assert cur.calls == []


def test_mark_duplicate_guards_against_a_concurrent_transition():
    cur = FakeCursor()
    repo.mark_duplicate(cur, "ART-1", "ART-2", current=ArtifactState.STORED)
    sql, params = cur.calls[0]
    assert "AND state = %s" in sql
    assert ArtifactState.STORED.value in params


def test_a_flat_multi_row_queue_is_rejected_rather_than_silently_truncated():
    """The footgun the previous fake had.

    Written flat — the style used everywhere else in this suite — a multi-row
    fetchall returned only the first row and left the rest to corrupt the next
    call, silently. Raising costs nothing and the message says what to write.
    """
    cur = FakeCursor([("S1", "acme.test"), ("S2", "other.test")])
    with pytest.raises(TypeError, match="LIST of rows"):
        cur.fetchall()


def test_a_transition_that_matched_no_row_is_reported_not_silent(caplog):
    """A guarded UPDATE matching zero rows means another writer moved the row
    first. Returning None made that indistinguishable from success."""
    cur = FakeCursor()
    cur.rowcount = 0
    with caplog.at_level("WARNING"):
        applied = repo.transition_artifact(
            cur, "ART-1", ArtifactState.GATED, ArtifactState.STORED
        )
    assert applied is False
    assert any("transition_skipped" in r.getMessage() for r in caplog.records)


def test_safe_log_fields_drops_content_and_keeps_correlation_keys():
    fields = repo.safe_log_fields(
        {
            "bundle_id": "B-1",
            "rfc_message_id": "<a@b.test>",
            "state": "parsed",
            "body_text": "our bank details have changed",
            "body_html": "<p>secret</p>",
            "subject": "Quote QT-1",
            "sender_address": "quotes@copperleaf.test",
            "raw_mime_key": "mail-intake/raw/user-7/x.eml",
        }
    )
    assert fields["rfc_message_id"] == "<a@b.test>"
    assert "body_text" not in fields
    assert "body_html" not in fields
    assert "subject" not in fields
    blob = repr(fields).lower()
    assert "bank details" not in blob
    assert "secret" not in blob


def test_safe_log_fields_redacts_addresses_outside_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    fields = repo.safe_log_fields(
        {"bundle_id": "B-1", "sender_address": "quotes@copperleaf.test"}
    )
    assert fields["sender_address"] != "quotes@copperleaf.test"
    assert "copperleaf.test" in fields["sender_address"]


@pytest.mark.parametrize(
    "rows",
    [
        pytest.param([("S1", "acme.test")], id="tuple-cursor"),
        pytest.param(
            [{"supplier_id": "S1", "domain": "acme.test"}], id="realdict-cursor"
        ),
    ],
)
def test_supplier_candidates_reads_either_cursor_shape(rows):
    """The live path uses a RealDictCursor, and this indexed positionally.

    It raised KeyError: 0 — and `supplier_bind` catches every exception and
    returns "unmatched", so the failure was invisible: every sender would have
    silently failed to match a supplier while the unmatched-domain rate, which
    is watched as a security signal, read as real.
    """
    cur = FakeCursor([rows])
    assert repo.supplier_candidates(cur, "acme.test") == [("S1", "acme.test")]
