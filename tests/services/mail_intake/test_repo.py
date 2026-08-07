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
    cur = FakeCursor([{"artifact_id": "ART-1"}])
    rows = repo.claim_ready_artifacts(cur, limit=10)
    sql, params = cur.calls[0]
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
    _, params = cur.calls[0]
    assert "user-7" in params


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
