"""End-to-end orchestration over a fake cursor and a fake object store.

The tests that matter most here are the negative ones: a malformed message must
produce a quarantined bundle and NO artifacts, and a gate rejection must leave
the artifact unreachable from the emit queue. Fail-closed is only real if it is
observable.
"""
import pathlib
from datetime import datetime, timezone

import pytest

from src.services.mail_intake import pipeline as pipeline_module
from src.services.mail_intake.pipeline import IngestPipeline
from src.services.mail_intake.sources.base import RawMessage
from src.services.mail_intake.states import ArtifactState, BundleState, QuarantineReason

from .conftest import FakeCursor, FakeStore

FIXTURES = pathlib.Path("tests/fixtures/mail_intake")


class FakeBus:
    def __init__(self):
        self.published = []

    def publish(self, name, payload=None):
        self.published.append((name, payload))


BUNDLE_UUID = "11111111-1111-1111-1111-111111111111"


def _seeded_cursor():
    """A cursor whose first fetchone answers the bundle INSERT ... RETURNING.

    Order matters and is worth stating: insert_bundle fetches first (and gets
    this row), supplier_candidates then fetchall's an empty queue (no supplier
    match), and each artifact's duplicate lookup fetchone's an empty queue
    (returns None, i.e. not a duplicate). A bare FakeCursor with nothing queued
    therefore models the ON CONFLICT case — see the re-delivery test.
    """
    return FakeCursor().queue((BUNDLE_UUID,))


def _message(fixture):
    return RawMessage(
        raw=(FIXTURES / fixture).read_bytes(),
        provider_message_id=f"emails/{fixture}",
        received_at=datetime(2026, 8, 5, 9, 0, tzinfo=timezone.utc),
        source_ref=f"s3://procwisemvp/emails/{fixture}",
    )


def _binding():
    return {"binding_id": 1, "owner_ref": "user-7", "folder_scope": "emails/"}


def _pipeline(store=None, bus=None):
    return IngestPipeline(
        object_store=store or FakeStore(),
        bucket="procwisemvp",
        event_bus=bus,
    )


def test_simple_message_produces_one_artifact_with_provenance():
    cur, store = _seeded_cursor(), FakeStore()
    outcome = _pipeline(store).ingest(cur, _binding(), _message("simple_pdf.eml"))

    assert len(outcome.artifact_ids) == 1
    assert not outcome.quarantined
    assert cur.statements("INSERT INTO proc.bp_ingest_bundle")
    assert cur.statements("INSERT INTO proc.bp_ingest_sender_auth")
    assert cur.statements("INSERT INTO proc.bp_ingest_arrival")
    # Raw MIME is persisted before anything is parsed.
    assert any(k.startswith("mail-intake/raw/") for k in store.objects)


def test_raw_mime_is_stored_even_when_parsing_fails():
    cur, store = _seeded_cursor(), FakeStore()
    outcome = _pipeline(store).ingest(cur, _binding(), _message("malformed.eml"))

    assert outcome.artifact_ids == ()
    assert any(k.startswith("mail-intake/raw/") for k in store.objects)
    assert any(
        QuarantineReason.MALFORMED_MIME.value in (p or ())
        for p in cur.states_written()
    )


def test_nested_quote_is_ingested_with_its_depth():
    cur = _seeded_cursor()
    outcome = _pipeline().ingest(cur, _binding(), _message("forwarded_two_levels.eml"))
    assert len(outcome.artifact_ids) == 1
    inserts = cur.statements("INSERT INTO proc.bp_ingest_artifact")
    assert any(2 in (params or ()) for _sql, params in inserts)


def test_cid_logo_never_becomes_an_artifact():
    cur = _seeded_cursor()
    outcome = _pipeline().ingest(cur, _binding(), _message("cid_signature_logo.eml"))
    assert outcome.artifact_ids == ()
    # Dropped, not quarantined. An empty artifact_ids alone would also be
    # satisfied by a logo that reached the gate and was held there.
    assert outcome.quarantined == ()
    assert not cur.statements("INSERT INTO proc.bp_ingest_artifact")


def test_non_cid_inline_document_is_ingested():
    cur = _seeded_cursor()
    outcome = _pipeline().ingest(
        cur, _binding(), _message("inline_non_cid_document.eml")
    )
    assert len(outcome.artifact_ids) == 1


def test_body_and_attachment_land_in_one_bundle():
    # The bundle is inserted before parsing (idempotency first), so the body
    # text parsing discovers arrives via update_bundle_from_parse's UPDATE,
    # not the initial INSERT. Checking every UPDATE to bp_ingest_bundle finds
    # it regardless of which one carries it.
    cur = _seeded_cursor()
    _pipeline().ingest(cur, _binding(), _message("body_validity_window.eml"))
    updates = cur.statements("UPDATE proc.bp_ingest_bundle")
    assert any(
        isinstance(p, str) and "valid until 30 September 2026" in p
        for _sql, params in updates
        for p in (params or ())
    )


def test_gate_rejection_never_reaches_ready_for_extraction():
    cur, store = _seeded_cursor(), FakeStore()
    # Build a message whose attachment is a macro workbook.
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = "Sales <quotes@copperleaf.test>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Prices"
    msg["Message-ID"] = "<macro@copperleaf.test>"
    msg.set_content("Attached.")
    msg.add_attachment(
        (FIXTURES / "macro_workbook.xlsm").read_bytes(),
        maintype="application",
        subtype="vnd.ms-excel.sheet.macroEnabled.12",
        filename="prices.xlsm",
    )
    message = RawMessage(msg.as_bytes(), "emails/macro.eml", None, "ref")

    outcome = _pipeline(store).ingest(cur, _binding(), message)

    assert outcome.artifact_ids == ()
    assert outcome.quarantined
    assert outcome.quarantined[0][1] is QuarantineReason.MACRO
    written = [p for p in cur.states_written()]
    assert not any(
        ArtifactState.READY_FOR_EXTRACTION.value in (p or ()) for p in written
    )
    # Quarantined bytes go to the quarantine prefix, not the artifact prefix.
    assert any(k.startswith("mail-intake/quarantine/") for k in store.objects)
    assert not any(k.startswith("mail-intake/artifacts/") for k in store.objects)


def test_hidden_text_quarantines_the_artifact():
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = "Sales <quotes@copperleaf.test>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Quote"
    msg["Message-ID"] = "<inject@copperleaf.test>"
    msg.set_content("Attached.")
    msg.add_attachment(
        (FIXTURES / "injection_zero_opacity.pdf").read_bytes(),
        maintype="application",
        subtype="pdf",
        filename="quote.pdf",
    )
    cur = _seeded_cursor()
    outcome = _pipeline().ingest(
        cur, _binding(), RawMessage(msg.as_bytes(), "emails/inject.eml", None, "ref")
    )
    assert outcome.artifact_ids == ()
    assert outcome.quarantined[0][1] is QuarantineReason.HIDDEN_TEXT


def test_scanned_ocr_document_is_not_quarantined():
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = "Sales <quotes@copperleaf.test>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Scanned invoice"
    msg["Message-ID"] = "<scan@copperleaf.test>"
    msg.set_content("Attached.")
    msg.add_attachment(
        (FIXTURES / "scanned_ocr_layer.pdf").read_bytes(),
        maintype="application",
        subtype="pdf",
        filename="invoice.pdf",
    )
    cur = _seeded_cursor()
    outcome = _pipeline().ingest(
        cur, _binding(), RawMessage(msg.as_bytes(), "emails/scan.eml", None, "ref")
    )
    assert len(outcome.artifact_ids) == 1
    assert not outcome.quarantined


def test_quarantine_is_logged_so_an_operator_can_find_it(caplog):
    """A held document is something a person has to act on.

    "What did we block today, and why" must be answerable from the logs alone,
    without database access. The line carries a stable greppable prefix, the
    reason, the correlation ids, and the byte-derived detail.
    """
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = "Sales <quotes@copperleaf.test>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Prices"
    msg["Message-ID"] = "<macro-log@copperleaf.test>"
    msg.set_content("Attached.")
    msg.add_attachment(
        (FIXTURES / "macro_workbook.xlsm").read_bytes(),
        maintype="application",
        subtype="vnd.ms-excel.sheet.macroEnabled.12",
        filename="prices.xlsm",
    )

    with caplog.at_level("WARNING"):
        _pipeline().ingest(
            _seeded_cursor(),
            _binding(),
            RawMessage(msg.as_bytes(), "emails/macro-log.eml", None, "ref"),
        )

    lines = [r.getMessage() for r in caplog.records]
    quarantine_lines = [line for line in lines if "mail_intake.quarantine" in line]
    assert quarantine_lines, "a quarantine must be discoverable from the logs"
    assert "reason=macro" in quarantine_lines[0]
    # Redaction still applies: no filename, no document bytes.
    assert "prices.xlsm" not in quarantine_lines[0]


def test_redelivered_message_is_a_no_op():
    """Acceptance-relevant: re-delivery must not produce a second set of
    artifacts. The conflict on (binding_id, provider_message_id) is what makes
    IMAP re-polling and duplicate provider events safe."""
    cur, store = FakeCursor(), FakeStore()
    # No row returned from the bundle INSERT => already ingested.
    outcome = _pipeline(store).ingest(cur, _binding(), _message("simple_pdf.eml"))
    assert outcome.artifact_ids == ()
    assert not any("INSERT INTO proc.bp_ingest_artifact" in sql for sql, _ in cur.calls)
    assert store.objects == {}


def test_a_message_that_cannot_be_parsed_still_leaves_its_bytes_and_a_record():
    """The failure the store-before-parse ordering exists to prevent.

    A body part declaring an unknown charset raises LookupError out of the
    stdlib — ordinary in real mail, and not MalformedMime. An earlier ordering
    parsed first and let that escape, destroying the message: no bytes stored,
    no bundle row, no quarantine record, nothing anyone could investigate.

    The message must be multipart to exercise this: mime_walk.parse's
    non-multipart branch already wraps `get_content()` in a try/except and
    converts a bad charset to MalformedMime by itself (verified directly), so
    a single-part message never reaches the unguarded path at all. The
    unguarded call is `_walk()`'s body-part branch, reached only when the
    message has more than one part — hence the attachment below, which forces
    the multipart/mixed structure that actually exercises it.
    """
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"] = "Sales <quotes@copperleaf.test>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Quote"
    msg["Message-ID"] = "<charset@copperleaf.test>"
    msg.set_content("body")
    msg.add_attachment(
        b"%PDF-1.4\n", maintype="application", subtype="pdf", filename="x.pdf"
    )
    raw = msg.as_bytes().replace(
        b'charset="utf-8"', b'charset="totally-not-a-charset"', 1
    )

    cur, store_ = _seeded_cursor(), FakeStore()
    outcome = _pipeline(store_).ingest(
        cur, _binding(), RawMessage(raw, "emails/charset.eml", None, "ref")
    )

    assert outcome.artifact_ids == ()
    # The bytes survive, which is the whole point.
    assert any(k.startswith("mail-intake/raw/") for k in store_.objects)
    assert any("INSERT INTO proc.bp_ingest_bundle" in sql for sql, _ in cur.calls)
    assert any(
        QuarantineReason.MALFORMED_MIME.value in (p or ())
        for p in cur.states_written()
    )


def test_the_sweep_announces_without_writing_anything():
    """A safety net that writes on every pass is a leak, not a net.

    The sweep re-announces a ready artifact until a consumer moves it, so it
    runs against the same rows repeatedly by design. An earlier version appended
    a derivation row each pass — one a minute per artifact, unbounded, while no
    consumer existed. A derivation records that an extraction RUN happened, and
    this function does not run extraction.
    """
    cur, bus = FakeCursor(), FakeBus()
    cur.queue(
        [
            {
                "artifact_id": "ART-1",
                "bundle_id": "B-1",
                "object_key": "mail-intake/artifacts/user-7/ab/cd/abcd",
                "detected_mime": "application/pdf",
                "owner_ref": "user-7",
            }
        ]
    )
    pipeline_module.sweep_ready_artifacts(cur, event_bus=bus)

    assert not [
        sql for sql, _ in cur.calls
        if sql.strip().upper().startswith(("INSERT", "UPDATE", "DELETE"))
    ], "the sweep must announce, not write"


def test_sweep_publishes_ready_artifacts():
    cur, bus = FakeCursor(), FakeBus()
    cur.queue(
        [
            {
                "artifact_id": "ART-1",
                "bundle_id": "B-1",
                "object_key": "mail-intake/artifacts/user-7/ab/cd/abcd",
                "detected_mime": "application/pdf",
                "owner_ref": "user-7",
            }
        ]
    )
    count = pipeline_module.sweep_ready_artifacts(cur, event_bus=bus)
    assert count == 1
    assert bus.published[0][0] == pipeline_module.DOCUMENT_INGESTED
    assert bus.published[0][1]["artifact_id"] == "ART-1"


def test_emitted_payload_carries_no_document_content():
    cur, bus = FakeCursor(), FakeBus()
    cur.queue(
        [
            {
                "artifact_id": "ART-1",
                "bundle_id": "B-1",
                "object_key": "mail-intake/artifacts/user-7/ab/cd/abcd",
                "detected_mime": "application/pdf",
                "owner_ref": "user-7",
            }
        ]
    )
    pipeline_module.sweep_ready_artifacts(cur, event_bus=bus)
    payload = bus.published[0][1]
    assert "content" not in payload
    assert "body_text" not in payload
