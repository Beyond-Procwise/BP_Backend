"""Spec §10 acceptance cases in Plan 1 scope, each named for its case number.

These duplicate coverage that exists in the unit suites, deliberately. A unit
test proves a module behaves; these prove the SYSTEM does the thing the spec
promised, through the real pipeline, from committed raw MIME.
"""
import pathlib
from datetime import datetime, timezone
from email.message import EmailMessage

from src.services.mail_intake.pipeline import IngestPipeline
from src.services.mail_intake.sources.base import RawMessage
from src.services.mail_intake.states import QuarantineReason

from .conftest import FakeCursor, FakeStore

FIXTURES = pathlib.Path("tests/fixtures/mail_intake")


def _run(raw: bytes, provider_id="emails/x.eml"):
    # The queued row answers the bundle INSERT ... RETURNING. Without it the
    # pipeline takes the ON CONFLICT path and correctly does nothing, which
    # would make every assertion below pass or fail for the wrong reason.
    cur = FakeCursor().queue(("11111111-1111-1111-1111-111111111111",))
    store = FakeStore()
    outcome = IngestPipeline(object_store=store, bucket="procwisemvp").ingest(
        cur,
        {"binding_id": 1, "owner_ref": "user-7", "folder_scope": "emails/"},
        RawMessage(raw, provider_id, datetime(2026, 8, 5, tzinfo=timezone.utc), "ref"),
    )
    return outcome, cur, store


def _wrap(attachment: bytes, filename: str, subtype: str = "pdf") -> bytes:
    msg = EmailMessage()
    msg["From"] = "Sales <quotes@copperleaf.test>"
    msg["To"] = "intake@acme.procureiq.io"
    msg["Subject"] = "Quote"
    msg["Message-ID"] = f"<{filename}@copperleaf.test>"
    msg.set_content("Attached.")
    msg.add_attachment(
        attachment, maintype="application", subtype=subtype, filename=filename
    )
    return msg.as_bytes()


def test_case_01_simple_message_single_pdf():
    outcome, cur, store = _run((FIXTURES / "simple_pdf.eml").read_bytes())
    assert len(outcome.artifact_ids) == 1
    assert any(k.startswith("mail-intake/artifacts/") for k in store.objects)
    assert any("bp_ingest_sender_auth" in sql for sql, _ in cur.calls)


def test_case_02_quote_nested_two_levels_deep():
    outcome, cur, _ = _run((FIXTURES / "forwarded_two_levels.eml").read_bytes())
    assert len(outcome.artifact_ids) == 1
    inserts = [p for sql, p in cur.calls if "INSERT INTO proc.bp_ingest_artifact" in sql]
    assert any(2 in (p or ()) for p in inserts)


def test_case_03_cid_signature_logo_is_filtered():
    outcome, _, _ = _run((FIXTURES / "cid_signature_logo.eml").read_bytes())
    assert outcome.artifact_ids == ()


def test_case_04_non_cid_inline_document_is_ingested():
    outcome, _, _ = _run((FIXTURES / "inline_non_cid_document.eml").read_bytes())
    assert len(outcome.artifact_ids) == 1


def test_case_06_pdf_extension_over_zip_payload_quarantines():
    raw = _wrap((FIXTURES / "zip_named_quote.pdf").read_bytes(), "quote.pdf")
    outcome, _, store = _run(raw)
    assert outcome.artifact_ids == ()
    assert outcome.quarantined[0][1] in (
        QuarantineReason.TYPE_MISMATCH,
        QuarantineReason.ARCHIVE,
    )
    assert any(k.startswith("mail-intake/quarantine/") for k in store.objects)


def test_case_07_password_protected_pdf_is_retained_and_releasable():
    raw = _wrap((FIXTURES / "encrypted.pdf").read_bytes(), "quote.pdf")
    outcome, _, store = _run(raw)
    assert outcome.quarantined[0][1] is QuarantineReason.ENCRYPTED
    # Retained: a human can release it, so the bytes must still exist.
    assert any(k.startswith("mail-intake/quarantine/") for k in store.objects)


def test_case_08_macro_workbook_quarantines_and_is_never_opened():
    raw = _wrap(
        (FIXTURES / "macro_workbook.xlsm").read_bytes(),
        "prices.xlsm",
        subtype="vnd.ms-excel.sheet.macroEnabled.12",
    )
    outcome, _, _ = _run(raw)
    assert outcome.quarantined[0][1] is QuarantineReason.MACRO


def test_case_11_zero_opacity_injection_quarantines_before_extraction():
    raw = _wrap((FIXTURES / "injection_zero_opacity.pdf").read_bytes(), "quote.pdf")
    outcome, _, store = _run(raw)
    assert outcome.artifact_ids == ()
    assert outcome.quarantined[0][1] is QuarantineReason.HIDDEN_TEXT
    # No extracted value can reach a Finding, because nothing was ever emitted.
    assert not any(k.startswith("mail-intake/artifacts/") for k in store.objects)


def test_case_14_malformed_mime_quarantines_with_raw_retained():
    outcome, _, store = _run((FIXTURES / "malformed.eml").read_bytes())
    assert outcome.artifact_ids == ()
    assert any(k.startswith("mail-intake/raw/") for k in store.objects)


def test_case_17_body_terms_and_attachment_pricing_share_one_bundle():
    outcome, cur, _ = _run((FIXTURES / "body_validity_window.eml").read_bytes())
    assert len(outcome.artifact_ids) == 1
    # The body arrives via UPDATE proc.bp_ingest_bundle (repo.update_bundle_
    # from_parse), not the initial INSERT: the bundle row is created — with
    # only binding_id/owner_ref/provider_message_id/raw_mime_key — before the
    # message is parsed, so that raw MIME is safely stored and the idempotency
    # check has already run before parsing can fail. What parsing learns,
    # including body_text, is filled in afterwards by an UPDATE against the
    # same bundle_id. Searching the INSERT for body text would pass or fail
    # for the wrong reason (it never carries body_text at all).
    bundle_update = next(
        p for sql, p in cur.calls if "UPDATE proc.bp_ingest_bundle" in sql
    )
    assert any(
        isinstance(v, str) and "valid until 30 September 2026" in v
        for v in bundle_update
    )


def test_scanned_ocr_invoice_is_not_quarantined():
    """Not a numbered case, but the production failure mode this feature most
    likely dies from. See spec §8.1."""
    raw = _wrap((FIXTURES / "scanned_ocr_layer.pdf").read_bytes(), "invoice.pdf")
    outcome, _, _ = _run(raw)
    assert len(outcome.artifact_ids) == 1
    assert not outcome.quarantined
