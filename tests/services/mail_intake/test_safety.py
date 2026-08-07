# tests/services/mail_intake/test_safety.py
"""The gate. Everything here fails closed: an unsure verdict quarantines.

Nothing in this module may open, execute or expand its input. The macro test
in particular must pass without the workbook ever being handed to openpyxl.
"""
import pathlib

from src.services.mail_intake import safety, typing_
from src.services.mail_intake.states import QuarantineReason

FIXTURES = pathlib.Path("tests/fixtures/mail_intake")


def _verdict(name, declared, filename=None, ceiling=safety.DEFAULT_SIZE_CEILING_BYTES):
    content = (FIXTURES / name).read_bytes()
    return safety.evaluate(
        content,
        declared_mime=declared,
        detected_mime=typing_.detect(content),
        filename=filename or name,
        size_ceiling_bytes=ceiling,
    )


def test_clean_pdf_passes():
    assert _verdict("clean.pdf", "application/pdf").ok


def test_password_protected_pdf_quarantines():
    verdict = _verdict("encrypted.pdf", "application/pdf")
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ENCRYPTED


def test_macro_workbook_quarantines():
    verdict = _verdict(
        "macro_workbook.xlsm",
        "application/vnd.ms-excel.sheet.macroEnabled.12",
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.MACRO


def test_password_protected_workbook_quarantines():
    verdict = _verdict(
        "encrypted_workbook.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ENCRYPTED


def test_document_skeleton_wrapped_around_a_payload_quarantines():
    """Type detection alone is foolable, and this is the case that fools it.

    The container holds a genuine minimal Word document AND an executable, so
    it types as a Word document rather than an archive — passing a gate that
    would have quarantined the identical payload arriving as a plain .zip.
    """
    verdict = _verdict(
        "smuggling_document.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ARCHIVE
    assert "payload.exe" not in repr(verdict.detail)  # structural facts only
    assert verdict.detail["foreign_members"]


def test_ordinary_workbook_is_not_falsely_flagged():
    """The false-positive guard for both Office heuristics.

    A macro or encryption detector that fires on ordinary workbooks would fill
    the review queue with legitimate price lists, and a queue of false
    positives is one people stop reading.
    """
    verdict = _verdict(
        "clean_workbook.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    assert verdict.ok, f"clean workbook wrongly quarantined as {verdict.reason}"


def test_zip_wearing_a_pdf_name_quarantines_on_mismatch():
    verdict = _verdict("zip_named_quote.pdf", "application/pdf")
    assert not verdict.ok
    # Both readings are defensible; the gate must report the more specific one.
    assert verdict.reason in (
        QuarantineReason.TYPE_MISMATCH,
        QuarantineReason.ARCHIVE,
    )


def test_archive_is_never_expanded():
    content = (FIXTURES / "zip_named_quote.pdf").read_bytes()
    verdict = safety.evaluate(
        content,
        declared_mime="application/zip",
        detected_mime="application/zip",
        filename="quotes.zip",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.ARCHIVE


def test_oversize_quarantines():
    verdict = _verdict("clean.pdf", "application/pdf", ceiling=10)
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.OVERSIZE


def test_executable_quarantines_regardless_of_extension():
    verdict = safety.evaluate(
        b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 64,
        declared_mime="application/pdf",
        detected_mime="application/x-executable",
        filename="quote.pdf",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.EXECUTABLE


def test_shell_script_quarantines():
    verdict = safety.evaluate(
        b"#!/bin/sh\necho hi\n",
        declared_mime="text/plain",
        detected_mime="text/x-shellscript",
        filename="notes.txt",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.EXECUTABLE


def test_unknown_type_quarantines():
    verdict = safety.evaluate(
        b"\x00\x01\x02\x03",
        declared_mime="application/pdf",
        detected_mime="application/octet-stream",
        filename="mystery.pdf",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert not verdict.ok
    assert verdict.reason is QuarantineReason.UNKNOWN_TYPE


def test_benign_mislabel_passes():
    content = (FIXTURES / "clean.pdf").read_bytes()
    verdict = safety.evaluate(
        content,
        declared_mime="application/octet-stream",
        detected_mime="application/pdf",
        filename="quote.pdf",
        size_ceiling_bytes=safety.DEFAULT_SIZE_CEILING_BYTES,
    )
    assert verdict.ok


def test_detail_never_carries_document_content():
    verdict = _verdict("encrypted.pdf", "application/pdf")
    blob = repr(verdict.detail).lower()
    assert "confidential" not in blob
