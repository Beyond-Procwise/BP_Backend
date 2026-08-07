"""Type detection from magic bytes, never from the declared type or extension.

The load-bearing case is the OOXML/ZIP one: libmagic reports a .docx as
application/zip, because a .docx IS a zip. Trusting libmagic alone would send
every Word document to the archive quarantine, and trusting the declared type
would let a zip through as a document. Both are wrong; container inspection
resolves it.
"""
import io
import zipfile

from src.services.mail_intake import typing_


def _ooxml_bytes(inner_path: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr(inner_path, "<x/>")
    return buf.getvalue()


def _plain_zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("quote.txt", "not really a pdf")
    return buf.getvalue()


def test_pdf_detected_from_magic_bytes():
    assert typing_.detect(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n") == "application/pdf"


def test_docx_is_not_reported_as_a_bare_zip():
    detected = typing_.detect(_ooxml_bytes("word/document.xml"))
    assert detected == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_xlsx_is_not_reported_as_a_bare_zip():
    detected = typing_.detect(_ooxml_bytes("xl/workbook.xml"))
    assert detected == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


def test_plain_zip_stays_a_zip():
    assert typing_.detect(_plain_zip_bytes()) == "application/zip"


def test_declared_type_is_never_trusted():
    # Acceptance test 6: a ZIP payload wearing a .pdf extension.
    detected = typing_.detect(_plain_zip_bytes())
    assert detected != "application/pdf"
    assert not typing_.is_permitted_alias(
        "application/pdf", detected, "quote.pdf"
    )


def test_benign_alias_is_permitted():
    # Plenty of senders label a .xlsx as octet-stream. That is sloppiness, not
    # evasion, and it must not quarantine a legitimate workbook.
    assert typing_.is_permitted_alias(
        "application/octet-stream",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "prices.xlsx",
    )


def test_exact_match_is_always_permitted():
    assert typing_.is_permitted_alias(
        "application/pdf", "application/pdf", "quote.pdf"
    )


def test_unrecognised_content_returns_octet_stream():
    assert typing_.detect(b"\x00\x01\x02not a known format") == (
        "application/octet-stream"
    )
