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


def _ooxml_bytes(inner_path: str, extra: dict | None = None) -> bytes:
    """A well-formed minimal OOXML package, plus any extra members given.

    `_rels/.rels` is mandatory in the Open Packaging Conventions and is present
    in every real workbook, so the happy-path fixture must carry it — otherwise
    the fixture would not be a document by the definition the code uses.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr(inner_path, "<x/>")
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
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


# -- adversarial: a hostile sender must not obtain a document verdict --------
#
# These are the cases the module exists for. Absent them, the friendly tests
# above pass against an implementation that is trivially foolable.

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


def test_package_without_the_mandatory_relationships_part_is_not_a_document():
    """A document skeleton is not a document.

    _rels/.rels is mandatory in the Open Packaging Conventions. A container
    without it is a zip that happens to hold a document-shaped entry, and must
    be reported as the archive it is so the gate holds it for a person.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<x/>")
    assert typing_.detect(buf.getvalue()) == "application/zip"


def test_well_formed_package_carrying_a_payload_is_still_typed_but_gated():
    """Detection reports what a thing IS; the gate decides whether it is safe.

    A genuinely well-formed package that also carries a payload still types as
    a document here — that is the honest answer. Task 5's foreign-member check
    is what quarantines it, and this test pins the division of labour so a
    future change cannot quietly move it.
    """
    content = _ooxml_bytes("word/document.xml", {"payload.exe": b"MZ\x90\x00"})
    assert typing_.detect(content) == DOCX


def test_entry_names_escaping_the_package_are_never_a_document():
    """Path traversal in a member name is a deliberate construction.

    No real Office file contains one (verified across every workbook in this
    repo), so its presence alone disqualifies the container from a document
    verdict regardless of how well formed the rest of it looks.
    """
    for hostile_name in ("../../etc/passwd", "/etc/passwd", "..\\..\\windows\\x"):
        content = _ooxml_bytes("word/document.xml", {hostile_name: "x"})
        assert typing_.detect(content) == "application/zip", hostile_name


def test_octet_stream_alias_never_excuses_an_archive():
    """Senders label everything octet-stream, so the alias set is broad. It must
    still not launder an archive into an acceptable document."""
    assert not typing_.is_permitted_alias(
        "application/octet-stream", "application/zip", "quote.docx"
    )


def test_extension_never_influences_the_alias_verdict():
    """`filename` is accepted for call-site symmetry and ignored. The extension
    is exactly as untrustworthy as the declared type."""
    assert typing_.is_permitted_alias(
        "application/octet-stream", "application/pdf", "anything.exe"
    ) is typing_.is_permitted_alias(
        "application/octet-stream", "application/pdf", "quote.pdf"
    )


def test_package_libmagic_calls_octet_stream_is_still_recognised():
    """Reproduces a real 19% false-positive rate on this repo's Word documents.

    libmagic identifies an OOXML package by looking for recognisable parts near
    the start of the archive. LibreOffice writes its `[trash]` padding first, so
    magic reports `application/octet-stream` and — while inspection was keyed off
    magic's verdict — the file was never inspected, typed as unknown, and
    quarantined. Five of this repo's 26 real .docx files are shaped this way.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        # Order matters: the padding must come first, as LibreOffice writes it.
        zf.writestr("[trash]/0000.dat", b"\xff\xff\xff\xff" + b"\x00" * 308)
        zf.writestr("[trash]/0003.dat", b"\xff\xff\xff\xff" + b"\x00" * 270)
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("_rels/.rels", "<Relationships/>")
        zf.writestr("word/document.xml", "<document/>")
    assert typing_.detect(buf.getvalue()) == DOCX


def test_widening_the_trigger_does_not_widen_what_earns_a_document_verdict():
    """Inspection now runs on any zip, so confirm strictness still decides.

    A malformed package must still come back as an archive even though it now
    reaches the inspector by a different route.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[trash]/0000.dat", b"\x00" * 64)  # magic sees octet-stream
        zf.writestr("[Content_Types].xml", "<Types/>")
        zf.writestr("word/document.xml", "<document/>")  # no _rels/.rels
    assert typing_.detect(buf.getvalue()) == "application/zip"
